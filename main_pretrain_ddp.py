#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import argparse
import datetime
import json
import numpy as np
import os
import time
from glob import glob
from pathlib import Path
from PIL import Image

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import timm
assert timm.__version__ == "0.3.2"
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
import models_mae
from engine_pretrain import train_one_epoch

class MedicalSliceDataset(Dataset):
    """讀取 data_path/*/slices/*/*/*.png 的 2D 切片"""
    def __init__(self, data_path, transform=None):
        # glob 所有 png
        self.image_paths = glob(os.path.join(data_path, '*', 'slices', '*', '*', '*.png'))
        self.transform = transform
        print(f"Loaded {len(self.image_paths)} medical slices from {data_path}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, 0  # pretrain 不用 label

def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training for medical slices (8 GPUs)', add_help=False)
    # 基本參數
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size per GPU')
    parser.add_argument('--epochs', default=800, type=int)
    parser.add_argument('--accum_iter', default=1, type=int, help='Accumulate gradient iterations')
    # 模型參數
    parser.add_argument('--model', default='mae_vit_large_patch16', type=str, metavar='MODEL')
    parser.add_argument('--input_size', default=224, type=int)
    parser.add_argument('--mask_ratio', default=0.75, type=float)
    parser.add_argument('--norm_pix_loss', action='store_true', help='Use normalized pixel loss')
    parser.set_defaults(norm_pix_loss=True)
    # 優化器參數
    parser.add_argument('--weight_decay', default=0.05, type=float)
    parser.add_argument('--lr', type=float, default=None, metavar='LR')
    parser.add_argument('--blr', default=1.5e-4, type=float, help='Base learning rate')
    parser.add_argument('--min_lr', default=0., type=float)
    parser.add_argument('--warmup_epochs', default=40, type=int)
    # 資料集參數
    parser.add_argument('--data_path', default='/home/jovyan/Desktop/GBM-OpenDataset', type=str,
                        help='Path to the dataset root directory')
    parser.add_argument('--output_dir', default='/home/jovyan/Desktop/mae/output_dir', type=str)
    parser.add_argument('--log_dir', default='/home/jovyan/Desktop/mae/log_dir', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--num_workers', default=8, type=int, help='DataLoader num_workers')
    parser.add_argument('--pin_mem', action='store_true')
    parser.set_defaults(pin_mem=True)
    # DDP 參數
    parser.add_argument('--world_size', default=8, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_url', default='env://')
    parser.add_argument('--dist_on_itp', action='store_true')
    return parser

def main(args):
    # 1) 初始化分散式
    misc.init_distributed_mode(args)
    # args.gpu 由 init_distributed_mode 填入
    torch.cuda.set_device(args.gpu)
    device = torch.device(f'cuda:{args.gpu}')

    print('Job dir:', os.path.dirname(os.path.realpath(__file__)))
    print("Arguments:\n", json.dumps(vars(args), indent=4))

    # 2) 固定 seed
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # 3) 資料增強 & Dataset / DataLoader
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset_train = MedicalSliceDataset(args.data_path, transform=transform_train)

    if args.distributed:
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train,
            num_replicas=misc.get_world_size(),
            rank=misc.get_rank(),
            shuffle=True
        )
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    log_writer = None
    if misc.is_main_process() and args.log_dir:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)

    data_loader_train = DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    print(f"Total slices: {len(dataset_train)}")

    # 4) 建立 MAE 模型 + DDP
    model = models_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)
    model.to(device)
    model_without_ddp = model

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], output_device=args.gpu, find_unused_parameters=True
        )
        model_without_ddp = model.module

    # 5) 調整 learning rate
    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256
    print(f"Base LR: {args.lr * 256 / eff_batch_size:.2e}, Actual LR: {args.lr:.2e}")
    print(f"Accumulate iter: {args.accum_iter}, Effective batch size: {eff_batch_size}")

    # 6) Optimizer + Scaler
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    loss_scaler = NativeScaler()

    # 7) Resume checkpoint
    misc.load_model(args=args, model_without_ddp=model_without_ddp,
                    optimizer=optimizer, loss_scaler=loss_scaler)

    # 8) Training loop
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )

        # checkpoint
        if args.output_dir and misc.is_main_process() and (epoch % 20 == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args,
                model=model,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                loss_scaler=loss_scaler,
                epoch=epoch
            )

        # log to file
        if args.output_dir and misc.is_main_process():
            if log_writer:
                log_writer.flush()
            stats = {f'train_{k}': v for k, v in train_stats.items()}
            stats['epoch'] = epoch
            with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                f.write(json.dumps(stats) + "\n")

    total_time = time.time() - start_time
    print('Training time', str(datetime.timedelta(seconds=int(total_time))))

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    # 確保 output/log 目錄存在
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.log_dir:
        Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    main(args)
