#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Modified for 8-GPU medical slice pre-training by Grok, xAI.

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
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

import timm
assert timm.__version__ == "0.3.2"
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
import models_mae
from engine_pretrain import train_one_epoch

# 自訂醫療切片資料集，依據您的資料結構讀取 PNG 切片
class MedicalSliceDataset(Dataset):
    def __init__(self, data_path, transform=None):
        # 假設資料結構: data_path/{PatientID}/slices/{modality}/{orientation}/*.png
        self.image_paths = glob(os.path.join(data_path, '*', 'slices', '*', '*', '*.png'))
        self.transform = transform
        print(f"Loaded {len(self.image_paths)} medical slices from {data_path}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, 0  # MAE 不需要標籤

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
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--num_workers', default=8, type=int)  # 調整以適配8 GPU
    parser.add_argument('--pin_mem', action='store_true')
    parser.set_defaults(pin_mem=True)

    # 分散式訓練參數
    parser.add_argument('--world_size', default=8, type=int, help='Total number of GPUs (across nodes)')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_url', default='env://')
    parser.add_argument('--dist_on_itp', action='store_true', help='Flag for using distributed training on ITP')
    
    return parser

def main(args):
    # 初始化分散式環境
    misc.init_distributed_mode(args)

    print('Job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("Arguments:\n{}".format(json.dumps(vars(args), indent=4)))

    device = torch.device(args.device)
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # 定義資料增強與預處理（根據您的醫學影像特性可進一步調整）
    transform_train = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),  # 調整為224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # 使用自訂的 MedicalSliceDataset
    dataset_train = MedicalSliceDataset(args.data_path, transform=transform_train)
    print(dataset_train)

    if args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    log_writer = None
    if misc.get_rank() == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)

    data_loader_train = DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    # 建立模型
    model = models_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)
    model.to(device)
    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256

    print("Base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("Actual lr: %.2e" % args.lr)
    print("Accumulate grad iterations: %d" % args.accum_iter)
    print("Effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )
        if args.output_dir and (epoch % 20 == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch
            )

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch}
        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == '__main__':
    args = get_args_parser().parse_args()
    # 如果未指定 dist_on_itp，則設為 False
    if not hasattr(args, "dist_on_itp"):
        args.dist_on_itp = False
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
