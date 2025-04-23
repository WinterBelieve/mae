#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
從 2D‑slice 資料夾讀取 PNG，利用預訓練 MAE 模型提取 patch‑embed 特徵（平均 pooling），
並分別把每個 Patient／orientation／modality 的特徵逐 row 存成 CSV。
"""

import argparse
import os
import time
import datetime
from glob import glob

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from tqdm import tqdm

import timm
assert timm.__version__ == "0.3.2"
import models_mae
from torchvision import transforms


class SliceDataset(Dataset):
    def __init__(self, root_dir, transform):
        # root_dir/*/*/*/*.png -> patient/orientation/modality/*.png
        self.paths = sorted(glob(os.path.join(root_dir, '*', '*', '*', '*.png')))
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        img = Image.open(p).convert('RGB')
        img = self.transform(img)
        # 從路徑拆出 patient, orientation, modality
        parts = p.split(os.sep)
        patient    = parts[-4]
        orientation= parts[-3]
        modality   = parts[-2]
        info = {
            'patient': patient,
            'orientation': orientation,
            'modality': modality,
        }
        return img, info


def my_collate(batch):
    """
    把 batch: List[(img_tensor, info_dict)] 拆成
    imgs: Tensor[B,C,H,W] 和 infos: List[Dict]
    """
    imgs  = torch.stack([b[0] for b in batch], dim=0)
    infos = [b[1] for b in batch]
    return imgs, infos


def get_args_parser():
    parser = argparse.ArgumentParser('MAE 特徵提取', add_help=True)
    parser.add_argument('--data_dir',     type=str, required=True,
                        help='根目錄，底下是 Patient/axial/t1/*.png')
    parser.add_argument('--output_dir',   type=str, required=True,
                        help='特徵 CSV 輸出根目錄')
    parser.add_argument('--ckpt_path',    type=str, required=True,
                        help='預訓練 checkpoint 檔')
    parser.add_argument('--model',        type=str, default='mae_vit_large_patch16')
    parser.add_argument('--batch_size',   type=int, default=32)
    parser.add_argument('--num_workers',  type=int, default=4)
    parser.add_argument('--seed',         type=int, default=0)
    return parser


def main():
    args = get_args_parser().parse_args()

    # 種子、cudnn 加速
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 圖像預處理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Dataset & DataLoader
    dataset = SliceDataset(args.data_dir, transform)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=my_collate
    )
    print(f'Loaded {len(dataset)} slices from {args.data_dir}')

    # 載入 MAE 模型 + DataParallel
    print('Loading pretrained MAE model ...')
    model = models_mae.__dict__[args.model](norm_pix_loss=True)
    ckpt = torch.load(args.ckpt_path, map_location='cpu')
    sd = ckpt.get('model', ckpt)
    model.load_state_dict(sd, strict=False)
    model = nn.DataParallel(model)
    model.to(device)
    model.eval()

    # 記錄已經開檔的 (pat, ori, mod)
    written = set()

    start_time = time.time()
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Extracting')
        for imgs, infos in pbar:
            imgs = imgs.to(device)
            # forward_encoder
            out = model.module.forward_encoder(imgs, mask_ratio=0)
            feats = out[0] if isinstance(out, (tuple, list)) else out
            # (B, P, D) -> (B, D)
            feats = feats.mean(dim=1)
            feats = feats.cpu().numpy()

            # 依病患／orientation／modality 逐 row 寫入 CSV
            for feat, info in zip(feats, infos):
                pt = info['patient']
                ori= info['orientation']
                md = info['modality']
                out_dir = os.path.join(args.output_dir, pt, ori)
                os.makedirs(out_dir, exist_ok=True)
                csv_path = os.path.join(out_dir, f'{md}.csv')

                key = (pt, ori, md)
                if key not in written:
                    # 第一次寫，若舊檔在就先刪
                    if os.path.exists(csv_path):
                        os.remove(csv_path)
                    written.add(key)

                # append 這一 row
                line = ','.join(f'{x:.6f}' for x in feat.tolist())
                with open(csv_path, 'a') as f:
                    f.write(line + '\n')

    elapsed = time.time() - start_time
    print(f'Done! 耗時 {str(datetime.timedelta(seconds=int(elapsed)))}')


if __name__ == '__main__':
    main()
