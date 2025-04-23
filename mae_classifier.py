#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import os
import json
import argparse
import numpy as np
import pandas as pd
from glob import glob
from sklearn.model_selection import StratifiedKFold
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler, Subset

# ---------------------------------------------------------------------------- #
#                       Dataset: 平均池化＋串接 features                         #
# ---------------------------------------------------------------------------- #
class PatientFeatureDataset(Dataset):
    def __init__(self, data_root, label_df, indices, modal_list, orient_list):
        """
        data_root: 根目錄, 下面有 GPxxx/{modality}/{orientation}.csv
        label_df: pandas.DataFrame with columns ['Patient ID','recurrence pattern']
        indices: 這個 fold 取用的 patient index list
        modal_list: e.g. ['flair','t1','t1c','t2']
        orient_list: e.g. ['axial','coronal','sagittal']
        """
        self.modal_list = modal_list
        self.orient_list = orient_list
        patients = label_df.loc[indices, 'Patient ID'].values
        labels   = label_df.loc[indices, 'recurrence pattern'].values.astype(int)
        self.samples = []
        for pid, lab in zip(patients, labels):
            self.samples.append((pid, lab))
        self.data_root = data_root

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pid, label = self.samples[idx]
        feats = []
        for m in self.modal_list:
            for o in self.orient_list:
                csv_p = os.path.join(self.data_root, pid, m, f"{o}.csv")
                arr   = np.loadtxt(csv_p, delimiter=',')  # shape: (num_slices, feat_dim)
                arr   = arr.mean(axis=0)                 # avg pooling over slices
                feats.append(arr)
        x = np.concatenate(feats, axis=0)  # 一維向量
        return torch.from_numpy(x).float(), torch.tensor(label, dtype=torch.long)

# ---------------------------------------------------------------------------- #
#                                  MLP 模型                                     #
# ---------------------------------------------------------------------------- #
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims=[512,256], num_classes=2, dropout=0.5):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# ---------------------------------------------------------------------------- #
#                               訓練／驗證函數                                  #
# ---------------------------------------------------------------------------- #
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds==y).sum().item()
        total += x.size(0)
    return running_loss/total, correct/total

@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        running_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds==y).sum().item()
        total += x.size(0)
    return running_loss/total, correct/total

# ---------------------------------------------------------------------------- #
#                                   主程式                                     #
# ---------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root',   required=True)
    parser.add_argument('--label_file',  required=True)
    parser.add_argument('--modalities',  default="flair,t1,t1c,t2")
    parser.add_argument('--orients',     default="axial,coronal,sagittal")
    parser.add_argument('--output_dir',  required=True)
    parser.add_argument('--log_dir',     required=True)
    parser.add_argument('--epochs',      type=int, default=50)
    parser.add_argument('--batch_size',  type=int, default=32)
    parser.add_argument('--lr',          type=float, default=1e-3)
    parser.add_argument('--wd',          type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--dropout',     type=float, default=0.5)
    parser.add_argument('--hidden_dims', type=str, default="512,256")
    parser.add_argument('--folds',       type=int, default=5)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir,    exist_ok=True)

    # ---------------- Distributed init ----------------
    dist.init_process_group(backend='nccl', init_method='env://')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    # ---------------- read labels ----------------
    df = pd.read_excel(args.label_file)
    df = df.rename(columns={'Patient ID':'Patient ID','recurrence pattern':'recurrence pattern'})
    # stratified k‑fold
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=0)

    modals = args.modalities.split(',')
    ornts  = args.orients.split(',')
    hdims  = [int(x) for x in args.hidden_dims.split(',')]

    for fold, (tr_idx, va_idx) in enumerate(skf.split(df, df['recurrence pattern'])):
        if rank==0:
            print(f"\n=== Fold {fold+1}/{args.folds} ===")

        # dataset + sampler
        ds_tr = PatientFeatureDataset(args.data_root, df, tr_idx, modals, ornts)
        ds_va = PatientFeatureDataset(args.data_root, df, va_idx, modals, ornts)

        sam_tr = DistributedSampler(ds_tr, num_replicas=world_size, rank=rank, shuffle=True)
        sam_va = DistributedSampler(ds_va, num_replicas=world_size, rank=rank, shuffle=False)

        loader_tr = DataLoader(ds_tr, batch_size=args.batch_size, sampler=sam_tr,
                               num_workers=args.num_workers, pin_memory=True)
        loader_va = DataLoader(ds_va, batch_size=args.batch_size, sampler=sam_va,
                               num_workers=args.num_workers, pin_memory=True)

        # build model
        # 輸入維度 = modalities × orients × feat_dim
        # 我們先讀第一筆算 feat_dim
        sample_x, _ = ds_tr[0]
        input_dim = sample_x.numel()
        model = MLPClassifier(input_dim, hidden_dims=hdims,
                              num_classes=2, dropout=args.dropout)
        model = model.to(device)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

        # loss & optimizer
        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

        best_va_acc = 0.0
        for ep in range(args.epochs):
            sam_tr.set_epoch(ep)
            tr_loss, tr_acc = train_epoch(model, loader_tr, criterion, optimizer, device)
            va_loss, va_acc = eval_epoch(model, loader_va, criterion, device)

            if rank==0:
                print(f"[F{fold} E{ep:03d}] tr_loss={tr_loss:.4f}, tr_acc={tr_acc:.4f} | "
                      f"va_loss={va_loss:.4f}, va_acc={va_acc:.4f}")

            # save best
            if rank==0 and va_acc>best_va_acc:
                best_va_acc = va_acc
                torch.save(model.module.state_dict(),
                           os.path.join(args.output_dir, f"fold{fold}_best.pth"))

        # fold end barrier
        dist.barrier()

    dist.destroy_process_group()

if __name__=='__main__':
    main()
