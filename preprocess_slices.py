#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocess: 從多個病患資料夾讀取 NIfTI(.nii.gz)，沿 axial/coronal/sagittal 三個方向
抽出所有 2D 切片，存成 PNG 到 output_dir/{patient_id}/{modality}/{orientation}/slice_####.png
"""
import argparse, os
from glob import glob
from pathlib import Path
import nibabel as nib
import numpy as np
from PIL import Image
from tqdm import tqdm

def get_args_parser():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input_dirs", type=str,
        help="逗號分隔的原始 NIfTI 根目錄，例如 Patients_RPG,Patients"
    )
    p.add_argument(
        "--output_dir", type=str,
        help="PNG 切片輸出根目錄"
    )
    return p

def normalize_uint8(x):
    x = x.astype(np.float32)
    x = (x - x.min()) / (x.ptp() + 1e-8) * 255.0
    return x.astype(np.uint8)

def main():
    args = get_args_parser().parse_args()
    output_root = Path(args.output_dir)
    input_dirs = [d.strip() for d in args.input_dirs.split(",")]

    for root in input_dirs:
        for nii_path in sorted(glob(f"{root}/*/*.nii.gz")):
            patient = Path(nii_path).parent.name
            fname = Path(nii_path).name.lower()
            # 判斷 modality
            if "t1c" in fname or "t1ce" in fname or "t1gd" in fname:
                mod = "t1c"
            elif "flair" in fname:
                mod = "flair"
            elif "t2" in fname:
                mod = "t2"
            elif "t1" in fname:
                mod = "t1"
            else:
                continue
            # load volume
            vol = nib.load(nii_path).get_fdata()
            H, W, D = vol.shape[:3]
            for orientation, axis in [("axial", 2), ("coronal", 1), ("sagittal", 0)]:
                out_dir = output_root / patient / mod / orientation
                out_dir.mkdir(parents=True, exist_ok=True)
                n_slices = vol.shape[axis]
                desc = f"{patient} {mod} {orientation}"
                for i in tqdm(range(n_slices), desc=desc, unit="slice"):
                    if axis == 2:
                        slice2d = vol[:, :, i]
                    elif axis == 1:
                        slice2d = vol[:, i, :]
                    else:
                        slice2d = vol[i, :, :]
                    img = normalize_uint8(slice2d)
                    im = Image.fromarray(img).convert("RGB")
                    im.save(out_dir / f"slice_{i:04d}.png")

if __name__ == "__main__":
    main()
