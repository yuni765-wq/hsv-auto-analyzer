# -*- coding: utf-8 -*-
"""
Simple folder dataset for grayscale frames.
- Expected folder: data/hsv_frames/{train,val}/*.png (128x128)
- If not grayscale, image will be converted.
"""
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class FrameFolder(Dataset):
def __init__(self, root):
self.files = sorted([p for p in Path(root).glob('*.png')])
if not self.files:
raise FileNotFoundError(f"No PNG files in {root}")
def __len__(self):
return len(self.files)
def __getitem__(self, idx):
img = Image.open(self.files[idx]).convert('L').resize((128,128))
x = np.asarray(img, dtype=np.float32) / 255.0
x = np.expand_dims(x, 0) # [1,H,W]
return torch.from_numpy(x)


def make_loaders(base_dir='data/hsv_frames', bsz=32):
train_ds = FrameFolder(Path(base_dir)/'train')
val_ds = FrameFolder(Path(base_dir)/'val')
train_dl = DataLoader(train_ds, batch_size=bsz, shuffle=True, num_workers=0)
val_dl = DataLoader(val_ds, batch_size=bsz, shuffle=False, num_workers=0)
return train_dl, val_dl