# models/vision/datasets.py
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset

# Try Albumentations first; fall back to torchvision
_HAS_ALBU = True
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
except Exception:
    _HAS_ALBU = False
    from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def _pil_open(path: Path) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGB")

def _build_train_aug_albu(img_size: int):
    # Stochastic, realistic augs (hard task)
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.05),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.10, rotate_limit=12,
                          border_mode=0, fill_value=0, p=0.90),  # changed value to fill_value
        A.Perspective(scale=(0.02, 0.05), keep_size=True, 
                     fit_output=True, p=0.20),  # removed pad_mode and pad_val
        A.OneOf([
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8,8), p=1.0),
            A.RandomBrightnessContrast(0.15, 0.15, p=1.0),
            A.HueSaturationValue(5, 12, 8, p=1.0),
        ], p=0.50),
        A.GaussNoise(per_channel=True, mean=0, var=15.0, p=0.20),  # changed var_limit to var
        A.MotionBlur(blur_limit=5, p=0.10),
        A.ElasticTransform(alpha=10, sigma=5, p=0.10),  # removed alpha_affine and value
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])

def _build_val_aug_albu(img_size: int):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])

def _build_train_aug_torch(img_size: int):
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.05),
        transforms.RandomAffine(degrees=12, translate=(0.05, 0.05), scale=(0.90, 1.10)),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.12, hue=0.02),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.10),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

def _build_val_aug_torch(img_size: int):
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

class TeethImageDataset(Dataset):
    """
    CSV must contain:
      - image_name (str), split ('train'|'val'|'test')
      - y_majority (int 0/1) for task='hard'
      - Optional: origin_id, aug_idx, weight, p_indirect (used by other tasks)
    """

    def __init__(self,
                 csv_path: str,
                 images_root: str,
                 split: str,
                 task: str,
                 img_size: int = 512,
                 aug: bool = True,
                 df_override: Optional[pd.DataFrame] = None):
        super().__init__()
        self.images_root = Path(images_root)
        self.split = split.lower()
        self.task = task.lower()
        self.img_size = int(img_size)
        self.aug = bool(aug)

        if df_override is not None:
            df = df_override.copy()
        else:
            df_all = pd.read_csv(csv_path)
            mask = df_all["split"].astype(str).str.lower() == self.split
            df = df_all.loc[mask].reset_index(drop=True)

        if "image_name" not in df.columns: raise KeyError("CSV missing 'image_name'")
        if "split" not in df.columns: raise KeyError("CSV missing 'split'")

        if self.task == "hard":
            if "y_majority" not in df.columns: raise KeyError("CSV missing 'y_majority' for hard task")
            df["y_majority"] = df["y_majority"].astype(int)
        else:
            raise ValueError("This dataset class is for task='hard' usage here.")

        self.df = df.reset_index(drop=True)

        if _HAS_ALBU:
            self.tf = _build_train_aug_albu(self.img_size) if (self.aug and self.split == "train") \
                      else _build_val_aug_albu(self.img_size)
        else:
            self.tf = _build_train_aug_torch(self.img_size) if (self.aug and self.split == "train") \
                      else _build_val_aug_torch(self.img_size)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        path = (self.images_root / row["image_name"]).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        if _HAS_ALBU:
            img = np.array(_pil_open(path))
            img_t = self.tf(image=img)["image"]
        else:
            img_t = _build_val_aug_torch(self.img_size)(_pil_open(path)) if self.split != "train" or not self.aug \
                    else _build_train_aug_torch(self.img_size)(_pil_open(path))

        y = torch.tensor(int(row["y_majority"]), dtype=torch.long)
        return img_t, y
