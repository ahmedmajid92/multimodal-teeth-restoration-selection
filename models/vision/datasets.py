# models/vision/datasets.py
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset

# Prefer Albumentations; fall back to torchvision transforms
_HAS_ALBU = True
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
except Exception:
    _HAS_ALBU = False
    from torchvision import transforms  # noqa: F401

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


# -----------------
# Helper functions
# -----------------
def _pil_open(path: Path) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGB")


# -----------------------
# Albumentations builders
# -----------------------
def _build_train_aug_albu_hard(img_size: int):
    """Stronger but realistic augs for HARD labels."""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.05),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.10, rotate_limit=12,
                           border_mode=0, value=(0, 0, 0), p=0.90),
        A.Perspective(scale=(0.02, 0.05), p=0.20),
        A.OneOf([
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=1.0),
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=12, val_shift_limit=8, p=1.0),
        ], p=0.50),
        A.GaussNoise(var_limit=(5.0, 15.0), mean=0, per_channel=True, p=0.20),
        A.MotionBlur(blur_limit=5, p=0.10),
        # keep elastic very mild; it can be harmful for calibration (we use it only for HARD)
        A.ElasticTransform(alpha=10, sigma=5, alpha_affine=0, border_mode=0, value=(0, 0, 0), p=0.10),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def _build_train_aug_albu_soft(img_size: int):
    """Gentler augs for SOFT labels (focus on calibration/stability)."""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.03),
        A.ShiftScaleRotate(shift_limit=0.04, scale_limit=0.08, rotate_limit=8,
                           border_mode=0, value=(0, 0, 0), p=0.85),
        # Skip perspective/elastic by default to avoid label drift for soft targets
        A.RandomBrightnessContrast(brightness_limit=0.10, contrast_limit=0.10, p=0.50),
        A.HueSaturationValue(hue_shift_limit=4, sat_shift_limit=10, val_shift_limit=6, p=0.30),
        A.GaussNoise(var_limit=(4.0, 10.0), mean=0, per_channel=True, p=0.10),
        A.MotionBlur(blur_limit=3, p=0.05),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def _build_val_aug_albu(img_size: int):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


# -----------------------
# Torchvision builders
# -----------------------
def _build_train_aug_torch_hard(img_size: int):
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


def _build_train_aug_torch_soft(img_size: int):
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.03),
        transforms.RandomAffine(degrees=8, translate=(0.04, 0.04), scale=(0.92, 1.08)),
        transforms.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.10, hue=0.02),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.05),
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


# ---------------
# Dataset class
# ---------------
class TeethImageDataset(Dataset):
    """
    Supports two tasks:
      - task='hard': returns (image, label_long)
        * expects column: y_majority ∈ {0,1}
      - task='soft': returns (image, prob_float[1], weight_float[1])
        * expects column: p_indirect ∈ [0,1]
        * optional column: weight (defaults to 1.0)

    Common required columns:
      - image_name (str)
      - split ∈ {'train','val','test'}
    Optional leakage-control columns:
      - origin_id, aug_idx (not used here but produced by your augmentation pipeline)
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
        self.split = str(split).lower()
        self.task = str(task).lower()
        self.img_size = int(img_size)
        self.aug = bool(aug)

        # Load frame
        if df_override is not None:
            df = df_override.copy()
        else:
            df_all = pd.read_csv(csv_path)
            mask = df_all["split"].astype(str).str.lower() == self.split
            df = df_all.loc[mask].reset_index(drop=True)

        if "image_name" not in df.columns:
            raise KeyError("CSV missing 'image_name'")
        if "split" not in df.columns:
            raise KeyError("CSV missing 'split'")

        # Task-specific checks
        if self.task == "hard":
            if "y_majority" not in df.columns:
                raise KeyError("CSV missing 'y_majority' for task='hard'")
            df["y_majority"] = df["y_majority"].astype(int)

        elif self.task == "soft":
            if "p_indirect" not in df.columns:
                raise KeyError("CSV missing 'p_indirect' for task='soft'")
            df["p_indirect"] = df["p_indirect"].astype(float)
            if "weight" not in df.columns:
                df["weight"] = 1.0
            df["weight"] = df["weight"].astype(float)

        else:
            raise ValueError(f"Unknown task: {self.task}")

        self.df = df.reset_index(drop=True)

        # Build transforms
        if _HAS_ALBU:
            if self.aug and self.split == "train":
                self.tf = _build_train_aug_albu_hard(self.img_size) if self.task == "hard" \
                          else _build_train_aug_albu_soft(self.img_size)
            else:
                self.tf = _build_val_aug_albu(self.img_size)
        else:
            if self.aug and self.split == "train":
                self.tf = _build_train_aug_torch_hard(self.img_size) if self.task == "hard" \
                          else _build_train_aug_torch_soft(self.img_size)
            else:
                self.tf = _build_val_aug_torch(self.img_size)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        path = (self.images_root / row["image_name"]).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        if _HAS_ALBU:
            img_np = np.array(_pil_open(path))  # HWC uint8 RGB
            img_t = self.tf(image=img_np)["image"]  # Tensor CHW
        else:
            # Build once per item to keep it simple; transforms are lightweight
            from torchvision import transforms
            pil = _pil_open(path)
            img_t = self.tf(pil)

        if self.task == "hard":
            y = torch.tensor(int(row["y_majority"]), dtype=torch.long)
            return img_t, y

        # soft
        p = float(row["p_indirect"])
        w = float(row.get("weight", 1.0))
        y_t = torch.tensor([p], dtype=torch.float32)   # [1]
        w_t = torch.tensor([w], dtype=torch.float32)   # [1]
        return img_t, y_t, w_t
