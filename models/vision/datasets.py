# models/vision/datasets.py
import os
from typing import Optional, Tuple
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def default_transforms(img_size: int = 384, aug: bool = True):
    if aug:
        return transforms.Compose([
            transforms.Resize(int(img_size * 1.15)),
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(int(img_size * 1.15)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

def _try_paths(root: str, fname: str):
    candidates = [
        os.path.join(root, fname),
        os.path.join(root, fname.lower()),
        os.path.join(root, os.path.splitext(fname)[0] + ".jpg"),
        os.path.join(root, os.path.splitext(fname)[0] + ".JPG"),
        os.path.join(root, os.path.splitext(fname)[0] + ".jpeg"),
        os.path.join(root, os.path.splitext(fname)[0] + ".png"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(f"Image not found for {fname} under {root}")

class TeethImageDataset(Dataset):
    """
    task='hard': returns (image, class_idx int)
    task='soft': returns (image, target_prob float, weight float)
    """
    def __init__(self,
                 csv_path: str,
                 images_root: str,
                 split: str,
                 task: str = "hard",
                 img_size: int = 384,
                 aug: bool = True,
                 df_override: Optional[pd.DataFrame] = None):
        super().__init__()
        self.task = task
        self.images_root = images_root
        self.t = default_transforms(img_size, aug)
        self.df = df_override if df_override is not None else pd.read_csv(csv_path)

        # Use provided split column; you can pre-create a 'val' set or we’ll filter train/test.
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)

        # Map labels
        if self.task == "hard":
            # 0 = Direct, 1 = Indirect (as inferred from CSV)
            assert "y_majority" in self.df.columns, "y_majority not found in CSV"
        else:
            assert "p_indirect" in self.df.columns, "p_indirect not found in CSV"
            if "weight" not in self.df.columns:
                self.df["weight"] = 1.0

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        path = _try_paths(self.images_root, row["image_name"])
        img = Image.open(path).convert("RGB")
        img = self.t(img)

        if self.task == "hard":
            y = int(row["y_majority"])
            return img, y
        else:
            # Soft target is probability of INDIRECT
            y = float(row["p_indirect"])
            w = float(row.get("weight", 1.0))
            return img, torch.tensor([y], dtype=torch.float32), torch.tensor([w], dtype=torch.float32)
