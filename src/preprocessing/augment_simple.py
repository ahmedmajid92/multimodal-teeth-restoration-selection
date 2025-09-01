"""
Simple, label-free augmentation:
- Takes a flat folder of images (e.g., data/processed/images)
- Creates N augmented images per source into output_dir
- Filenames: <stem>_<k>.jpg, e.g. 1.jpg -> 1_1.jpg ... 1_10.jpg
"""
from __future__ import annotations
from pathlib import Path
import os, random
import cv2
import numpy as np
import albumentations as A
from typing import Iterable

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# ---------------- helpers (version-safe transforms) ----------------
def T_Affine(p=0.85, translate=0.10, scale=0.10, rotate=25, border=cv2.BORDER_REFLECT_101):
    """Prefer A.Affine (v2). Fallback to ShiftScaleRotate (v1)."""
    if hasattr(A, "Affine"):
        return A.Affine(
            translate_percent={"x": (-translate, translate), "y": (-translate, translate)},
            scale=(1 - scale, 1 + scale),
            rotate=(-rotate, rotate),
            fit_output=False,
            mode=border,
            p=p,
        )
    # fallback (older albumentations)
    return A.ShiftScaleRotate(shift_limit=translate, scale_limit=scale,
                              rotate_limit=rotate, border_mode=border, p=p)

def T_GaussianNoise(p=0.35):
    """Use GaussianNoise if available; otherwise disable (cleaner than warnings)."""
    if hasattr(A, "GaussianNoise"):
        try:
            return A.GaussianNoise(var_limit=(5.0, 25.0), p=p)
        except TypeError:
            # Some builds use sigma_limit
            return A.GaussianNoise(sigma_limit=(5.0, 25.0), p=p)
    if hasattr(A, "GaussNoise"):
        try:
            return A.GaussNoise(var_limit=(5.0, 25.0), p=p)
        except TypeError:
            return A.GaussNoise(var_limit=(5, 25), p=p)
    # no-op if not present
    return A.NoOp(p=0.0)

def T_Elastic(alpha=20, sigma=4, p=0.20):
    """v2 no longer needs alpha_affine; keep parameters mild."""
    try:
        return A.ElasticTransform(alpha=alpha, sigma=sigma, p=p)
    except TypeError:
        # Older signature supports alpha_affine
        return A.ElasticTransform(alpha=alpha, sigma=sigma, alpha_affine=0, p=p)

def T_Cutout(num=1, max_frac=0.12, size=512, p=0.20):
    """Use Cutout (widely supported) instead of CoarseDropout to avoid warnings."""
    max_h = int(max_frac * size)
    max_w = int(max_frac * size)
    if hasattr(A, "Cutout"):
        return A.Cutout(num_holes=num, max_h_size=max_h, max_w_size=max_w, fill_value=0, p=p)
    # Fallback to CoarseDropout if Cutout is missing
    try:
        return A.CoarseDropout(max_holes=num, max_height=max_h, max_width=max_w, p=p)
    except TypeError:
        return A.NoOp(p=0.0)

def T_MotionBlur(limit=3, p=0.20, use_blur=True):
    if not use_blur:
        return A.NoOp(p=0.0)
    return A.MotionBlur(blur_limit=limit, p=p)

def build_pipeline(size: int = 512, strength: str = "medium", use_blur: bool = True) -> A.Compose:
    base = [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.ShiftScaleRotate(shift_limit=0.10, scale_limit=0.10, rotate_limit=25,
                           border_mode=cv2.BORDER_REFLECT_101, p=0.85),
        A.RandomBrightnessContrast(0.2, 0.2, p=0.7),
        A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=20, val_shift_limit=10, p=0.6),
        A.GaussNoise(var_limit=(5.0, 25.0), p=0.35),
        A.Resize(size, size)
    ]
    if strength == "light":
        extra = ([A.MotionBlur(blur_limit=3, p=0.15)] if use_blur else [])
    elif strength == "strong":
        extra = (
            ([A.MotionBlur(blur_limit=5, p=0.25)] if use_blur else []) +
            [A.ElasticTransform(alpha=28, sigma=5, alpha_affine=6, p=0.25),
             A.GridDistortion(num_steps=5, distort_limit=0.20, p=0.20),
             A.OpticalDistortion(distort_limit=0.12, shift_limit=0.05, p=0.15),
             A.CoarseDropout(max_holes=2, max_height=int(0.12*size), max_width=int(0.12*size),
                             fill_value=None, p=0.25)]
        )
    else:  # medium
        extra = (
            ([A.MotionBlur(blur_limit=3, p=0.20)] if use_blur else []) +
            [A.ElasticTransform(alpha=20, sigma=4, alpha_affine=0, p=0.20),
             A.GridDistortion(num_steps=5, distort_limit=0.15, p=0.15),
             A.CoarseDropout(max_holes=1, max_height=int(0.10*size), max_width=int(0.10*size),
                             fill_value=None, p=0.20)]
        )
    return A.Compose(base + extra)

# ---------------- pipeline builder ----------------
def build_pipeline(size: int = 512, strength: str = "medium", use_blur: bool = True) -> A.Compose:
    base = [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        T_Affine(p=0.85, translate=0.10, scale=0.10, rotate=25),
        A.RandomBrightnessContrast(0.2, 0.2, p=0.7),
        A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=20, val_shift_limit=10, p=0.6),
        T_GaussianNoise(p=0.35),
    ]

    if strength == "light":
        extra = [
            T_MotionBlur(limit=3, p=0.15, use_blur=use_blur),
        ]
    elif strength == "strong":
        extra = [
            T_MotionBlur(limit=5, p=0.25, use_blur=use_blur),
            T_Elastic(alpha=28, sigma=5, p=0.25),
            A.GridDistortion(num_steps=5, distort_limit=0.20, p=0.20),
            A.OpticalDistortion(distort_limit=0.12, shift_limit=0.05, p=0.15),
            T_Cutout(num=2, max_frac=0.15, size=size, p=0.25),
        ]
    else:  # medium
        extra = [
            T_MotionBlur(limit=3, p=0.20, use_blur=use_blur),
            T_Elastic(alpha=20, sigma=4, p=0.20),
            A.GridDistortion(num_steps=5, distort_limit=0.15, p=0.15),
            T_Cutout(num=1, max_frac=0.10, size=size, p=0.20),
        ]

    return A.Compose(base + extra + [A.Resize(size, size)])

# ---------------- driver ----------------
def iter_images(folder: Path):
    for p in sorted(Path(folder).iterdir()):
        if p.suffix.lower() in IMG_EXTS:
            yield p

def augment_folder_fixed_multiplicity(
    input_dir: Path,
    output_dir: Path,
    multiplier: int = 10,
    size: int = 512,
    strength: str = "medium",
    seed: int = 42,
    quality: int = 95,
    use_blur: bool = True,
) -> None:
    """
    For every image in input_dir, write <multiplier> augmented images
    to output_dir with names <stem>_<k>.jpg (k starts at 1).
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rnd = random.Random(seed)
    pipe = build_pipeline(size=size, strength=strength, use_blur=use_blur)

    count_src = 0
    count_out = 0
    for src in iter_images(input_dir):
        img = cv2.imread(str(src))
        if img is None:
            continue
        count_src += 1

        # deterministic-ish variety per source
        base_seed = hash(src.stem) % (2**32 - 1)

        for k in range(1, multiplier + 1):
            s = (base_seed + k) ^ seed
            np.random.seed(s); rnd.seed(s)
            aug = pipe(image=img)["image"]
            out = output_dir / f"{src.stem}_{k}.jpg"
            cv2.imwrite(str(out), aug, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            count_out += 1

    print(f"Augmented {count_src} source images → {count_out} new files at {output_dir}")