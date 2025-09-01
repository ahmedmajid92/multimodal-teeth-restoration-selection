"""
Augmentation utilities for molar images (classification: direct vs indirect).
Uses Albumentations. Works with 1-folder-per-class or CSV label mapping.

Main entry points:
- build_pipeline(strength, size) -> albumentations.Compose
- augment_image(img, pipeline)   -> np.ndarray
- augment_dataset(...)           -> expands a dataset to a target size
"""
from __future__ import annotations
from pathlib import Path
import math, random
import cv2
import numpy as np
import pandas as pd
import albumentations as A

# ------------------------ Pipelines -----------------------------------------
def build_pipeline(strength: str = "medium", size: int = 512) -> A.Compose:
    """Return an Albumentations pipeline tuned for dental photos."""
    # shared bits
    basic = [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.ShiftScaleRotate(shift_limit=0.10, scale_limit=0.10, rotate_limit=25,
                           border_mode=cv2.BORDER_REFLECT_101, p=0.8),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
        A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=20, val_shift_limit=10, p=0.6),
        A.GaussNoise(var_limit=(5.0, 25.0), p=0.35),
        A.Resize(size, size)  # keep final size consistent
    ]

    light = basic + [
        A.MotionBlur(blur_limit=3, p=0.15),
        A.CoarseDropout(max_holes=1, max_height=int(0.10*size), max_width=int(0.10*size),
                        fill_value=None, p=0.15),
    ]

    medium = basic + [
        A.MotionBlur(blur_limit=3, p=0.2),
        A.ElasticTransform(alpha=20, sigma=4, alpha_affine=0, p=0.2),
        A.GridDistortion(num_steps=5, distort_limit=0.15, p=0.15),
        A.CoarseDropout(max_holes=2, max_height=int(0.12*size), max_width=int(0.12*size),
                        fill_value=None, p=0.25),
    ]

    strong = basic + [
        A.MotionBlur(blur_limit=5, p=0.25),
        A.ElasticTransform(alpha=28, sigma=5, alpha_affine=6, p=0.25),
        A.GridDistortion(num_steps=5, distort_limit=0.25, p=0.20),
        A.OpticalDistortion(distort_limit=0.12, shift_limit=0.05, p=0.15),
        A.CoarseDropout(max_holes=3, max_height=int(0.15*size), max_width=int(0.15*size),
                        fill_value=None, p=0.30),
        A.RandomGamma(gamma_limit=(80, 120), p=0.2),
    ]

    cfg = {"light": light, "medium": medium, "strong": strong}
    return A.Compose(cfg.get(strength, medium))

def augment_image(img: np.ndarray, pipeline: A.Compose) -> np.ndarray:
    """Apply pipeline to a single image (HWC, BGR or RGB both OK)."""
    # Albumentations expects uint8
    if img.dtype != np.uint8:
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    # Ensure 3 channels
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return pipeline(image=img)["image"]

# ------------------------ Dataset expansion ---------------------------------
def _load_labels(
    input_dir: Path,
    labels_csv: Path | None,
    label_column: str = "label",
    direct_column: str | None = None,
    indirect_column: str | None = None,
) -> dict[str, str]:
    """
    Return mapping image_name -> {'direct','indirect'}.
    If labels_csv provided:
      - If direct_column/indirect_column given, compute majority vote.
      - Else, read value from label_column (expects 'direct'/'indirect').
    Else:
      - infer from subfolders 'direct'/'indirect' under input_dir.
    """
    mapping: dict[str, str] = {}
    if labels_csv:
        df = pd.read_csv(labels_csv)
        if direct_column and indirect_column:
            df = df.assign(
                _label=np.where(df[direct_column].fillna(0) >= df[indirect_column].fillna(0),
                                "direct", "indirect")
            )
            name_col = "image_name" if "image_name" in df.columns else df.columns[0]
            for _, row in df.iterrows():
                mapping[str(row[name_col])] = row["_label"]
        else:
            name_col = "image_name" if "image_name" in df.columns else df.columns[0]
            for _, row in df.iterrows():
                mapping[str(row[name_col])] = str(row[label_column]).strip().lower()
        return mapping

    # infer from folder structure
    for cls in ["direct", "indirect"]:
        for p in (input_dir / cls).glob("*.*"):
            mapping[p.name] = cls
    return mapping

def augment_dataset(
    input_dir: Path,
    output_dir: Path,
    target_total: int | None = 5000,
    per_class_target: int | None = None,
    labels_csv: Path | None = None,
    label_column: str = "label",
    direct_column: str | None = None,
    indirect_column: str | None = None,
    strength: str = "medium",
    size: int = 512,
    copy_originals: bool = True,
    seed: int = 42,
) -> None:
    """
    Expand dataset to a desired total or per-class count.
    Saves to output_dir/{direct,indirect}/...  (creates folders).
    """
    random.seed(seed); np.random.seed(seed)

    input_dir = Path(input_dir); output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "direct").mkdir(exist_ok=True)
    (output_dir / "indirect").mkdir(exist_ok=True)

    # Collect label mapping and class lists
    mapping = _load_labels(input_dir, labels_csv, label_column, direct_column, indirect_column)
    if not mapping:
        # Assume two subfolders present; build mapping from them
        mapping = _load_labels(input_dir, None)

    # Build class-wise file lists
    class_files = {"direct": [], "indirect": []}
    for p in sorted(input_dir.glob("*.*")):
        if p.name in mapping:
            class_files[mapping[p.name]].append(p)
    # Also handle folder-per-class inputs
    for cls in ["direct", "indirect"]:
        for p in sorted((input_dir / cls).glob("*.*")):
            class_files[cls].append(p)

    n_direct = len(class_files["direct"])
    n_indirect = len(class_files["indirect"])
    if n_direct == 0 and n_indirect == 0:
        raise RuntimeError("No images found. Ensure input_dir has images or class subfolders, or provide labels_csv.")

    # Work out targets
    if per_class_target is None:
        if target_total is None:
            target_total = (n_direct + n_indirect)  # no-op
        per_class_target = target_total // 2  # enforce balance for classification

    # Build augmentation pipeline
    pipeline = build_pipeline(strength=strength, size=size)

    # Optionally copy originals
    if copy_originals:
        for cls in ["direct", "indirect"]:
            for src in class_files[cls]:
                dst = output_dir / cls / src.name
                if not dst.exists():
                    img = cv2.imread(str(src))
                    if img is None: 
                        continue
                    # ensure final size
                    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
                    cv2.imwrite(str(dst), img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    # Generate augmented images until we hit per-class target
    for cls in ["direct", "indirect"]:
        files = class_files[cls]
        if not files:
            continue
        existing = len(list((output_dir / cls).glob("*.*"))) if copy_originals else 0
        need = max(0, per_class_target - existing)
        if need == 0:
            continue

        # Round-robin over the class images to generate 'need' samples
        i = 0
        gen = 0
        while gen < need:
            src = files[i % len(files)]
            img = cv2.imread(str(src))
            if img is None:
                i += 1; continue
            aug = augment_image(img, pipeline)
            stem = src.stem
            out = output_dir / cls / f"{stem}_aug_{gen:05d}.jpg"
            cv2.imwrite(str(out), aug, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            gen += 1; i += 1
