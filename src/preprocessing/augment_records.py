"""
Augment images AND duplicate their tabular records.

- Input images live in a flat folder: e.g., data/processed/images/1.jpg ... 422.jpg
- A spreadsheet (Excel) has one row per original image with attributes/labels.
- We generate N augmented images per original, naming them sequentially
  starting at the next integer (e.g., 423.jpg, 424.jpg, ...).
- For each new image, we append a cloned row to the table with updated IDs/names.
- Output images go to data/augmented/
- Output table is written to Excel + CSV in data/excel/

Excel mapping is flexible:
- Tries to match on one of: ["image_name", "filename", "file", "image"]
  If numeric IDs exist, also tries ["id", "image_id"] against integer(stem).
"""

from __future__ import annotations
from pathlib import Path
from typing import Iterable, Optional, Tuple, Dict, List
import re
import random
import cv2
import numpy as np
import pandas as pd
import albumentations as A

# ---------------- helpers (version-safe transforms, same as your latest) -----
def T_Affine(p=0.85, translate=0.10, scale=0.10, rotate=25, border=cv2.BORDER_REFLECT_101):
    if hasattr(A, "Affine"):
        return A.Affine(
            translate_percent={"x": (-translate, translate), "y": (-translate, translate)},
            scale=(1 - scale, 1 + scale),
            rotate=(-rotate, rotate),
            fit_output=False,
            border_mode=border,
            p=p,
        )
    return A.ShiftScaleRotate(shift_limit=translate, scale_limit=scale,
                              rotate_limit=rotate, border_mode=border, p=p)

def T_GaussianNoise(p=0.35):
    if hasattr(A, "GaussianNoise"):
        try:
            return A.GaussianNoise(var_limit=(5.0, 25.0), p=p)
        except TypeError:
            return A.GaussianNoise(sigma_limit=(5.0, 25.0), p=p)
    if hasattr(A, "GaussNoise"):
        try:
            return A.GaussNoise(noise_scale_factor=(5.0, 25.0), p=p)
        except TypeError:
            return A.GaussNoise(var_limit=(5, 25), p=p)
    return A.NoOp(p=0.0)

def T_Elastic(alpha=20, sigma=4, p=0.20):
    try:
        return A.ElasticTransform(alpha=alpha, sigma=sigma, p=p)
    except TypeError:
        return A.ElasticTransform(alpha=alpha, sigma=sigma, alpha_affine=0, p=p)

def T_Cutout(num=1, max_frac=0.12, size=512, p=0.20):
    max_h = int(max_frac * size); max_w = int(max_frac * size)
    if hasattr(A, "Cutout"):
        return A.Cutout(num_holes=num, max_h_size=max_h, max_w_size=max_w, fill_value=0, p=p)
    try:
        return A.CoarseDropout(max_holes=num, max_height=max_h, max_width=max_w, p=p)
    except TypeError:
        return A.NoOp(p=0.0)

def T_MotionBlur(limit=3, p=0.20, use_blur=True):
    if not use_blur:
        return A.NoOp(p=0.0)
    return A.MotionBlur(blur_limit=limit, p=p)

def build_pipeline(size: int = 512, strength: str = "medium", use_blur: bool = False) -> A.Compose:
    base = [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        T_Affine(p=0.85, translate=0.10, scale=0.10, rotate=25),
        A.RandomBrightnessContrast(0.2, 0.2, p=0.7),
        A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=20, val_shift_limit=10, p=0.6),
        T_GaussianNoise(p=0.25),   # slightly reduced by default
    ]
    if strength == "light":
        extra = [T_MotionBlur(limit=3, p=0.1, use_blur=use_blur)]
    elif strength == "strong":
        extra = [
            T_MotionBlur(limit=5, p=0.2, use_blur=use_blur),
            T_Elastic(alpha=28, sigma=5, p=0.25),
            A.GridDistortion(num_steps=5, distort_limit=0.20, p=0.20),
            A.OpticalDistortion(distort_limit=0.12, shift_limit=0.05, p=0.15),
            T_Cutout(num=2, max_frac=0.15, size=size, p=0.25),
        ]
    else:
        extra = [
            T_MotionBlur(limit=3, p=0.15, use_blur=use_blur),
            T_Elastic(alpha=20, sigma=4, p=0.20),
            A.GridDistortion(num_steps=5, distort_limit=0.15, p=0.15),
            T_Cutout(num=1, max_frac=0.10, size=size, p=0.20),
        ]
    return A.Compose(base + extra + [A.Resize(size, size)])

# ---------------- IO helpers -------------------------------------------------
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def iter_images(folder: Path):
    for p in sorted(Path(folder).iterdir(), key=lambda x: x.name.lower()):
        if p.suffix.lower() in IMG_EXTS:
            yield p

def numeric_stem(path: Path) -> Optional[int]:
    m = re.fullmatch(r"(\d+)", path.stem)
    return int(m.group(1)) if m else None

def max_existing_id(input_dir: Path, output_dir: Path) -> int:
    """Return the maximum numeric ID found in input and output folders (0 if none)."""
    mx = 0
    for folder in [input_dir, output_dir]:
        if not Path(folder).exists(): 
            continue
        for p in Path(folder).iterdir():
            if p.suffix.lower() in IMG_EXTS:
                n = numeric_stem(p)
                if n is not None:
                    mx = max(mx, n)
    return mx

def read_excel_any(path: Path) -> pd.DataFrame:
    # requires openpyxl (pip install openpyxl)
    return pd.read_excel(path)

def find_row_for_image(df: pd.DataFrame, src_path: Path) -> Optional[int]:
    """Find the row index for src image; try name columns and id columns."""
    name_candidates = [c for c in df.columns if str(c).lower() in {"image_name", "filename", "file", "image"}]
    id_candidates   = [c for c in df.columns if str(c).lower() in {"id", "image_id"}]

    # first try by filename string
    name_variants = {src_path.name.lower(), src_path.stem.lower(), f"{src_path.stem}.jpg"}
    for c in name_candidates:
        s = df[c].astype(str).str.lower().str.strip()
        idx = s[s.isin(name_variants)].index
        if len(idx) > 0:
            return int(idx[0])

    # then by numeric id
    stem_num = numeric_stem(src_path)
    if stem_num is not None:
        for c in id_candidates:
            try:
                col = pd.to_numeric(df[c], errors="coerce")
                idx = col[col == stem_num].index
                if len(idx) > 0:
                    return int(idx[0])
            except Exception:
                continue

    return None

def update_row_identifiers(row: pd.Series, new_id: int) -> pd.Series:
    """Clone row and update id/image_name fields if present."""
    r = row.copy()
    for c in r.index:
        lc = str(c).lower()
        if lc in {"id", "image_id"}:
            with pd.option_context('mode.chained_assignment', None):
                try:
                    r[c] = int(new_id)
                except Exception:
                    r[c] = new_id
        if lc in {"image_name", "filename", "file", "image"}:
            r[c] = f"{new_id}.jpg"
    return r

# ---------------- main driver -----------------------------------------------
def augment_with_records(
    input_dir: Path,
    output_dir: Path,
    excel_in: Path,
    excel_out: Path,
    csv_out: Path,
    multiplier: int = 10,
    size: int = 512,
    strength: str = "medium",
    seed: int = 42,
    use_blur: bool = False,
    quality: int = 95,
) -> Tuple[int, int]:
    """
    Generate `multiplier` augmented images per source, name them consecutively
    starting at next available integer, and append a cloned row for each.

    Returns: (num_sources_processed, num_augmented_written)
    """
    input_dir = Path(input_dir); output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # read spreadsheet
    df = read_excel_any(Path(excel_in))
    df_out = df.copy()

    # augmentation pipeline
    pipe = build_pipeline(size=size, strength=strength, use_blur=use_blur)
    rnd = random.Random(seed)

    # compute starting id
    start_id = max_existing_id(input_dir, output_dir) + 1
    next_id = start_id

    n_src = 0
    n_out = 0

    for src in iter_images(input_dir):
        img = cv2.imread(str(src))
        if img is None:
            continue

        # find source row in excel
        row_idx = find_row_for_image(df, src)
        if row_idx is None:
            # if no row, skip (or you could raise)
            print(f"[WARN] No table row found for {src.name}; skipping.")
            continue

        row = df.iloc[row_idx]
        n_src += 1

        # determinism per-source (optional)
        base_seed = (hash(src.stem) ^ seed) & 0xFFFFFFFF

        for k in range(multiplier):
            s = (base_seed + k) & 0xFFFFFFFF
            np.random.seed(s); rnd.seed(s)

            aug = pipe(image=img)["image"]
            out_path = output_dir / f"{next_id}.jpg"
            cv2.imwrite(str(out_path), aug, [int(cv2.IMWRITE_JPEG_QUALITY), quality])

            # append cloned row with updated identifiers
            df_out.loc[len(df_out)] = update_row_identifiers(row, next_id)
            n_out += 1
            next_id += 1

    # write outputs
    excel_out = Path(excel_out); excel_out.parent.mkdir(parents=True, exist_ok=True)
    csv_out   = Path(csv_out);   csv_out.parent.mkdir(parents=True, exist_ok=True)

    # Excel + CSV
    df_out.to_excel(excel_out, index=False)
    df_out.to_csv(csv_out, index=False)

    print(f"[DONE] Sources: {n_src}  →  New images: {n_out}  (saved to {output_dir})")
    print(f"[TABLE] Wrote Excel: {excel_out}  and CSV: {csv_out}")
    return n_src, n_out
