# src/preprocessing/augment_records.py
import argparse
import random
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter
from sklearn.model_selection import GroupShuffleSplit

# Optional strong augmenter (Albumentations)
try:
    import albumentations as A
    import cv2
    _HAS_ALBU = True
except Exception:
    _HAS_ALBU = False


# =========
# Utilities
# =========

def seed_all(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def to_jpg_name(name: str) -> str:
    stem = Path(name).stem
    return f"{stem}.jpg"

def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    elif path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported table format: {path}")

def write_table(df: pd.DataFrame, out_csv: Path, out_xlsx: Path):
    ensure_dir(out_csv.parent)
    ensure_dir(out_xlsx.parent)
    df.to_csv(out_csv, index=False)
    df.to_excel(out_xlsx, index=False)

def infer_image_id(row: pd.Series) -> int:
    if "image_id" in row and pd.notna(row["image_id"]):
        return int(row["image_id"])
    if "image_name" in row and isinstance(row["image_name"], str):
        s = Path(row["image_name"]).stem
        try:
            return int(s)
        except Exception:
            pass
    return int(row.name) + 1

def normalize_filename(src_path: Path, dst_path: Path):
    with Image.open(src_path) as im:
        im = im.convert("RGB")
        im.save(dst_path, format="JPEG", quality=95)

def pil_to_np(img: Image.Image):
    return np.array(img)[:, :, ::-1]  # RGB -> BGR for cv2

def np_to_pil(arr: np.ndarray):
    return Image.fromarray(arr[:, :, ::-1])  # BGR -> RGB


# ==========================
# Augmentation configurations
# ==========================

# ---- Recommended "legacy" random pipeline (stochastic, realistic) ----
def _legacy_compose(img_size: int = 512, use_motion_blur: bool = True):
    """
    Strong but realistic, stochastic per-image pipeline for OFFLINE augmentation.
    Matches the recommended per-epoch recipe (small geometry + modest photometrics).
    """
    return A.Compose([
        # Geometry (small-range)
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.05),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.10, rotate_limit=12,
                           border_mode=cv2.BORDER_CONSTANT, value=(0, 0, 0), p=0.90),
        # Very light perspective wobble
        A.Perspective(scale=(0.02, 0.05), keep_size=True, pad_mode=cv2.BORDER_CONSTANT,
                      pad_val=(0, 0, 0), p=0.20),
        # Photometrics (modest)
        A.OneOf([
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=1.0),
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=12, val_shift_limit=8, p=1.0),
        ], p=0.50),
        # Light noise / blur
        A.GaussNoise(var_limit=(5.0, 15.0), mean=0, per_channel=True, p=0.20),
        (A.MotionBlur(blur_limit=(3, 5), p=0.10) if use_motion_blur else A.GaussianBlur(blur_limit=(3, 3), p=0.10)),
        # Mild elastic (kept small to avoid unrealistic warps)
        A.ElasticTransform(alpha=10, sigma=5, alpha_affine=0,
                           border_mode=cv2.BORDER_CONSTANT, value=(0, 0, 0), p=0.10),
        # Optional tiny occluders (very conservative)
        A.CoarseDropout(max_holes=1, max_height=max(1, img_size // 24), max_width=max(1, img_size // 24),
                        fill_value=(0, 0, 0), p=0.10)
    ])

def _apply_albu(aug, img_np, rng):
    """Apply an Albumentations transform with reproducible randomness from rng."""
    state = np.random.get_state()
    np.random.seed(rng.randrange(0, 2**31 - 1))
    out = aug(image=img_np)["image"]
    np.random.set_state(state)
    return out

def augment_once_legacy(img: Image.Image, rng: random.Random, img_size: int = 512, use_motion_blur: bool = True) -> Image.Image:
    if not _HAS_ALBU:
        # Fallback to simple PIL-based approximation
        return augment_once_simple(img, rng)
    aug = _legacy_compose(img_size, use_motion_blur)
    im_np = pil_to_np(img)
    out = _apply_albu(aug, im_np, rng)
    if out.shape[:2] != im_np.shape[:2]:
        out = cv2.resize(out, (im_np.shape[1], im_np.shape[0]), interpolation=cv2.INTER_LINEAR)
    return np_to_pil(np.clip(out, 0, 255).astype(np.uint8))

# ---- Lighter PIL-only pipeline (fallback / debugging) ----
def augment_once_simple(img: Image.Image, rng: random.Random) -> Image.Image:
    w, h = img.size
    out = img
    deg = rng.choice([-10, -7, -5, -3, 0, 3, 5, 7, 10])
    out = out.rotate(deg, resample=Image.BILINEAR, expand=False, fillcolor=(0, 0, 0))
    if rng.random() < 0.5:  out = out.transpose(Image.FLIP_LEFT_RIGHT)
    if rng.random() < 0.05: out = out.transpose(Image.FLIP_TOP_BOTTOM)
    b = 0.90 + 0.20 * rng.random()
    c = 0.90 + 0.20 * rng.random()
    out = ImageEnhance.Brightness(out).enhance(b)
    out = ImageEnhance.Contrast(out).enhance(c)
    if rng.random() < 0.15:
        out = out.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.3, 0.8)))
    elif rng.random() < 0.15:
        out = ImageEnhance.Sharpness(out).enhance(1.0 + rng.uniform(0.3, 0.8))
    if rng.random() < 0.5:
        pad = rng.randint(2, 10)
        left, top = pad, pad
        right, bottom = w - pad, h - pad
        if right > left and bottom > top:
            out = out.crop((left, top, right, bottom)).resize((w, h), Image.BILINEAR)
    return out

# ---- Exact 10-methods generator (optional) ----
def fixed_ten_variants(img: Image.Image, rng: random.Random, img_size: int = 512, no_blur: bool = False) -> List[Image.Image]:
    """Return exactly 10 distinct-method variants (kept for users who want determinism)."""
    if not _HAS_ALBU:
        # crude PIL approximations
        w, h = img.size
        out = []
        out.append(img.transpose(Image.FLIP_LEFT_RIGHT))
        out.append(img.transpose(Image.FLIP_TOP_BOTTOM))
        tx = rng.randint(int(0.03*w), int(0.07*w)) * rng.choice([-1, 1])
        ty = rng.randint(int(0.03*h), int(0.07*h)) * rng.choice([-1, 1])
        out.append(img.transform((w, h), Image.AFFINE, (1, 0, tx, 0, 1, ty), resample=Image.BILINEAR, fillcolor=(0,0,0)))
        scale = 1.0 + rng.uniform(-0.10, 0.10); sw, sh = max(1,int(w*scale)), max(1,int(h*scale))
        canvas = Image.new("RGB", (w, h), (0,0,0)); canvas.paste(img.resize((sw, sh), Image.BILINEAR), ((w-sw)//2,(h-sh)//2)); out.append(canvas)
        out.append(img.rotate(rng.uniform(-25, 25), resample=Image.BILINEAR, expand=False, fillcolor=(0,0,0)))
        bc = ImageEnhance.Brightness(img).enhance(0.9 + 0.2*rng.random()); bc = ImageEnhance.Contrast(bc).enhance(0.9 + 0.2*rng.random()); out.append(bc)
        hsv = ImageEnhance.Color(img).enhance(0.9 + 0.2*rng.random()); hsv = ImageEnhance.Brightness(hsv).enhance(0.95 + 0.1*rng.random()); out.append(hsv)
        arr = np.array(img).astype(np.float32); arr = np.clip(arr + np.random.randn(*arr.shape)*8.0, 0, 255).astype(np.uint8); out.append(Image.fromarray(arr))
        out.append(img.filter(ImageFilter.GaussianBlur(radius=0.8 if no_blur else 1.2)))
        pad = rng.randint(2, 6); out.append(img.crop((pad,pad,w-pad,h-pad)).resize((w,h), Image.BILINEAR))
        return out

    img_np = pil_to_np(img); H, W = img_np.shape[:2]; border=cv2.BORDER_CONSTANT; fill=(0,0,0)
    def _apply(aug): return _apply_albu(aug, img_np, rng)

    outs = []
    outs.append(_apply(A.Compose([A.HorizontalFlip(p=1.0)])))
    outs.append(_apply(A.Compose([A.VerticalFlip(p=1.0)])))
    tx = float(rng.uniform(0.03, 0.07)) * rng.choice([-1,1]); ty = float(rng.uniform(0.03,0.07)) * rng.choice([-1,1])
    outs.append(_apply(A.Compose([A.Affine(translate_percent={"x":tx,"y":ty}, scale=1.0, rotate=0, cval=fill, mode=border, p=1.0)])))
    scale = float(rng.uniform(0.9,1.1))
    outs.append(_apply(A.Compose([A.Affine(scale=scale, rotate=0, translate_percent={"x":0,"y":0}, cval=fill, mode=border, p=1.0)])))
    rot = float(rng.uniform(-25,25))
    outs.append(_apply(A.Compose([A.Affine(rotate=rot, scale=1.0, translate_percent={"x":0,"y":0}, cval=fill, mode=border, p=1.0)])))
    outs.append(_apply(A.Compose([A.RandomBrightnessContrast(0.15,0.15,p=1.0)])))
    outs.append(_apply(A.Compose([A.HueSaturationValue(5,12,8,p=1.0)])))
    outs.append(_apply(A.Compose([A.GaussNoise(var_limit=(5.0,15.0), p=1.0)])))
    outs.append(_apply(A.Compose([A.GaussianBlur(blur_limit=(3,3),p=1.0)]) if no_blur else A.Compose([A.MotionBlur(blur_limit=(3,5),p=1.0)])))
    outs.append(_apply(A.Compose([A.ElasticTransform(alpha=10, sigma=5, alpha_affine=0, border_mode=border, value=fill, p=1.0)])))

    pil_outs = []
    for v in outs:
        if v.shape[:2] != img_np.shape[:2]:
            v = cv2.resize(v, (W, H), interpolation=cv2.INTER_LINEAR)
        pil_outs.append(np_to_pil(np.clip(v, 0, 255).astype(np.uint8)))
    return pil_outs


def get_augmenter(preset: str, img_size: int = 512, use_motion_blur: bool = True):
    """
    Returns a callable(img, rng) -> PIL.Image for 'legacy'/'simple',
    'TEN' token for exact 10 methods, or identity for 'none'.
    """
    preset = (preset or "legacy").lower()
    if preset == "none":
        return lambda img, rng: img
    if preset == "legacy":
        return lambda img, rng: augment_once_legacy(img, rng, img_size, use_motion_blur)
    if preset == "ten":
        return "TEN"
    return augment_once_simple


# ==============
# Core pipeline
# ==============

def build_augmented_table(
    df_orig: pd.DataFrame,
    images_src: Path,
    images_dst: Path,
    num_aug_per_image: int,
    start_id: int = None,
    copy_originals: bool = True,
    make_val: bool = True,
    val_frac: float = 0.12,
    seed: int = 42,
    aug_preset: str = "legacy",
    img_size_for_aug: int = 512,
    no_blur: bool = False,
) -> pd.DataFrame:
    """
    Leakage-safe offline augmentation:
      - Adds origin_id, aug_idx (0 original, 1..N children).
      - Inherits parent's split to children.
      - Optional grouped 'val' split on TRAIN via origin_id.
      - Writes images to images_dst with normalized .jpg names.
    """
    seed_all(seed)
    ensure_dir(images_dst)
    df = df_orig.copy()

    if "image_name" not in df.columns:
        if "image_id" in df.columns:
            df["image_name"] = df["image_id"].map(lambda x: f"{int(x)}.jpg")
        else:
            raise ValueError("Input must contain 'image_name' or 'image_id'.")

    if "image_id" not in df.columns:
        df["image_id"] = df.apply(infer_image_id, axis=1).astype(int)

    df["image_name"] = df["image_name"].astype(str).map(lambda s: to_jpg_name(s.lower()))

    if "split" not in df.columns:
        print("No 'split' column; creating grouped train/test split (80/20) by originals.")
        groups = df["image_id"].values
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        tr_idx, te_idx = next(gss.split(df, groups=groups))
        df.loc[df.index[tr_idx], "split"] = "train"
        df.loc[df.index[te_idx], "split"] = "test"

    df["origin_id"] = df["image_id"].astype(int)
    df["aug_idx"] = 0

    if copy_originals:
        print("Copying/normalizing originals into augmented folder as .jpg…")
        for _, r in df.iterrows():
            src = (images_src / r["image_name"]).resolve()
            dst = (images_dst / to_jpg_name(r["image_name"])).resolve()
            if not dst.exists():
                if src.exists():
                    normalize_filename(src, dst)
                else:
                    raise FileNotFoundError(f"Original image not found: {src}")

    if start_id is None:
        start_id = int(df["image_id"].max()) + 1

    augmenter = get_augmenter(aug_preset, img_size_for_aug, use_motion_blur=(not no_blur))

    aug_rows: List[pd.Series] = []
    next_id = start_id

    for _, row in df.iterrows():
        origin_id = int(row["origin_id"])
        split = str(row["split"]).lower()
        img_name = row["image_name"]
        src_path = images_dst / img_name
        if not src_path.exists():
            alt = images_src / img_name
            if not alt.exists():
                raise FileNotFoundError(f"Source image not found: {img_name}")
            src_path = alt

        rng = random.Random(seed * 1000003 + origin_id)

        with Image.open(src_path) as im_src:
            im_src = im_src.convert("RGB")

            if augmenter == "TEN":
                children = fixed_ten_variants(im_src, rng, img_size_for_aug, no_blur=no_blur)
                N = min(num_aug_per_image, 10)
                for j in range(1, N + 1):
                    child = children[j - 1]
                    child_id = next_id
                    child_name = f"{child_id}.jpg"
                    child.save(images_dst / child_name, format="JPEG", quality=95)

                    new_row = row.copy()
                    new_row["image_id"] = child_id
                    new_row["image_name"] = child_name
                    new_row["origin_id"] = origin_id
                    new_row["aug_idx"] = j
                    new_row["split"] = split
                    aug_rows.append(new_row)
                    next_id += 1

                # extras (if requested >10): sample random legacy/simple
                extras = num_aug_per_image - N
                for j in range(N + 1, N + 1 + max(0, extras)):
                    aug_img = augment_once_legacy(im_src, rng, img_size_for_aug, use_motion_blur=(not no_blur)) if _HAS_ALBU else augment_once_simple(im_src, rng)
                    child_id = next_id
                    child_name = f"{child_id}.jpg"
                    aug_img.save(images_dst / child_name, format="JPEG", quality=95)
                    new_row = row.copy()
                    new_row["image_id"] = child_id
                    new_row["image_name"] = child_name
                    new_row["origin_id"] = origin_id
                    new_row["aug_idx"] = j
                    new_row["split"] = split
                    aug_rows.append(new_row)
                    next_id += 1

            else:
                # recommended: 'legacy' randomized per child
                for j in range(1, num_aug_per_image + 1):
                    child = augmenter(im_src, rng)
                    child_id = next_id
                    child_name = f"{child_id}.jpg"
                    child.save(images_dst / child_name, format="JPEG", quality=95)

                    new_row = row.copy()
                    new_row["image_id"] = child_id
                    new_row["image_name"] = child_name
                    new_row["origin_id"] = origin_id
                    new_row["aug_idx"] = j
                    new_row["split"] = split
                    aug_rows.append(new_row)
                    next_id += 1

    df_aug = pd.concat([df, pd.DataFrame(aug_rows)], ignore_index=True)

    if make_val:
        print(f"Creating grouped 'val' split inside TRAIN (val_frac={val_frac}) using origin_id…")
        tr_mask = df_aug["split"].astype(str).str.lower().eq("train")
        fam_df = df_aug.loc[tr_mask, ["origin_id"]].drop_duplicates().reset_index(drop=True)
        if len(fam_df) > 0:
            gss = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
            tr_idx, va_idx = next(gss.split(fam_df, groups=fam_df["origin_id"].values))
            val_fams = set(fam_df.iloc[va_idx]["origin_id"].tolist())
            is_val = df_aug["origin_id"].isin(val_fams) & tr_mask
            df_aug.loc[is_val, "split"] = "val"
        else:
            print("Warning: No training families found to split into validation.")

    lead = ["image_id", "image_name", "origin_id", "aug_idx", "split"]
    rest = [c for c in df_aug.columns if c not in lead]
    df_aug = df_aug[lead + rest]

    n_total = len(df_aug)
    n_train = int((df_aug["split"].astype(str).str.lower() == "train").sum())
    n_val   = int((df_aug["split"].astype(str).str.lower() == "val").sum())
    n_test  = int((df_aug["split"].astype(str).str.lower() == "test").sum())
    print(f"Rows: total={n_total} | train={n_train} | val={n_val} | test={n_test}")

    return df_aug


# ==================
# CLI & compat shim
# ==================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-table", default="data/excel/data_dl.xlsx")
    ap.add_argument("--images-src", default="data/processed/images")
    ap.add_argument("--images-dst", default="data/augmented")
    ap.add_argument("--num-aug-per-image", type=int, default=10)
    ap.add_argument("--start-id", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--make-val", action="store_true")
    ap.add_argument("--val-frac", type=float, default=0.12)
    ap.add_argument("--aug-preset", choices=["legacy", "simple", "ten", "none"], default="legacy",
                    help="Recommended: legacy (random realistic). Others kept for compatibility.")
    ap.add_argument("--img-size-for-aug", type=int, default=512)
    ap.add_argument("--no-blur", action="store_true", help="Disable MotionBlur in legacy/ten.")
    ap.add_argument("--out-csv", default="data/excel/data_dl_augmented.csv")
    ap.add_argument("--out-xlsx", default="data/excel/data_dl_augmented.xlsx")
    args = ap.parse_args()

    seed_all(args.seed)

    if args.aug_preset in ("legacy", "ten") and not _HAS_ALBU:
        print("⚠️  Albumentations not found. Install:")
        print("    pip install albumentations opencv-python-headless")
        if args.aug_preset == "legacy":
            print("    Falling back to 'simple' augmentation.")
            args.aug_preset = "simple"
        else:
            print("    Proceeding with PIL approximations for 'ten'.")

    df_orig = read_table(Path(args.input_table))
    df_aug = build_augmented_table(
        df_orig=df_orig,
        images_src=Path(args.images_src),
        images_dst=Path(args.images_dst),
        num_aug_per_image=args.num_aug_per_image,
        start_id=args.start_id,
        copy_originals=True,
        make_val=args.make_val,
        val_frac=args.val_frac,
        seed=args.seed,
        aug_preset=args.aug_preset,
        img_size_for_aug=args.img_size_for_aug,
        no_blur=args.no_blur,
    )

    write_table(df_aug, Path(args.out_csv), Path(args.out_xlsx))
    print("\n✅ Done.")
    print(f"   Saved CSV : {args.out_csv}")
    print(f"   Saved XLSX: {args.out_xlsx}")
    print(f"   Images in : {args.images_dst}")

# Backwards-compat callable for any existing runner
def augment_records(*, input_table, images_src, images_dst, num_aug_per_image,
                    start_id=None, seed=42, make_val=False, val_frac=0.12,
                    out_csv="data/excel/data_dl_augmented.csv",
                    out_xlsx="data/excel/data_dl_augmented.xlsx",
                    aug_preset="legacy", img_size_for_aug=512, no_blur=False):
    seed_all(seed)
    if aug_preset in ("legacy", "ten") and not _HAS_ALBU:
        print("⚠️  Albumentations not found. "
              + ("Using PIL approximations for 'ten'." if aug_preset=="ten" else "Falling back to 'simple'."))
        if aug_preset == "legacy":
            aug_preset = "simple"
    df_orig = read_table(Path(input_table))
    df_aug = build_augmented_table(
        df_orig=df_orig,
        images_src=Path(images_src),
        images_dst=Path(images_dst),
        num_aug_per_image=int(num_aug_per_image),
        start_id=start_id,
        copy_originals=True,
        make_val=make_val,
        val_frac=val_frac,
        seed=seed,
        aug_preset=aug_preset,
        img_size_for_aug=img_size_for_aug,
        no_blur=no_blur,
    )
    write_table(df_aug, Path(out_csv), Path(out_xlsx))
    return df_aug


if __name__ == "__main__":
    main()
