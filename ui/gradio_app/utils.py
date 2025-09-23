# ui/gradio_app/utils.py
import sys
import subprocess
import uuid
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Iterable

import pandas as pd
from PIL import Image

# -------------------------
# Small helpers
# -------------------------

def ensure_dir(path: Path) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)


def check_image_resolution(img: Image.Image, min_size: int = 512) -> Tuple[bool, str]:
    w, h = img.size
    if min(w, h) < min_size:
        return False, f"Image too small ({w}×{h}). Please upload ≥ {min_size}×{min_size}."
    return True, f"OK ({w}×{h})"

def has_any_images(root: Path) -> bool:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    root = Path(root)
    if root.is_file():
        return root.suffix.lower() in exts
    if root.is_dir():
        for p in root.rglob("*"):
            if p.suffix.lower() in exts:
                return True
    return False

def load_feature_choices_from_sheet(xlsx_path: Path) -> Dict[str, list]:
    """
    Reads your processed Excel and returns choices for the categorical fields
    used in the UI. If a column is missing, returns [] for that field.
    """
    xlsx_path = Path(xlsx_path)
    if not xlsx_path.exists():
        raise FileNotFoundError(f"Sheet not found: {xlsx_path}")

    try:
        df = pd.read_excel(xlsx_path, sheet_name=0)
    except Exception:
        df = pd.read_excel(xlsx_path)

    def uniq(col):
        if col not in df.columns:
            return []
        vals = df[col].dropna().astype(str).unique().tolist()
        vals = [v for v in vals if v != ""]
        return sorted(vals)

    fields = [
        "enamel_cracks", "occlusal_load", "carious_lesion",
        "opposing_type", "adjacent_teeth", "age_range", "cervical_lesion"
    ]
    return {f: uniq(f) for f in fields}


# -------------------------
# Session-scoped temp dirs & I/O
# -------------------------

def make_session_dirs(base_raw: Path, base_proc: Path) -> Tuple[Path, Path, str]:
    """
    Create unique raw/processed subfolders for this inference session.
    Returns (raw_dir, proc_dir, session_id)
    """
    sid = uuid.uuid4().hex[:12]
    raw_dir = Path(base_raw) / sid
    proc_dir = Path(base_proc) / sid
    ensure_dir(raw_dir)
    ensure_dir(proc_dir)
    return raw_dir, proc_dir, sid


def save_uploaded_image_to_dir(img_obj, dst_dir: Path, filename: str = "input.png") -> Path:
    """
    Save a Gradio image input (PIL.Image or a path-like) to dst_dir / filename.
    Returns the saved path.
    """
    ensure_dir(dst_dir)
    out_path = Path(dst_dir) / filename

    # Gradio with type="pil" gives PIL.Image.Image
    if isinstance(img_obj, Image.Image):
        img_obj.convert("RGB").save(out_path)
        return out_path

    # Sometimes a temp file path (str / Path)
    p = Path(str(img_obj))
    if p.exists() and p.is_file():
        # preserve extension if present
        if p.suffix:
            out_path = Path(dst_dir) / p.name
        Image.open(p).convert("RGB").save(out_path)
        return out_path

    raise ValueError("Unsupported image object type for saving.")


# -------------------------
# Image discovery helpers
# -------------------------

_IMAGE_EXTS: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".gif")

def _find_images_recursive(root: Path) -> List[Path]:
    root = Path(root)
    if not root.exists():
        return []
    return sorted([p for p in root.rglob("*") if p.suffix.lower() in _IMAGE_EXTS])

def _iter_images_shallow(root: Path, extensions: Optional[Iterable[str]] = None) -> Iterable[Path]:
    exts = set((extensions or _IMAGE_EXTS))
    for p in Path(root).iterdir():
        if p.is_file() and p.suffix.lower() in exts:
            yield p

def find_first_image(folder: Path, extensions: Optional[List[str]] = None, recursive: bool = True) -> Optional[Path]:
    """
    Return the first image found under folder (optionally non-recursive).
    """
    folder = Path(folder)
    if not folder.exists():
        return None
    exts = set([e.lower() if e.startswith(".") else "." + e.lower() for e in (extensions or [])]) or _IMAGE_EXTS
    if recursive:
        for p in folder.rglob("*"):
            if p.is_file() and p.suffix.lower() in exts:
                return p
    else:
        for p in folder.iterdir():
            if p.is_file() and p.suffix.lower() in exts:
                return p
    return None


# -------------------------
# Pipeline launcher
# -------------------------


def run_preprocessing_pipeline(
    pipeline_script: Path,
    input_dir: Path,
    output_dir: Path,
    segmenter_path: Path,
    no_crop: bool = True,
    no_rotate: bool = True,
) -> Tuple[Optional[Path], Optional[Path], str]:
    """
    Launches run_pipeline.py as a subprocess and returns:
        (produced_dir, first_image_path, log_text)

    Success is determined by presence of at least one image under output_dir (recursively).
    If none are found, we create a *passthrough* image in output_dir from input_dir
    so downstream models can still run.
    """
    ensure_dir(input_dir)
    ensure_dir(output_dir)

    cmd = [
        sys.executable,
        str(pipeline_script),
        "--input_dir", str(input_dir),
        "--output_dir", str(output_dir),
        "--model_path", str(segmenter_path),
    ]
    if no_crop:
        cmd.append("--no_crop")
    if no_rotate:
        cmd.append("--no_rotate")

    out_text = []
    # 1) Run external pipeline (best effort)
    try:
        res = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="ignore",
            check=False,
        )
        if res.stdout:
            out_text.append(res.stdout.strip())
        if res.stderr:
            out_text.append(res.stderr.strip())
    except Exception as e:
        out_text.append(f"[pipeline] failed to start: {e}")

    # 2) Did the pipeline produce anything?
    imgs = _find_images_recursive(Path(output_dir))
    if imgs:
        first_img = imgs[0]
        produced_dir = first_img.parent
        return produced_dir, first_img, "\n".join(out_text)

    # 3) Fallback: create a passthrough image in output_dir
    src_img = find_first_image(Path(input_dir), recursive=True)
    if src_img is None:
        out_text.append("[pipeline] WARNING: No images produced AND no source image found for fallback.")
        return None, None, "\n".join(out_text)

    try:
        ensure_dir(output_dir)
        # keep name informative
        out_name = src_img.stem + "_passthrough.png"
        out_path = Path(output_dir) / out_name

        # Use PIL to normalize mode & write to output
        with Image.open(src_img) as im:
            im = im.convert("RGB")
            im.save(out_path)

        out_text.append("[pipeline] WARNING: No images produced by pipeline; wrote passthrough image instead.")
        return out_path.parent, out_path, "\n".join(out_text)
    except Exception as e:
        out_text.append(f"[pipeline] ERROR creating passthrough image: {e}")
        return None, None, "\n".join(out_text)


def build_tab_vector(tab_inputs: Dict[str, object]) -> Dict[str, object]:
    """Pass-through container. Your TabEnsemble handles encoding internally."""
    return dict(tab_inputs)


# Keep a light CSS tweak (optional)
BLUE_CSS = """
.gradio-container {max-width: 1100px !important}
h1, h2, h3 { color: #154c79; }
footer {visibility: hidden}
"""
