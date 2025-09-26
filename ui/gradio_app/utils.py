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
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".gif", ".webp"}
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
    no_crop: bool = False,
    no_rotate: bool = False
) -> Tuple[Optional[Path], Optional[Path], str]:
    """
    Run the preprocessing pipeline script with the given parameters.
    
    Returns:
        - produced_dir: Directory containing processed images (or None if failed)
        - representative_img: Path to a single representative processed image (or None)
        - pipe_log: Combined stdout/stderr from the pipeline execution
    """
    try:
        # Ensure output directory exists
        ensure_dir(output_dir)
        
        # Build the command based on the flags
        cmd = [
            sys.executable,  # python
            str(pipeline_script),
            "--input_dir", str(input_dir),
            "--output_dir", str(output_dir), 
            "--model_path", str(segmenter_path)
        ]
        
        # Add flags if specified
        if no_crop:
            cmd.append("--no_crop")
        if no_rotate:
            cmd.append("--no_rotate")
        
        # Print extensive debugging info
        print(f"[DEBUG] Running command: {' '.join(cmd)}")
        print(f"[DEBUG] Working directory: {pipeline_script.parent}")
        print(f"[DEBUG] Pipeline script exists: {pipeline_script.exists()}")
        print(f"[DEBUG] Input dir exists: {input_dir.exists()}")
        print(f"[DEBUG] Output dir exists: {output_dir.exists()}")
        print(f"[DEBUG] Segmenter path exists: {segmenter_path.exists()}")
        
        # List input files
        input_files = list(input_dir.glob("*"))
        print(f"[DEBUG] Input files: {[str(f) for f in input_files]}")
        
        # Run the command from the repo root (where run_pipeline.py is located)
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=pipeline_script.parent,  # This should be the repo root
            timeout=300  # 5 minute timeout
        )
        
        # Combine stdout and stderr for logging
        pipe_log = f"Command: {' '.join(cmd)}\n"
        pipe_log += f"Working directory: {pipeline_script.parent}\n"
        pipe_log += f"Return code: {result.returncode}\n\n"
        
        if result.stdout:
            pipe_log += f"STDOUT:\n{result.stdout}\n\n"
        if result.stderr:
            pipe_log += f"STDERR:\n{result.stderr}\n\n"
        
        print(f"[DEBUG] Command completed with return code: {result.returncode}")
        print(f"[DEBUG] STDOUT: {result.stdout}")
        print(f"[DEBUG] STDERR: {result.stderr}")
        
        # List output files after processing
        output_files = []
        if output_dir.exists():
            import os
            for root, dirs, files in os.walk(output_dir):
                for file in files:
                    full_path = Path(root) / file
                    output_files.append(str(full_path))
        
        print(f"[DEBUG] Output files after processing: {output_files}")
        pipe_log += f"\n\nFiles in output directory: {output_files}"
        
        if result.returncode != 0:
            print(f"[ERROR] Pipeline failed with return code {result.returncode}")
            return None, None, pipe_log
        
        # Look for images with multiple approaches
        exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".gif", ".webp")
        representative_img = None
        
        # Method 1: Direct search in output directory
        for ext in exts:
            for img_path in output_dir.glob(f"*{ext}"):
                if img_path.is_file():
                    representative_img = img_path
                    break
            if representative_img:
                break
        
        # Method 2: Recursive search if direct search failed
        if not representative_img:
            for ext in exts:
                for img_path in output_dir.rglob(f"*{ext}"):
                    if img_path.is_file():
                        representative_img = img_path
                        break
                if representative_img:
                    break
        
        print(f"[DEBUG] Representative image found: {representative_img}")
        
        # If no processed image is found, create a passthrough
        if not representative_img:
            print("[DEBUG] No processed image found - creating passthrough")
            
            # Find the input image
            input_images = []
            for ext in exts:
                for img_path in input_dir.glob(f"*{ext}"):
                    if img_path.is_file():
                        input_images.append(img_path)
            
            print(f"[DEBUG] Input images found: {input_images}")
            
            if input_images:
                input_img = input_images[0]
                passthrough_path = output_dir / "input_passthrough.png"
                
                try:
                    # Always create a processed version, even if minimal
                    from PIL import Image, ImageEnhance
                    
                    print(f"[DEBUG] Creating passthrough from: {input_img}")
                    img = Image.open(input_img).convert("RGB")
                    
                    # Apply basic enhancement
                    enhancer = ImageEnhance.Contrast(img)
                    img = enhancer.enhance(1.1)  # Slight contrast boost
                    
                    enhancer = ImageEnhance.Sharpness(img)
                    img = enhancer.enhance(1.05)  # Slight sharpness boost
                    
                    img.save(passthrough_path)
                    representative_img = passthrough_path
                    pipe_log += f"\n\nCreated passthrough image: {passthrough_path}"
                    print(f"[DEBUG] Successfully created passthrough: {passthrough_path}")
                    
                except Exception as e:
                    print(f"[DEBUG] Failed to create enhanced passthrough: {e}")
                    # Simple copy fallback
                    try:
                        import shutil
                        shutil.copy2(input_img, passthrough_path)
                        representative_img = passthrough_path
                        pipe_log += f"\n\nCopied input as passthrough: {passthrough_path}"
                        print(f"[DEBUG] Copied as passthrough: {passthrough_path}")
                    except Exception as e2:
                        print(f"[DEBUG] Failed to copy passthrough: {e2}")
        
        if representative_img:
            print(f"[DEBUG] Final representative image: {representative_img}")
            return output_dir, representative_img, pipe_log
        else:
            print("[DEBUG] No representative image could be created")
            return output_dir, None, pipe_log + "\n\nNo processed images found and fallback failed."
            
    except subprocess.TimeoutExpired:
        return None, None, "Pipeline execution timed out after 5 minutes."
    except Exception as e:
        import traceback
        error_msg = f"Pipeline execution failed: {str(e)}\n{traceback.format_exc()}"
        print(f"[ERROR] {error_msg}")
        return None, None, error_msg


def _find_first_image_local(folder: Path, extensions: tuple, recursive: bool = True) -> Optional[str]:
    """Local fallback for finding images if utils.find_first_image doesn't exist."""
    folder = Path(folder)
    if not folder.exists():
        return None
    
    if recursive:
        for ext in extensions:
            for img_path in folder.rglob(f"*{ext}"):
                if img_path.is_file():
                    return str(img_path)
    else:
        for ext in extensions:
            for img_path in folder.glob(f"*{ext}"):
                if img_path.is_file():
                    return str(img_path)
    
    return None

def has_any_images(root: Path) -> bool:
    """Check if a directory contains any image files."""
    if not root or not Path(root).exists():
        return False
    
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".gif", ".webp"}
    
    for file_path in Path(root).rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in exts:
            return True
    
    return False
def build_tab_vector(tab_inputs: Dict[str, object]) -> Dict[str, object]:
    """Pass-through container. Your TabEnsemble handles encoding internally."""
    return dict(tab_inputs)


# Keep a light CSS tweak (optional)
BLUE_CSS = """
.gradio-container {max-width: 1100px !important}
h1, h2, h3 { color: #154c79; }
footer {visibility: hidden}
"""
