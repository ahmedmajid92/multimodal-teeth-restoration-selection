import os, sys, subprocess, shutil, uuid
from pathlib import Path
from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
from PIL import Image

TAB_FEATURES = [
    'depth','width','enamel_cracks','occlusal_load','carious_lesion',
    'opposing_type','adjacent_teeth','age_range','cervical_lesion'
]
CAT_FEATURES = ['enamel_cracks','occlusal_load','carious_lesion','opposing_type','adjacent_teeth','age_range','cervical_lesion']
CONT_FEATURES = ['depth','width']

BLUE_CSS = """
#component-0, .gradio-container {
  --color-accent: #3b82f6;
  --color-accent-soft: #dbeafe;
}
.gr-button { border-radius: 8px; }
.gradio-container { background: #f8fafc; }
"""

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def load_feature_choices_from_sheet(sheet_path: Path) -> Dict[str, list]:
    """Read unique values for categorical fields from your data_processed.xlsx to populate dropdowns."""
    if sheet_path.suffix.lower()==".xlsx":
        df = pd.read_excel(sheet_path)
    else:
        df = pd.read_csv(sheet_path)
    choices = {}
    for c in CAT_FEATURES:
        if c in df.columns:
            vals = sorted([int(v) for v in pd.Series(df[c]).dropna().astype(int).unique().tolist()])
            choices[c] = [str(v) for v in vals]
        else:
            choices[c] = []
    return choices

def check_image_resolution(img: Image.Image, min_size=512) -> Tuple[bool,str]:
    w,h = img.size
    if w < min_size or h < min_size:
        return False, f"Image too small ({w}x{h}). Please upload ≥ {min_size}×{min_size}."
    return True, "OK"

def run_preprocessing_pipeline(pipeline_script: Path, input_dir: Path, output_dir: Path,
                               segmenter_path: Path, no_crop: bool, no_rotate: bool) -> Path | None:
    """Invoke your run_pipeline.py as a subprocess and return one processed image path."""
    ensure_dir(input_dir); ensure_dir(output_dir)
    # Clean output dir
    for f in output_dir.glob("*"): 
        try: f.unlink()
        except: pass
    cmd = [
        sys.executable, str(pipeline_script),
        "--input_dir", str(input_dir),
        "--output_dir", str(output_dir),
        "--model_path", str(segmenter_path)
    ]
    if no_crop:   cmd.append("--no_crop")
    if no_rotate: cmd.append("--no_rotate")
    print("Running:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print("Pipeline stderr:", e.stderr)
        return None

    # Return the first image produced
    outs = list(output_dir.glob("*.jpg")) + list(output_dir.glob("*.png"))
    return outs[0] if outs else None

def build_tab_vector(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and coerce tab inputs to numeric dict in the expected order."""
    vec = {}
    for c in TAB_FEATURES:
        v = inputs.get(c, None)
        if v is None or v=="":
            raise ValueError(f"Missing field: {c}")
        if c in CONT_FEATURES:
            vec[c] = float(v)
        else:
            vec[c] = int(v)
    return vec
