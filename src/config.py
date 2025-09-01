"""
Global configuration flags.
Edit paths once here; other modules import them.
"""
from pathlib import Path

PROJECT_ROOT   = Path(__file__).resolve().parents[1]
RAW_IMG_DIR    = PROJECT_ROOT / "data" / "raw" / "images"
PROC_IMG_DIR   = PROJECT_ROOT / "data" / "processed" / "images"
LOG_DIR        = PROJECT_ROOT / "data" / "processed" / "logs"

# Preprocessing parameters
MIN_EDGE_PX    = 400           # discard smaller images
OUTPUT_SIZE    = 512           # final square dimension
CLAHE_CLIP     = 3.0           # contrast-limit
CLAHE_TILEGR   = (8, 8)
ROT_TOLERANCE  = 15            # allowed residual rotation in degrees
CROP_MARGIN_PX = 15            # extra border around predicted mask
