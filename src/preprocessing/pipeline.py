"""
Full step-1 pipeline with **toggles** for cropping and rotation.

Run it in two modes for cropping:
• Default (cropping ON)   → deskew + Mask-RCNN + crop + resize 256²
• --no_crop               → (optional deskew) + centre-resize full frame

And a rotation toggle:
• Default (rotation ON)   → apply CLAHE, then deskew (rotate)
• --no_rotate             → apply CLAHE only (no rotation)
"""
from pathlib import Path
import cv2, json, traceback
import numpy as np
from ..config import (MIN_EDGE_PX, RAW_IMG_DIR, PROC_IMG_DIR,
                      LOG_DIR, OUTPUT_SIZE)
from ..utils.io import ensure_dir, save_json, timestamp
from .normalise import apply_clahe, deskew
from .segment import MolarSegmenter, crop_with_mask

# --------------------------------------------------------------

def centre_crop_resize(img: np.ndarray, size: int) -> np.ndarray:
    """Center-crop to square then resize to (size, size)."""
    h, w = img.shape[:2]
    d = min(h, w)
    y0, x0 = (h - d) // 2, (w - d) // 2
    crop = img[y0:y0 + d, x0:x0 + d]
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)

# --------------------------------------------------------------
class ImagePreprocessor:
    def __init__(self, seg_model_path: Path, crop: bool = True, rotate: bool = True):
        self.crop = crop
        self.rotate = rotate
        # Only initialize segmenter if cropping enabled
        self.segmenter = None if not crop else MolarSegmenter(seg_model_path)
        ensure_dir(PROC_IMG_DIR)
        ensure_dir(LOG_DIR)
        crop_mode = "ON" if crop else "OFF (no crop)"
        rot_mode = "ON" if rotate else "OFF (no rotate)"
        print(f"[Preproc] Cropping: {crop_mode} | Rotation: {rot_mode}")

    # --------------------------------------------------
    def _save(self, img: np.ndarray, stem: str) -> str:
        out_name = stem + ".jpg"
        out_path = PROC_IMG_DIR / out_name
        cv2.imwrite(str(out_path), img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        return out_name

    # --------------------------------------------------
    def process_file(self, path: Path) -> dict:
        info = {"file": path.name}
        try:
            img = cv2.imread(str(path))
            if img is None:
                raise ValueError("OpenCV could not read file")
            if min(img.shape[:2]) < MIN_EDGE_PX:
                raise ValueError("Image too small (<400 px)")

            # 1) CLAHE (colour/contrast)
            img = apply_clahe(img)

            # 2) Deskew (rotation) – optional
            if self.rotate:
                img, rot = deskew(img)
                info["rotation_deg"] = rot
            else:
                info["rotation_deg"] = 0.0

            # 3) Crop or keep whole frame
            if self.crop:
                try:
                    mask = self.segmenter(img)
                    img = crop_with_mask(img, mask)
                    info["crop_mode"] = "maskrcnn"
                except Exception as seg_err:
                    img = centre_crop_resize(img, OUTPUT_SIZE)
                    info["crop_mode"] = "centre_fallback"
                    info["segmentation_error"] = str(seg_err)
            else:
                # Resize/pad full-frame to 256²
                img = centre_crop_resize(img, OUTPUT_SIZE)
                info["crop_mode"] = "none"

            # 4) Save processed image
            info["out_file"] = self._save(img, path.stem)
            info["status"] = "ok"

        except Exception as e:
            info.update({
                "status": "error",
                "error_msg": str(e),
                "traceback": traceback.format_exc(limit=1)
            })
        return info

    # --------------------------------------------------
    def process_dir(self, in_dir: Path):
        logs = []
        for p in in_dir.iterdir():
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                logs.append(self.process_file(p))
        save_json(logs, LOG_DIR / f"preprocess_{timestamp()}.json")
