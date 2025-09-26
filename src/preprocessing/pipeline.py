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
    def __init__(self, seg_model_path: Path, output_dir: Path, crop: bool = True, rotate: bool = True):
        self.crop = crop
        self.rotate = rotate
        self.output_dir = output_dir  # Use the provided output directory
        
        # Ensure the output directory exists
        ensure_dir(self.output_dir)

        # Only initialize segmenter if cropping enabled
        self.segmenter = None if not crop else MolarSegmenter(seg_model_path)
        ensure_dir(LOG_DIR)
        crop_mode = "ON" if crop else "OFF (no crop)"
        rot_mode = "ON" if rotate else "OFF (no rotate)"
        print(f"[Preproc] Cropping: {crop_mode} | Rotation: {rot_mode}")

    # --------------------------------------------------
    def _save(self, img: np.ndarray, stem: str) -> str:
        """Save processed image to the current output directory."""
        out_name = stem + ".jpg"
        # Use the configured output directory (set by config.PROC_IMG_DIR)
        out_path = self.output_dir / out_name
        
        # Ensure the output directory exists
        ensure_dir(self.output_dir)
        
        print(f"[DEBUG] Saving processed image to: {out_path}")
        success = cv2.imwrite(str(out_path), img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        
        if not success:
            print(f"[ERROR] Failed to save image to {out_path}")
            raise ValueError(f"Failed to save processed image to {out_path}")
        else:
            print(f"[DEBUG] Successfully saved: {out_path}")
            
        return out_name

    # --------------------------------------------------
    def process_file(self, path: Path) -> dict:
        """Process a single image file."""
        print(f"[DEBUG] Processing file: {path}")
        info = {"file": path.name}
        
        try:
            img = cv2.imread(str(path))
            if img is None:
                raise ValueError("OpenCV could not read file")
            if min(img.shape[:2]) < MIN_EDGE_PX:
                raise ValueError("Image too small (<400 px)")

            print(f"[DEBUG] Loaded image: {img.shape}")

            # 1) CLAHE (colour/contrast)
            print("[DEBUG] Applying CLAHE...")
            img = apply_clahe(img)

            # 2) Deskew (rotation) – optional
            if self.rotate:
                print("[DEBUG] Applying deskew/rotation...")
                img, rot = deskew(img)
                info["rotation_deg"] = rot
                print(f"[DEBUG] Rotation applied: {rot:.2f} degrees")
            else:
                info["rotation_deg"] = 0.0
                print("[DEBUG] Skipping rotation (--no_rotate)")

            # 3) Crop or keep whole frame
            if self.crop:
                print("[DEBUG] Attempting segmentation cropping...")
                try:
                    mask = self.segmenter(img)
                    print("[DEBUG] Segmentation successful, cropping...")
                    img = crop_with_mask(img, mask)
                    info["crop_mode"] = "maskrcnn"
                    print(f"[DEBUG] Cropped image size: {img.shape}")
                except Exception as seg_err:
                    print(f"[DEBUG] Segmentation failed: {seg_err}, using center crop fallback")
                    img = centre_crop_resize(img, OUTPUT_SIZE)
                    info["crop_mode"] = "centre_fallback"
                    info["segmentation_error"] = str(seg_err)
            else:
                print("[DEBUG] Skipping crop (--no_crop), using center resize...")
                # Resize/pad full-frame to 256²
                img = centre_crop_resize(img, OUTPUT_SIZE)
                info["crop_mode"] = "none"

            print(f"[DEBUG] Final processed image size: {img.shape}")

            # 4) Save processed image
            out_filename = self._save(img, path.stem)
            info["out_file"] = out_filename
            info["status"] = "ok"
            print(f"[DEBUG] Processing completed successfully: {out_filename}")

        except Exception as e:
            print(f"[ERROR] Processing failed for {path}: {e}")
            info.update({
                "status": "error",
                "error_msg": str(e),
                "traceback": traceback.format_exc(limit=1)
            })
        return info

    # --------------------------------------------------
    def process_dir(self, in_dir: Path):
        """Process all images in the input directory."""
        print(f"[DEBUG] Processing directory: {in_dir}")
        print(f"[DEBUG] Output directory is set to: {self.output_dir}")
        
        logs = []
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        
        processed_count = 0
        for p in in_dir.iterdir():
            if p.suffix.lower() in image_extensions:
                print(f"[DEBUG] Found image file: {p}")
                log_entry = self.process_file(p)
                logs.append(log_entry)
                if log_entry["status"] == "ok":
                    processed_count += 1
                    
        print(f"[DEBUG] Processed {processed_count} images successfully")
        
        if logs:
            log_file = LOG_DIR / f"preprocess_{timestamp()}.json"
            ensure_dir(LOG_DIR)
            save_json(logs, log_file)
            print(f"[DEBUG] Saved processing log to: {log_file}")
