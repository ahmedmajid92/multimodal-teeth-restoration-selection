"""
Wrapper around a Mask-RCNN model (torchvision) to extract a binary mask
for the *largest predicted tooth instance* (assumed to be the target molar).
"""
import torch
import cv2
import numpy as np
from torchvision.models.detection import (
    maskrcnn_resnet50_fpn,
    MaskRCNN_ResNet50_FPN_Weights
)
from ..config import CROP_MARGIN_PX, OUTPUT_SIZE

class MolarSegmenter:
    def __init__(self, model_path=None, conf_thresh: float = 0.0):
        # Device selection
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Load full COCO-pretrained Mask R-CNN (91 classes)
        weights = MaskRCNN_ResNet50_FPN_Weights.COCO_V1
        self.model = maskrcnn_resnet50_fpn(weights=weights).to(self.device).eval()
        self.conf_thresh = conf_thresh

    @torch.inference_mode()
    def __call__(self, img_bgr: np.ndarray) -> np.ndarray:
        # Convert to RGB tensor
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        tensor  = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        tensor  = tensor.to(self.device)
        # Predict
        outputs = self.model([tensor])[0]
        valid = []
        h,w = img_bgr.shape[:2]
        for m,sc in zip(outputs["masks"], outputs["scores"]):
            if sc < 0.05:      # very low conf, skip
                continue
            mask = (m[0].cpu().numpy() > 0.5)
            # Heuristic-1: reject “gray” regions (metal) ---------
            hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
            if hsv[...,1][mask].mean() < 40:   # low saturation
                continue
            # Heuristic-2: distance from centre ------------------
            ys, xs = np.where(mask)
            cx, cy = xs.mean(), ys.mean()
            centre_dist = np.hypot(cx - w/2, cy - h/2)
            valid.append((centre_dist, mask))

        if not valid:
            raise RuntimeError("No valid tooth mask")
        # pick mask closest to centre
        mask = min(valid, key=lambda v: v[0])[1]
        if len(outputs["scores"]) == 0:
            raise RuntimeError("No objects detected – check model or image quality.")
        # Select highest-confidence mask
        idx = int(outputs["scores"].argmax())
        if outputs["scores"][idx] < self.conf_thresh:
            raise RuntimeError("Detection score below threshold.")
        mask = outputs["masks"][idx, 0].cpu().numpy() > 0.5
        return (mask.astype(np.uint8) * 255)

def crop_with_mask(img: np.ndarray, mask: np.ndarray,
                   margin: int = CROP_MARGIN_PX) -> np.ndarray:
    # Locate mask bounds
    y, x = np.where(mask)
    if y.size == 0 or x.size == 0:
        raise RuntimeError("Empty mask – no region to crop.")
    y0, y1 = y.min(), y.max()
    x0, x1 = x.min(), x.max()
    # Apply margin and clamp to image
    y0 = max(y0 - margin, 0)
    x0 = max(x0 - margin, 0)
    y1 = min(y1 + margin, img.shape[0])
    x1 = min(x1 + margin, img.shape[1])
    crop = img[y0:y1, x0:x1]
    # Pad to square
    h, w = crop.shape[:2]
    d = max(h, w)
    padded = np.zeros((d, d, 3), dtype=crop.dtype)
    y_off, x_off = (d - h) // 2, (d - w) // 2
    padded[y_off:y_off+h, x_off:x_off+w] = crop
    # Resize to output size
    return cv2.resize(padded, (OUTPUT_SIZE, OUTPUT_SIZE),
                      interpolation=cv2.INTER_LINEAR)