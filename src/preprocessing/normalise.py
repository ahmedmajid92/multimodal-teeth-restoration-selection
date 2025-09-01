"""
Colour, contrast and geometric normalisation utilities.
"""
import cv2
import numpy as np

from ..config import CLAHE_CLIP, CLAHE_TILEGR, ROT_TOLERANCE

# ---------- colour & contrast ------------------------------------------------
def apply_clahe(img_bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILEGR)
    l2 = clahe.apply(l)
    lab = cv2.merge((l2, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

# ---------- orientation ------------------------------------------------------
def deskew(img_bgr: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Roughly rotate so that the dominant axis of edge points lies horizontal.
    Returns (rotated_image, applied_angle_in_degrees).
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    coords = np.column_stack(np.where(edges > 0))  # (y, x) pairs

    # If too few edge points, skip rotation
    if coords.shape[0] < 10:
        return img_bgr, 0.0

    # Center and compute covariance
    centered = coords - coords.mean(axis=0)
    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)

    # Principal component = eigenvector with largest eigenvalue
    principal = eigvecs[:, np.argmax(eigvals)]  # [e_y, e_x]

    # Angle between that axis and horizontal
    angle_rad = np.arctan2(principal[0], principal[1])
    angle_deg = np.rad2deg(angle_rad)

    # Skip tiny corrections
    if abs(angle_deg) < ROT_TOLERANCE:
        return img_bgr, 0.0

    # Rotate about the image center
    (h, w) = img_bgr.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated = cv2.warpAffine(
        img_bgr, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )
    return rotated, angle_deg
