# src/fusion/prepare_streams.py
from pathlib import Path
import numpy as np
import pandas as pd

# -----------------------------
# Feature spec used by ML training (9 base + 7 engineered = 16)
# -----------------------------
_BASE_FEATURES = [
    "depth", "width", "enamel_cracks", "occlusal_load", "carious_lesion",
    "opposing_type", "adjacent_teeth", "age_range", "cervical_lesion"
]
_ENGINEERED = [
    "deep_and_thin", "deep_or_cracks", "load_implant",
    "risk_plus_cervical", "stable_wall", "depth_x_load", "depth_x_risk"
]
_ALL_FEATURES = _BASE_FEATURES + _ENGINEERED

def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recreate the DomainFeatures logic from training on a copy of df
    and return a DataFrame with the exact 16 columns in the right order.
    """
    missing = [c for c in _BASE_FEATURES if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing base features: {missing}")

    X = df[_BASE_FEATURES].copy()

    # Ensure numeric + no NaNs (mirror imputer safety)
    for c in _BASE_FEATURES:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0).astype(int)

    # Engineered features (same rules as training DomainFeatures)
    X["deep_and_thin"]     = ((X["depth"] == 1) & (X["width"] == 0)).astype(int)
    X["deep_or_cracks"]    = ((X["depth"] == 1) | (X["enamel_cracks"] == 1)).astype(int)
    X["load_implant"]      = ((X["occlusal_load"] == 1) & (X["opposing_type"] == 3)).astype(int)
    X["risk_plus_cervical"]= ((X["carious_lesion"] == 1) & (X["cervical_lesion"] == 1)).astype(int)
    X["stable_wall"]       = ((X["width"] == 1) & (X["enamel_cracks"] == 0) & (X["occlusal_load"] == 0)).astype(int)
    X["depth_x_load"]      = (X["depth"] * X["occlusal_load"]).astype(int)
    X["depth_x_risk"]      = (X["depth"] * X["carious_lesion"]).astype(int)

    # Return in fixed order
    return X[_ALL_FEATURES]

def _find_model(path: Path, patterns):
    """
    Find the first model file that matches the given glob patterns.
    Ignores files containing 'threshold' in their names.
    """
    cands = []
    for pat in patterns:
        cands += list(path.glob(pat))
    cands = [p for p in cands if "threshold" not in p.name.lower()]
    return cands[0] if cands else None

# -----------------------------
# Helpers for vision streams
# -----------------------------
def _resolve_image(image_root: Path, name: str) -> Path:
    """
    Resolve an image file robustly:
    - respects exact path if it exists
    - otherwise tries common extensions with case variations
    """
    p = image_root / str(name)
    if p.exists():
        return p
    stem = Path(name).stem
    for ext in (".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG"):
        q = image_root / f"{stem}{ext}"
        if q.exists():
            return q
    raise FileNotFoundError(f"Image not found: {image_root}/{name}")

# -----------------------------
# Main collector
# -----------------------------
def collect_base_preds(
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    image_root: Path,
    weight_dir: Path,
    ml_dir: Path,
    xgb_model_path: Path | None = None,
    lgbm_model_path: Path | None = None,
    skip_tabular: bool = False
):
    """
    Collect base-model predictions for VAL and TEST splits.

    Returns a dict:
      {
        "val":  {"v_hard": ndarray|None, "v_soft": ndarray|None, "xgb": ndarray|None, "lgbm": ndarray|None},
        "test": {"v_hard": ndarray|None, "v_soft": ndarray|None, "xgb": ndarray|None, "lgbm": ndarray|None}
      }

    Any stream that cannot be produced will be set to None (and safely ignored by fusion).
    """
    out = {"val": {}, "test": {}}

    # -------- Vision streams --------
    try:
        from ..vision.predict_vision import predict_image

        vh_ckpt = weight_dir / "vision_hard_best.pt"
        vs_ckpt = weight_dir / "vision_soft_best.pt"

        if not vh_ckpt.exists():
            raise FileNotFoundError(f"Missing vision HARD checkpoint: {vh_ckpt}")

        if not vs_ckpt.exists():
            # Vision-soft is optional; skip if not present
            vs_ckpt = None

        def _batch(df: pd.DataFrame, ckpt: Path | None, is_reg: bool = False):
            if ckpt is None:
                return None
            preds = []
            # prefer "image_name", fall back to "image_path"
            col = "image_name" if "image_name" in df.columns else ("image_path" if "image_path" in df.columns else None)
            if col is None:
                raise KeyError("Neither 'image_name' nor 'image_path' column found in dataframe.")
            for name in df[col].astype(str).tolist():
                ip = _resolve_image(image_root, name)
                preds.append(predict_image(str(ckpt), str(ip), is_regressor=is_reg))
            return np.asarray(preds, dtype=float)

        out["val"]["v_hard"] = _batch(df_val, vh_ckpt, is_reg=False)
        out["test"]["v_hard"] = _batch(df_test, vh_ckpt, is_reg=False)
        out["val"]["v_soft"] = _batch(df_val, vs_ckpt, is_reg=True) if vs_ckpt else None
        out["test"]["v_soft"] = _batch(df_test, vs_ckpt, is_reg=True) if vs_ckpt else None

    except Exception as e:
        print(f"[WARN] Vision stream failed: {e}")
        out["val"]["v_hard"] = out["test"]["v_hard"] = None
        out["val"]["v_soft"]  = out["test"]["v_soft"]  = None

    # -------- Tabular streams --------
    try:
        if not skip_tabular:
            from ..tabular.predict_tabular import predict_xgb, predict_lgbm

            # Build features for BOTH splits with the SAME recipe
            X_val_df  = _build_features(df_val)
            X_test_df = _build_features(df_test)

            # explicit paths override discovery
            if xgb_model_path is None:
                xgb_model_path = _find_model(
                    ml_dir,
                    ["**/*xgb*.pkl", "**/*xgb*.joblib", "**/*xgb*.json", "**/*xgb*.ubj", "**/*xgb*.bin"]
                )
            if lgbm_model_path is None:
                lgbm_model_path = _find_model(
                    ml_dir,
                    ["**/*lgbm*.pkl", "**/*lgbm*.joblib", "**/*lgbm*.txt", "**/*lgbm*.json"]
                )

            # NOTE: we pass numpy arrays (same 16-col order) to predictors
            Xv = X_val_df[_ALL_FEATURES]   # DataFrame, keeps names
            Xt = X_test_df[_ALL_FEATURES]

            out["val"]["xgb"]   = predict_xgb(xgb_model_path, Xv) if xgb_model_path else None
            out["test"]["xgb"]  = predict_xgb(xgb_model_path, Xt) if xgb_model_path else None
            out["val"]["lgbm"]  = predict_lgbm(lgbm_model_path, Xv) if lgbm_model_path else None
            out["test"]["lgbm"] = predict_lgbm(lgbm_model_path, Xt) if lgbm_model_path else None

        else:
            out["val"]["xgb"] = out["test"]["xgb"] = None
            out["val"]["lgbm"] = out["test"]["lgbm"] = None

    except Exception as e:
        print(f"[WARN] Tabular stream failed: {e}")
        out["val"]["xgb"] = out["test"]["xgb"] = None
        out["val"]["lgbm"] = out["test"]["lgbm"] = None

    return out
