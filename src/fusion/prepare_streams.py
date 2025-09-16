# src/fusion/prepare_streams.py
from pathlib import Path
import numpy as np
import pandas as pd

# -----------------------------
# Helpers for tabular features
# -----------------------------
def _numeric_feature_matrix(df: pd.DataFrame):
    """
    Build a numeric feature matrix for tabular models.
    Excludes common label/metadata columns if they are numeric.
    """
    drop_cols = {
        "split", "image_name", "image_path",
        "y", "y_hard", "y_soft", "y_majority",
        "soft_label", "p_indirect", "label", "target",
        "weight"
    }
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cols = [c for c in num_cols if c not in drop_cols]
    if not cols:
        raise RuntimeError("No numeric feature columns found for tabular models.")
    return df[cols].to_numpy(), cols

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

            X_val, cols = _numeric_feature_matrix(df_val)
            X_test = df_test[cols].to_numpy()

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

            out["val"]["xgb"]   = predict_xgb(xgb_model_path, X_val) if xgb_model_path else None
            out["test"]["xgb"]  = predict_xgb(xgb_model_path, X_test) if xgb_model_path else None
            out["val"]["lgbm"]  = predict_lgbm(lgbm_model_path, X_val) if lgbm_model_path else None
            out["test"]["lgbm"] = predict_lgbm(lgbm_model_path, X_test) if lgbm_model_path else None
        else:
            out["val"]["xgb"] = out["test"]["xgb"] = None
            out["val"]["lgbm"] = out["test"]["lgbm"] = None

    except Exception as e:
        print(f"[WARN] Tabular stream failed: {e}")
        out["val"]["xgb"] = out["test"]["xgb"] = None
        out["val"]["lgbm"] = out["test"]["lgbm"] = None

    return out
