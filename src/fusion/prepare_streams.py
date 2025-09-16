# src/fusion/prepare_streams.py
from pathlib import Path
import numpy as np
import pandas as pd

def _numeric_feature_matrix(df: pd.DataFrame):
    drop_cols = {
        "split","image_name","y","y_hard","y_soft","y_majority",
        "soft_label","p_indirect","label","target"
    }
    cols = [c for c in df.select_dtypes(include=["number"]).columns if c not in drop_cols]
    if not cols:
        raise RuntimeError("No numeric feature columns found for tabular models.")
    return df[cols].to_numpy(), cols

def _find_model(path: Path, patterns):
    # patterns is a list of glob patterns; return the *first* good match ignoring "*threshold*"
    cands = []
    for pat in patterns:
        cands += list(path.glob(pat))
    cands = [p for p in cands if "threshold" not in p.name.lower()]
    return cands[0] if cands else None

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
    out = {"val": {}, "test": {}}

    # -------- Vision streams --------
    try:
        from ..vision.predict_vision import predict_image
        vh_ckpt = weight_dir / "vision_hard_best.pt"
        vs_ckpt = weight_dir / "vision_soft_best.pt"

        if not vh_ckpt.exists():
            raise FileNotFoundError(vh_ckpt)
        if not vs_ckpt.exists():
            # vision soft is optional; we'll just skip if missing
            vs_ckpt = None

        def _batch(df, ckpt, is_reg=False):
            if ckpt is None:
                return None
            ps = []
            for p in df["image_name"]:
                ip = image_root / str(p)
                if not ip.exists():
                    raise FileNotFoundError(f"Image not found: {ip}")
                ps.append(predict_image(str(ckpt), str(ip), is_regressor=is_reg))
            return np.asarray(ps)

        out["val"]["v_hard"] = _batch(df_val, vh_ckpt, is_reg=False)
        out["test"]["v_hard"] = _batch(df_test, vh_ckpt, is_reg=False)
        out["val"]["v_soft"] = _batch(df_val, vs_ckpt, is_reg=True) if vs_ckpt else None
        out["test"]["v_soft"] = _batch(df_test, vs_ckpt, is_reg=True) if vs_ckpt else None
    except Exception as e:
        print(f"[WARN] Vision stream failed: {e}")
        out["val"]["v_hard"] = out["test"]["v_hard"] = None
        out["val"]["v_soft"] = out["test"]["v_soft"] = None

    # -------- Tabular streams --------
    try:
        if not skip_tabular:
            from ..tabular.predict_tabular import predict_xgb, predict_lgbm

            X_val, cols = _numeric_feature_matrix(df_val)
            X_test = df_test[cols].to_numpy()

            # explicit paths override discovery
            if xgb_model_path is None:
                xgb_model_path = _find_model(ml_dir, ["**/*xgb*.pkl","**/*xgb*.joblib","**/*xgb*.json","**/*xgb*.ubj","**/*xgb*.bin"])
            if lgbm_model_path is None:
                lgbm_model_path = _find_model(ml_dir, ["**/*lgbm*.pkl","**/*lgbm*.joblib","**/*lgbm*.txt","**/*lgbm*.json"])

            # Predict if we have a model; otherwise leave as None
            out["val"]["xgb"]  = predict_xgb(xgb_model_path, X_val) if xgb_model_path else None
            out["test"]["xgb"] = predict_xgb(xgb_model_path, X_test) if xgb_model_path else None
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
