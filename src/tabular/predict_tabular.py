# src/tabular/predict_tabular.py
from pathlib import Path
import json
import numpy as np
import joblib

def _is_xgb_json(path: Path) -> bool:
    # XGBoost JSON contains a "learner"/"gradient_booster" structure
    try:
        with open(path, "r", encoding="utf-8") as f:
            j = json.load(f)
        return "learner" in j or "gradient_booster" in j or "version" in j and "learner" in j.get("model", {})
    except Exception:
        return False

def _is_lgb_text(path: Path) -> bool:
    # LightGBM text boosters start with "tree\n" or "feature_names"
    try:
        head = path.read_text(encoding="utf-8", errors="ignore")[:64].lower()
        return "feature_names" in head or head.startswith("tree")
    except Exception:
        return False

def _safe_load_model(p):
    p = Path(p)
    if not p.exists():
        raise FileNotFoundError(p)

    # sklearn/joblib wrappers
    if p.suffix in [".pkl", ".joblib"]:
        return joblib.load(p)

    # XGBoost native JSON/UBJ/BIN
    if p.suffix in [".json", ".ubj", ".bin"] and _is_xgb_json(p):
        import xgboost as xgb
        return xgb.Booster(model_file=str(p))

    # LightGBM native TXT (or JSON misnamed)
    if p.suffix in [".txt"] or _is_lgb_text(p):
        import lightgbm as lgb
        return lgb.Booster(model_file=str(p))

    # Last attempt: try XGB first, then LGB
    try:
        import xgboost as xgb
        return xgb.Booster(model_file=str(p))
    except Exception:
        import lightgbm as lgb
        return lgb.Booster(model_file=str(p))

def predict_lgbm(model_path, X):
    m = _safe_load_model(model_path)
    try:
        # sklearn API
        return np.asarray(m.predict_proba(X)[:, 1]).reshape(-1)
    except Exception:
        # Booster API
        import numpy as np
        try:
            import lightgbm as lgb
            return np.asarray(m.predict(X)).reshape(-1)
        except Exception:
            # If it's actually XGB but user passed it here, fall back
            import xgboost as xgb
            return np.asarray(m.predict(xgb.DMatrix(X))).reshape(-1)

def predict_xgb(model_path, X):
    m = _safe_load_model(model_path)
    try:
        return np.asarray(m.predict_proba(X)[:, 1]).reshape(-1)
    except Exception:
        import xgboost as xgb
        try:
            return np.asarray(m.predict(xgb.DMatrix(X))).reshape(-1)
        except Exception:
            # If it's LGB Booster, fall back
            import lightgbm as lgb
            return np.asarray(m.predict(X)).reshape(-1)
