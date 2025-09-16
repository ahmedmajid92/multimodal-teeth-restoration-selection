# src/tabular/predict_tabular.py
from pathlib import Path
import sys
import json
import numpy as np

def _register_dummy_main_class(name: str):
    """
    If a pickle references __main__.ImputerThenModel (or similar),
    create a minimal placeholder so joblib.load can succeed.
    """
    main = sys.modules.get("__main__")
    if main is None:
        return
    if not hasattr(main, name):
        # Minimal shell; actual learned params are injected by unpickling.
        class _Dummy:
            def __init__(self, *args, **kwargs):
                pass
        setattr(main, name, _Dummy)

def _is_xgb_json(path: Path) -> bool:
    try:
        with open(path, "r", encoding="utf-8") as f:
            j = json.load(f)
        # crude but reliable XGB signature
        return "learner" in j or "gradient_booster" in j or "version" in j
    except Exception:
        return False

def _is_lgb_text(path: Path) -> bool:
    try:
        head = path.read_text(encoding="utf-8", errors="ignore")[:128].lower()
        return "feature_names" in head or head.startswith("tree") or "num_class" in head
    except Exception:
        return False

def _joblib_load(p: Path):
    # Make custom classes available in __main__ for unpickling
    for cname in ("ImputerThenModel",):
        _register_dummy_main_class(cname)
    import joblib
    return joblib.load(p)

def _safe_load_model(p):
    """
    Load a model from a variety of formats:
      - sklearn/joblib .pkl/.joblib (pipelines or wrappers OK)
      - XGBoost native .json/.ubj/.bin
      - LightGBM native .txt (or text-like)
    Returns a Python object; use _unwrap_estimator() before predicting.
    """
    p = Path(p)
    if not p.exists():
        raise FileNotFoundError(p)

    # sklearn/joblib wrappers or pipelines
    if p.suffix in (".pkl", ".joblib"):
        return _joblib_load(p)

    # XGBoost native
    if p.suffix in (".json", ".ubj", ".bin") and _is_xgb_json(p):
        import xgboost as xgb
        return xgb.Booster(model_file=str(p))

    # LightGBM native
    if p.suffix == ".txt" or _is_lgb_text(p):
        import lightgbm as lgb
        return lgb.Booster(model_file=str(p))

    # Last attempts: try XGB then LGB as generic boosters
    try:
        import xgboost as xgb
        return xgb.Booster(model_file=str(p))
    except Exception:
        import lightgbm as lgb
        return lgb.Booster(model_file=str(p))

def _unwrap_estimator(obj):
    """
    Given a possibly complex sklearn object, try to retrieve the
    underlying estimator with predict/predict_proba — or a native Booster.
    """
    # Already usable?
    if hasattr(obj, "predict_proba") or hasattr(obj, "predict"):
        return obj

    # sklearn Pipeline
    if hasattr(obj, "steps"):
        for _, step in getattr(obj, "steps"):
            est = _unwrap_estimator(step)
            if est is not None:
                return est

    # Common attribute names in custom wrappers
    for attr in ("model_", "model", "estimator_", "estimator",
                 "clf", "base_model", "gbm", "xgb", "lgbm"):
        if hasattr(obj, attr):
            return _unwrap_estimator(getattr(obj, attr))

    # Dict-like containers
    if isinstance(obj, dict):
        for v in obj.values():
            est = _unwrap_estimator(v)
            if est is not None:
                return est

    # Fallback to the original object
    return obj

def _as_proba_from_estimator(m, X):
    """
    Try multiple strategies to get class-1 probabilities from an sklearn model.
    """
    # sklearn-style predict_proba
    if hasattr(m, "predict_proba"):
        return np.asarray(m.predict_proba(X)[:, 1]).reshape(-1)

    # decision_function + sigmoid as fallback
    if hasattr(m, "decision_function"):
        import scipy.special as sps
        z = np.asarray(m.decision_function(X)).reshape(-1)
        return sps.expit(z)

    # raw predict (maybe already proba-ish)
    if hasattr(m, "predict"):
        y = np.asarray(m.predict(X)).reshape(-1)
        # If values not in [0,1], try to squish
        if (y.min() < 0) or (y.max() > 1):
            y = (y - y.min()) / (y.max() - y.min() + 1e-9)
        return y

    raise TypeError("Estimator does not support predict/predict_proba/decision_function")

def _as_proba_from_booster(m, X):
    """
    Handle XGBoost or LightGBM Boosters with matrix conversion.
    """
    # XGBoost Booster
    try:
        import xgboost as xgb
        if isinstance(m, xgb.Booster):
            dm = xgb.DMatrix(X)
            y = m.predict(dm)
            return np.asarray(y).reshape(-1)
    except Exception:
        pass

    # LightGBM Booster
    try:
        import lightgbm as lgb
        if isinstance(m, lgb.Booster):
            y = m.predict(X)
            return np.asarray(y).reshape(-1)
    except Exception:
        pass

    return None

def predict_lgbm(model_path, X):
    m = _safe_load_model(model_path)
    m = _unwrap_estimator(m)

    # If it's a Booster, use its API
    y = _as_proba_from_booster(m, X)
    if y is not None:
        return y

    # Otherwise treat as sklearn-style estimator
    return _as_proba_from_estimator(m, X)

def predict_xgb(model_path, X):
    m = _safe_load_model(model_path)
    m = _unwrap_estimator(m)

    # If it's a Booster, use its API
    y = _as_proba_from_booster(m, X)
    if y is not None:
        return y

    # Otherwise treat as sklearn-style estimator
    return _as_proba_from_estimator(m, X)
