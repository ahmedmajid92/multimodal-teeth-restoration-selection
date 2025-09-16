# src/tabular/predict_tabular.py
from pathlib import Path
import sys
import json
import numpy as np

# ---------------------------
# Dynamic unpickle shims
# ---------------------------
def _install_main_dynamic_attr():
    """
    Some pipelines were pickled from a __main__ script and contain custom
    classes (e.g., ImputerThenModel, DomainFeatures). When unpickling,
    pickle asks for __main__.ClassName. If it doesn't exist, we create a
    minimal placeholder class on the fly via module-level __getattr__.
    """
    main = sys.modules.get("__main__")
    if main is None:
        return
    if not hasattr(main, "__getattr__"):
        def __getattr__(name):
            # Create a minimal dummy class and memoize it
            cls = type(name, (), {})
            setattr(main, name, cls)
            return cls
        setattr(main, "__getattr__", __getattr__)

def _joblib_load(p: Path):
    _install_main_dynamic_attr()
    import joblib
    return joblib.load(p)

# ---------------------------
# Format detection helpers
# ---------------------------
def _is_xgb_json(path: Path) -> bool:
    try:
        with open(path, "r", encoding="utf-8") as f:
            j = json.load(f)
        return "learner" in j or "gradient_booster" in j or "version" in j
    except Exception:
        return False

def _is_lgb_text(path: Path) -> bool:
    try:
        head = path.read_text(encoding="utf-8", errors="ignore")[:512].lower()
        return "feature_names" in head or head.startswith("tree") or "num_class" in head
    except Exception:
        return False

def _safe_load_model(p):
    """
    Load a model from:
      - sklearn/joblib .pkl/.joblib (pipelines/wrappers OK)
      - XGBoost native .json/.ubj/.bin
      - LightGBM native .txt (or text-like)
    Returns a Python object; call _unwrap_estimator() before predicting.
    """
    p = Path(p)
    if not p.exists():
        raise FileNotFoundError(p)

    if p.suffix in (".pkl", ".joblib"):
        return _joblib_load(p)

    if p.suffix in (".json", ".ubj", ".bin") and _is_xgb_json(p):
        import xgboost as xgb
        return xgb.Booster(model_file=str(p))

    if p.suffix == ".txt" or _is_lgb_text(p):
        import lightgbm as lgb
        return lgb.Booster(model_file=str(p))

    # Fallback attempts
    try:
        import xgboost as xgb
        return xgb.Booster(model_file=str(p))
    except Exception:
        import lightgbm as lgb
        return lgb.Booster(model_file=str(p))

# ---------------------------
# Deep unwrap
# ---------------------------
def _is_usable_estimator(obj):
    # direct sklearn-like
    if hasattr(obj, "predict_proba") or hasattr(obj, "decision_function") or hasattr(obj, "predict"):
        return True
    # native boosters
    try:
        import xgboost as xgb
        if isinstance(obj, xgb.Booster):
            return True
    except Exception:
        pass
    try:
        import lightgbm as lgb
        if isinstance(obj, lgb.Booster):
            return True
    except Exception:
        pass
    return False

def _unwrap_estimator(obj, _depth=0, _seen=None):
    """
    Aggressively search inside wrappers/pipelines to find a usable estimator.
    Traverses:
      - sklearn Pipeline.steps
      - ColumnTransformer.transformers
      - common attributes (model, estimator, ...; with and without trailing '_')
      - dicts/lists/tuples/sets
      - arbitrary instance __dict__
    """
    if obj is None:
        return None
    if _depth > 8:
        return None
    if _seen is None:
        _seen = set()
    oid = id(obj)
    if oid in _seen:
        return None
    _seen.add(oid)

    if _is_usable_estimator(obj):
        return obj

    # sklearn Pipeline
    if hasattr(obj, "steps"):
        try:
            for _, step in getattr(obj, "steps"):
                est = _unwrap_estimator(step, _depth+1, _seen)
                if est is not None:
                    return est
        except Exception:
            pass

    # ColumnTransformer
    if hasattr(obj, "transformers"):
        try:
            for _, trans, _cols in getattr(obj, "transformers"):
                est = _unwrap_estimator(trans, _depth+1, _seen)
                if est is not None:
                    return est
        except Exception:
            pass

    # common wrapper attribute names
    for attr in ("model_", "model", "estimator_", "estimator",
                 "final_estimator", "final_estimator_",
                 "best_estimator_", "base_estimator", "base_estimator_",
                 "clf", "gbm", "xgb", "lgbm", "regressor", "classifier"):
        if hasattr(obj, attr):
            try:
                est = _unwrap_estimator(getattr(obj, attr), _depth+1, _seen)
                if est is not None:
                    return est
            except Exception:
                pass

    # dict-like containers
    if isinstance(obj, dict):
        for v in obj.values():
            est = _unwrap_estimator(v, _depth+1, _seen)
            if est is not None:
                return est

    # list / tuple / set
    if isinstance(obj, (list, tuple, set)):
        for v in obj:
            est = _unwrap_estimator(v, _depth+1, _seen)
            if est is not None:
                return est

    # arbitrary instance attributes
    try:
        d = vars(obj)
        for v in d.values():
            est = _unwrap_estimator(v, _depth+1, _seen)
            if est is not None:
                return est
    except Exception:
        pass

    return None

# ---------------------------
# Predict utilities
# ---------------------------
def _as_proba_from_estimator(m, X):
    """
    Try multiple strategies to get class-1 probabilities from an sklearn model.
    Works for classifiers and regressors (regressed to probability).
    """
    # predict_proba (best)
    if hasattr(m, "predict_proba"):
        return np.asarray(m.predict_proba(X)[:, 1]).reshape(-1)

    # decision_function → sigmoid
    if hasattr(m, "decision_function"):
        import scipy.special as sps
        z = np.asarray(m.decision_function(X)).reshape(-1)
        return sps.expit(z)

    # raw predict (e.g., regressors); clamp to [0,1]
    if hasattr(m, "predict"):
        y = np.asarray(m.predict(X)).reshape(-1)
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
    inner = _unwrap_estimator(m)
    if inner is None:
        raise TypeError("Could not locate a usable estimator inside the loaded LGBM pipeline.")
    y = _as_proba_from_booster(inner, X)
    if y is not None:
        return y
    return _as_proba_from_estimator(inner, X)

def predict_xgb(model_path, X):
    m = _safe_load_model(model_path)
    inner = _unwrap_estimator(m)
    if inner is None:
        raise TypeError("Could not locate a usable estimator inside the loaded XGB pipeline.")
    y = _as_proba_from_booster(inner, X)
    if y is not None:
        return y
    return _as_proba_from_estimator(inner, X)
