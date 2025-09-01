# models/xgboost_model.py
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, f1_score, accuracy_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier
import joblib

FEATURES = ["depth","width","enamel_cracks","occlusal_load","carious_lesion",
            "opposing_type","adjacent_teeth","age_range","cervical_lesion"]

LABEL_HARD = "y_majority"
SAMPLE_WEIGHT = "weight"

THRESHOLD_FILE = "xgb_threshold.json"
MODEL_FILE = "xgb_classifier_pipeline.joblib"

# depth(+), width(-), cracks(+), load(+), lesion(+), opposing(?), adjacent(?), age(?), cervical(+)
DEFAULT_MONO = (1, -1, 1, 1, 1, 0, 0, 0, 1)

# -------- Feature engineering --------
class DomainFeatures(BaseEstimator, TransformerMixin):
    """Add a few interpretable interaction features that often help accuracy."""
    def __init__(self, base_cols: list[str]):
        self.base_cols = base_cols
    def fit(self, X, y=None):
        return self
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        # boolean combos
        df["deep_and_thin"] = ((df["depth"]==1) & (df["width"]==0)).astype(int)
        df["deep_or_cracks"] = ((df["depth"]==1) | (df["enamel_cracks"]==1)).astype(int)
        df["load_implant"]   = ((df["occlusal_load"]==1) & (df["opposing_type"]==3)).astype(int)
        df["risk_plus_cervical"] = ((df["carious_lesion"]==1) & (df["cervical_lesion"]==1)).astype(int)
        df["stable_wall"] = ((df["width"]==1) & (df["enamel_cracks"]==0) & (df["occlusal_load"]==0)).astype(int)
        # numeric interactions (carious_lesion is -1/0/1)
        df["depth_x_load"] = (df["depth"]*df["occlusal_load"]).astype(int)
        df["depth_x_risk"] = (df["depth"]*df["carious_lesion"]).astype(int)
        return df

class DataFrameImputer(BaseEstimator, TransformerMixin):
    def __init__(self, strategy="most_frequent"):
        self.imputer = SimpleImputer(strategy=strategy)
        self.feature_names_ = None
    def fit(self, X, y=None):
        self.feature_names_ = X.columns.tolist() if isinstance(X, pd.DataFrame) else None
        self.imputer.fit(X); return self
    def transform(self, X):
        X_imp = self.imputer.transform(X)
        cols = self.feature_names_ or (X.columns.tolist() if isinstance(X, pd.DataFrame) else [f"f{i}" for i in range(X.shape[1])])
        return pd.DataFrame(X_imp, columns=cols, index=(X.index if isinstance(X, pd.DataFrame) else None))

class ImputerThenModel:
    def __init__(self, fe: DomainFeatures, imputer: DataFrameImputer, calibrated_model):
        self.fe = fe
        self.imputer = imputer
        self.calibrated_model = calibrated_model
    def _prep(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.imputer.transform(self.fe.transform(X))
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.calibrated_model.predict_proba(self._prep(X))
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

def load_dataset(data_path: Path) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    missing = [c for c in FEATURES + [LABEL_HARD, SAMPLE_WEIGHT] if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")
    return df

def _find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray, metric="balanced_accuracy") -> Tuple[float, float]:
    grid = np.linspace(0.05, 0.95, 181)
    best_t, best_m = 0.5, -1.0
    for t in grid:
        y_pred = (y_prob >= t).astype(int)
        if metric == "balanced_accuracy":
            m = balanced_accuracy_score(y_true, y_pred)
        elif metric == "f1":
            m = f1_score(y_true, y_pred, zero_division=0)
        elif metric == "accuracy":
            m = accuracy_score(y_true, y_pred)
        else:
            raise ValueError("metric must be one of: balanced_accuracy, f1, accuracy")
        if m > best_m:
            best_m, best_t = m, t
    return float(best_t), float(best_m)

def train_xgb(
    data_path: Path,
    output_dir: Path,
    random_state: int = 42,
    test_size_val: float = 0.20,
    calibration: str = "sigmoid",
    tune_metric: str = "accuracy",          # <- default to accuracy for your goal
    consensus_power: float = 0.7,
    use_monotone: bool = False,             # <- off by default (wasn’t helping)
    min_weight: float = 0.0,                # <- drop very ambiguous rows (< this) from TRAINING
    use_domain_features: bool = True,       # <- turn engineered features on
) -> Tuple[ImputerThenModel, Dict[str, Any]]:
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    df = load_dataset(Path(data_path))
    assert "split" in df.columns, "Expected a 'split' column."
    df_train = df[df["split"].astype(str).str.lower() == "train"].copy()

    # (1) optional: drop highly ambiguous rows from training only
    if min_weight > 0:
        before = len(df_train)
        df_train = df_train[df_train[SAMPLE_WEIGHT].fillna(0) >= float(min_weight)].copy()
        after = len(df_train)
        print(f"[info] dropped {before - after} low-consensus rows (weight < {min_weight}) from TRAIN only")

    X = df_train[FEATURES].copy()
    y = df_train[LABEL_HARD].astype(int).values
    w_consensus = df_train[SAMPLE_WEIGHT].fillna(1.0).astype(float).values

    # (2) weights: soften + class-balance + normalize
    w_consensus = np.power(w_consensus, consensus_power)
    cls_w = compute_class_weight(class_weight="balanced", classes=np.array([0,1]), y=y)
    w_final = w_consensus * np.where(y == 1, cls_w[1], cls_w[0])
    w_final = w_final / (np.mean(w_final) if np.mean(w_final) > 0 else 1.0)

    # (3) split
    X_tr, X_val, y_tr, y_val, w_tr, w_val = train_test_split(
        X, y, w_final, test_size=test_size_val, random_state=random_state, stratify=y
    )

    # (4) domain features + impute
    fe = DomainFeatures(FEATURES) if use_domain_features else DomainFeatures([])
    X_tr_fe = fe.transform(X_tr)
    X_val_fe = fe.transform(X_val)

    imputer = DataFrameImputer(strategy="most_frequent").fit(X_tr_fe)
    X_tr_imp = imputer.transform(X_tr_fe)
    X_val_imp = imputer.transform(X_val_fe)

    # (5) model
    xgb_kwargs = dict(
        n_estimators=1200, learning_rate=0.03, max_depth=3,
        min_child_weight=5, gamma=1.0,
        subsample=0.9, colsample_bytree=0.9,
        reg_lambda=1.0, reg_alpha=0.5,
        objective="binary:logistic",
        eval_metric=["logloss", "auc"],
        random_state=random_state, n_jobs=-1,
        tree_method="hist", early_stopping_rounds=120,
    )
    if use_monotone:
        xgb_kwargs["monotone_constraints"] = f"({','.join(map(str, DEFAULT_MONO))})"

    xgb = XGBClassifier(**xgb_kwargs)
    xgb.fit(
        X_tr_imp, y_tr,
        sample_weight=w_tr,
        eval_set=[(X_val_imp, y_val)],
        sample_weight_eval_set=[w_val],
        verbose=False,
    )

    # (6) calibration + bundle
    cal = CalibratedClassifierCV(estimator=xgb, method=calibration, cv="prefit")
    cal.fit(X_val_imp, y_val, sample_weight=w_val)
    bundle = ImputerThenModel(fe, imputer, cal)

    # (7) tune threshold for YOUR objective (accuracy by default)
    val_prob = bundle.predict_proba(X_val)[:, 1]
    best_thr, best_metric = _find_best_threshold(y_val, val_prob, metric=tune_metric)

    # (8) save
    model_path = output_dir / MODEL_FILE
    joblib.dump(bundle, model_path)
    thr_path = output_dir / THRESHOLD_FILE
    with open(thr_path, "w") as f:
        json.dump({"threshold": best_thr, "metric": tune_metric, "metric_val": best_metric}, f, indent=2)

    info = {
        "model_path": str(model_path),
        "threshold_path": str(thr_path),
        "best_threshold": best_thr,
        "val_metric": best_metric,
        "val_metric_name": tune_metric,
        "n_train_rows": int(len(df_train)),
        "used_split": True,
    }
    return bundle, info

def _default_paths():
    here = Path(__file__).resolve(); root = here.parents[1]
    return root / "data" / "excel" / "data_processed.csv", root / "models" / "outputs"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train enhanced XGBoost on tabular features (accuracy-focused).")
    dflt_data, dflt_out = _default_paths()
    parser.add_argument("--data", type=Path, default=dflt_data)
    parser.add_argument("--out", type=Path, default=dflt_out)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--calibration", choices=["sigmoid","isotonic"], default="sigmoid")
    parser.add_argument("--tune-metric", choices=["accuracy","balanced_accuracy","f1"], default="accuracy")
    parser.add_argument("--consensus-power", type=float, default=0.7)
    parser.add_argument("--no-monotone", action="store_true")
    parser.add_argument("--min-weight", type=float, default=0.0)
    parser.add_argument("--no-domain-features", action="store_true")
    args = parser.parse_args()

    bundle, info = train_xgb(
        args.data, args.out, random_state=args.seed,
        calibration=args.calibration, tune_metric=args.tune_metric,
        consensus_power=args.consensus_power, use_monotone=not args.no_monotone,
        min_weight=args.min_weight, use_domain_features=not args.no_domain_features,
    )
    print("✅ XGBoost trained + calibrated.")
    print(f"   Train rows:          {info['n_train_rows']}")
    print(f"   Model saved to:      {info['model_path']}")
    print(f"   Threshold saved to:  {info['threshold_path']}")
    print(f"   Best threshold (val): {info['best_threshold']:.3f} ({info['val_metric_name']}={info['val_metric']:.3f})")
