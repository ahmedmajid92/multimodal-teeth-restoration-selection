# tests/evaluate_models.py
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Dict, Tuple
import numpy as np, pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, brier_score_loss, confusion_matrix,
                             balanced_accuracy_score)
import joblib

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH_DEFAULT = ROOT / "data" / "excel" / "data_processed.csv"
OUTDIR_DEFAULT = ROOT / "models" / "outputs"

FEATURES = ["depth","width","enamel_cracks","occlusal_load","carious_lesion",
            "opposing_type","adjacent_teeth","age_range","cervical_lesion"]
HARD_LABEL = "y_majority"

# --- Missing classes for compatibility with saved models ---
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

class LGBMProbWrapper:
    def __init__(self, fe: DomainFeatures, imputer: SimpleImputer, model):
        self.fe = fe
        self.imputer = imputer
        self.model = model
    def _prep(self, X: pd.DataFrame):
        return self.imputer.transform(self.fe.transform(X))
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        p = self.model.predict(self._prep(X))
        return np.clip(p, 0.0, 1.0)
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = self.predict(X)
        return np.column_stack([1.0 - p, p])

# --- data helpers ---
def load_data(p: Path) -> pd.DataFrame:
    if not p.exists(): raise FileNotFoundError(p)
    df = pd.read_csv(p)
    miss = set(FEATURES + [HARD_LABEL]) - set(df.columns)
    if miss: raise KeyError(miss)
    if "split" not in df.columns: raise KeyError("Expected 'split' column.")
    return df

def select_test(df: pd.DataFrame):
    t = df[df["split"].astype(str).str.lower() == "test"].copy()
    if t.empty: raise ValueError("No test rows.")
    return t[FEATURES].copy(), t[HARD_LABEL].astype(int).values

def select_train(df: pd.DataFrame):
    tr = df[df["split"].astype(str).str.lower() == "train"].copy()
    if tr.empty: raise ValueError("No train rows.")
    return tr[FEATURES].copy(), tr[HARD_LABEL].astype(int).values

# --- metrics ---
def metrics_binary(y, p, thr):
    yhat = (p >= thr).astype(int)
    out = {"threshold": float(thr),
           "accuracy": float(accuracy_score(y, yhat)),
           "precision": float(precision_score(y, yhat, zero_division=0)),
           "recall": float(recall_score(y, yhat, zero_division=0)),
           "f1": float(f1_score(y, yhat, zero_division=0)),
           "brier": float(brier_score_loss(y, p)),
           "roc_auc": None, "pr_auc": None,
           "confusion_matrix": confusion_matrix(y, yhat).tolist()}
    if len(np.unique(y)) == 2:
        out["roc_auc"] = float(roc_auc_score(y, p))
        out["pr_auc"] = float(average_precision_score(y, p))
    return out

def _load_thr(outdir: Path):
    p = outdir / "xgb_threshold.json"
    if p.exists():
        try:
            return float(json.load(open(p))["threshold"])
        except Exception: pass
    return None

# --- tuning helpers for blend ---
def _best_threshold(y_true: np.ndarray, y_prob: np.ndarray, metric: str = "f1") -> Tuple[float, float]:
    grid = np.linspace(0.05, 0.95, 181)
    best_t, best_m = 0.5, -1.0
    for t in grid:
        y_pred = (y_prob >= t).astype(int)
        if metric == "balanced_accuracy":
            m = balanced_accuracy_score(y_true, y_pred)
        elif metric == "accuracy":
            m = accuracy_score(y_true, y_pred)
        else:  # f1
            m = f1_score(y_true, y_pred, zero_division=0)
        if m > best_m:
            best_m, best_t = m, t
    return float(best_t), float(best_m)

def _tune_blend_threshold(xgb_prob_tr: np.ndarray, lgb_prob_tr: np.ndarray, y_tr: np.ndarray,
                          alpha: float, metric: str) -> Tuple[float, float]:
    blend_tr = alpha * xgb_prob_tr + (1 - alpha) * lgb_prob_tr
    return _best_threshold(y_tr, blend_tr, metric)

def _tune_blend_alpha(xgb_prob_tr: np.ndarray, lgb_prob_tr: np.ndarray, y_tr: np.ndarray,
                      metric: str) -> Tuple[float, float, float]:
    best_alpha, best_thr, best_score = 0.5, 0.5, -1.0
    for a in np.linspace(0.0, 1.0, 21):
        thr, score = _tune_blend_threshold(xgb_prob_tr, lgb_prob_tr, y_tr, a, metric)
        if score > best_score:
            best_alpha, best_thr, best_score = a, thr, score
    return float(best_alpha), float(best_thr), float(best_score)

def main():
    ap = argparse.ArgumentParser(description="Evaluate trained tabular models on test split.")
    ap.add_argument("--data", type=Path, default=DATA_PATH_DEFAULT)
    ap.add_argument("--outdir", type=Path, default=OUTDIR_DEFAULT)
    ap.add_argument("--model", choices=["xgb","lgbm","both","blend"], required=True)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--blend-alpha", type=float, default=0.5, help="Weight for XGB in blended prob.")
    ap.add_argument("--tune", choices=["none","threshold","alpha","both"], default="threshold",
                    help="For blend: tune threshold, alpha, or both on the train split.")
    ap.add_argument("--tune-metric", choices=["f1","balanced_accuracy","accuracy"], default="f1",
                    help="Objective for tuning on the train split.")
    args = ap.parse_args()

    outdir = args.outdir; outdir.mkdir(parents=True, exist_ok=True)
    df = load_data(args.data)
    X_test, y_test = select_test(df)

    reports = {}
    xgb_prob = lgb_prob = None

    # XGB
    if args.model in ("xgb","both","blend"):
        xgb = joblib.load(outdir / "xgb_classifier_pipeline.joblib")
        xgb_prob = xgb.predict_proba(X_test)[:, 1]
        thr = args.threshold if args.threshold is not None else (_load_thr(outdir) or 0.5)
        rep = metrics_binary(y_test, xgb_prob, thr)
        pd.DataFrame({"y_true": y_test, "score_prob_indirect": xgb_prob,
                      "y_pred": (xgb_prob >= thr).astype(int)}).to_csv(outdir / "xgb_test_predictions.csv", index=False)
        reports["xgboost"] = rep
        print("=== XGBoost evaluation ==="); print(json.dumps(rep, indent=2))

    # LGBM
    if args.model in ("lgbm","both","blend"):
        lgb = joblib.load(outdir / "lgbm_regressor_pipeline.joblib")
        lgb_prob = np.clip(lgb.predict(X_test), 0.0, 1.0)
        thr = args.threshold if args.threshold is not None else 0.5
        rep = metrics_binary(y_test, lgb_prob, thr)
        pd.DataFrame({"y_true": y_test, "score_prob_indirect": lgb_prob,
                      "y_pred": (lgb_prob >= thr).astype(int)}).to_csv(outdir / "lgbm_test_predictions.csv", index=False)
        reports["lightgbm"] = rep
        print("=== LightGBM evaluation ==="); print(json.dumps(rep, indent=2))

    # Blend (with optional tuning on train split)
    if args.model == "blend":
        if xgb_prob is None or lgb_prob is None:
            xgb = joblib.load(outdir / "xgb_classifier_pipeline.joblib")
            lgb = joblib.load(outdir / "lgbm_regressor_pipeline.joblib")

        X_train, y_train = select_train(df)
        xgb_prob_tr = xgb.predict_proba(X_train)[:, 1]
        lgb_prob_tr = np.clip(lgb.predict(X_train), 0.0, 1.0)

        alpha = float(args.blend_alpha)
        tuned_thr = None
        tuned_alpha = None
        tuned_score = None

        if args.tune in ("threshold","both"):
            tuned_thr, tuned_score = _tune_blend_threshold(xgb_prob_tr, lgb_prob_tr, y_train, alpha, args.tune_metric)
        if args.tune in ("alpha","both"):
            tuned_alpha, tuned_thr2, tuned_score2 = _tune_blend_alpha(xgb_prob_tr, lgb_prob_tr, y_train, args.tune_metric)
            if args.tune == "alpha":
                alpha, tuned_thr, tuned_score = tuned_alpha, tuned_thr2, tuned_score2
            else:
                alpha, tuned_thr, tuned_score = tuned_alpha, tuned_thr2, tuned_score2

        if tuned_thr is None:
            tuned_thr = args.threshold if args.threshold is not None else (_load_thr(outdir) or 0.5)

        blended_test = alpha * xgb_prob + (1 - alpha) * lgb_prob
        rep = metrics_binary(y_test, blended_test, tuned_thr)
        rep.update({"alpha": float(alpha), "tuned_threshold": float(tuned_thr),
                    "tune_metric": args.tune_metric, "tune_mode": args.tune})

        pd.DataFrame({
            "y_true": y_test,
            "score_prob_indirect": blended_test,
            "y_pred": (blended_test >= tuned_thr).astype(int),
        }).to_csv(outdir / "blend_test_predictions.csv", index=False)

        with open(outdir / "blend_params.json", "w") as f:
            json.dump({"alpha": float(alpha),
                       "threshold": float(tuned_thr),
                       "metric": args.tune_metric,
                       "mode": args.tune,
                       "train_score": float(tuned_score) if tuned_score is not None else None}, f, indent=2)

        reports["blend"] = rep
        print(f"=== Blended (alpha={alpha:.2f}, tuned on train: {args.tune}, metric={args.tune_metric}) ===")
        print(json.dumps(rep, indent=2))

    with open(outdir / ("metrics_" + args.model + ".json"), "w") as f:
        json.dump(reports, f, indent=2)
    print("Metrics saved.")
    
if __name__ == "__main__":
    main()
