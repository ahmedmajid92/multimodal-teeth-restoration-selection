# models/stack_blend.py
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, brier_score_loss,
    roc_auc_score, average_precision_score, confusion_matrix, balanced_accuracy_score
)
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier
from lightgbm import LGBMRegressor
import joblib

# ----- Project defaults -----
def _paths():
    here = Path(__file__).resolve()
    root = here.parents[1]
    data = root / "data" / "excel" / "data_processed.csv"
    out  = root / "models" / "outputs"
    return data, out

FEATURES = ["depth","width","enamel_cracks","occlusal_load","carious_lesion",
            "opposing_type","adjacent_teeth","age_range","cervical_lesion"]
Y_HARD = "y_majority"
W_COL  = "weight"

# ---------- Utilities ----------
def metrics(y_true: np.ndarray, prob: np.ndarray, thr: float) -> Dict:
    pred = (prob >= thr).astype(int)
    out = dict(
        threshold=float(thr),
        accuracy=float(accuracy_score(y_true, pred)),
        precision=float(precision_score(y_true, pred, zero_division=0)),
        recall=float(recall_score(y_true, pred, zero_division=0)),
        f1=float(f1_score(y_true, pred, zero_division=0)),
        brier=float(brier_score_loss(y_true, prob)),
        roc_auc=None,
        pr_auc=None,
        confusion_matrix=confusion_matrix(y_true, pred).tolist(),
    )
    if len(np.unique(y_true)) == 2:
        out["roc_auc"] = float(roc_auc_score(y_true, prob))
        out["pr_auc"]  = float(average_precision_score(y_true, prob))
    return out

def tune_threshold(y: np.ndarray, p: np.ndarray, objective: str = "accuracy") -> Tuple[float, float]:
    grid = np.linspace(0.05, 0.95, 181)
    best_t, best_s = 0.5, -1.0
    for t in grid:
        yhat = (p >= t).astype(int)
        if objective == "balanced_accuracy":
            s = balanced_accuracy_score(y, yhat)
        elif objective == "f1":
            s = f1_score(y, yhat, zero_division=0)
        else:
            s = accuracy_score(y, yhat)
        if s > best_s:
            best_s, best_t = s, t
    return float(best_t), float(best_s)

class DomainFeatures(BaseEstimator, TransformerMixin):
    """Same interactions we used in the boosted models."""
    def __init__(self, cols: List[str]): self.cols = cols
    def fit(self, X, y=None): return self
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        df["deep_and_thin"]      = ((df["depth"]==1) & (df["width"]==0)).astype(int)
        df["deep_or_cracks"]     = ((df["depth"]==1) | (df["enamel_cracks"]==1)).astype(int)
        df["load_implant"]       = ((df["occlusal_load"]==1) & (df["opposing_type"]==3)).astype(int)
        df["risk_plus_cervical"] = ((df["carious_lesion"]==1) & (df["cervical_lesion"]==1)).astype(int)
        df["stable_wall"]        = ((df["width"]==1) & (df["enamel_cracks"]==0) & (df["occlusal_load"]==0)).astype(int)
        df["depth_x_load"]       = (df["depth"]*df["occlusal_load"]).astype(int)
        df["depth_x_risk"]       = (df["depth"]*df["carious_lesion"]).astype(int)
        return df

# ---------- Main training/eval ----------
def main():
    ap = argparse.ArgumentParser(description="Stacked blend: OOF XGB+LGBM -> Logistic meta (accuracy-tuned).")
    dflt_data, dflt_out = _paths()
    ap.add_argument("--data", type=Path, default=dflt_data)
    ap.add_argument("--out", type=Path, default=dflt_out)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--consensus-power", type=float, default=0.6, help="Weight shaping exponent.")
    ap.add_argument("--min-weight", type=float, default=0.15, help="Drop train rows with weight below this.")
    ap.add_argument("--tune-metric", choices=["accuracy","balanced_accuracy","f1"], default="accuracy")
    args = ap.parse_args()

    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data)
    assert "split" in df.columns, "Expected 'split' column."
    train_df = df[df["split"].astype(str).str.lower()=="train"].copy()
    test_df  = df[df["split"].astype(str).str.lower()=="test"].copy()

    # drop very ambiguous rows from TRAIN only
    if args.min_weight > 0:
        n0 = len(train_df)
        train_df = train_df[train_df[W_COL].fillna(0) >= args.min_weight].copy()
        print(f"[stack] dropped {n0-len(train_df)} low-consensus rows (<{args.min_weight})")

    X_tr = train_df[FEATURES].copy()
    y_tr = train_df[Y_HARD].astype(int).values
    w    = train_df[W_COL].fillna(1.0).astype(float).values

    # weight shaping + class balance + normalization
    w = np.power(w, args.consensus_power)
    cls_w = compute_class_weight(class_weight="balanced", classes=np.array([0,1]), y=y_tr)
    w = w * np.where(y_tr==1, cls_w[1], cls_w[0])
    w = w / (np.mean(w) if np.mean(w)>0 else 1.0)

    X_te = test_df[FEATURES].copy()
    y_te = test_df[Y_HARD].astype(int).values

    # feature engineering + imputer (per fold)
    fe = DomainFeatures(FEATURES)

    # base learners (same bias as your current strong configs)
    def make_xgb():
        return XGBClassifier(
            n_estimators=1200, learning_rate=0.03, max_depth=3,
            min_child_weight=5, gamma=1.0, subsample=0.9, colsample_bytree=0.9,
            reg_lambda=1.0, reg_alpha=0.5, objective="binary:logistic",
            eval_metric="logloss", random_state=args.seed, n_jobs=-1, tree_method="hist",
        )
    def make_lgbm():
        return LGBMRegressor(
            n_estimators=1200, learning_rate=0.03, num_leaves=31, min_child_samples=20,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.9, reg_lambda=1.0, reg_alpha=0.0,
            random_state=args.seed, n_jobs=-1, verbose=-1  # Add verbose=-1 here to suppress output
        )

    # OOF arrays and test blending accumulators
    oof_xgb = np.zeros(len(X_tr), dtype=float)
    oof_lgb = np.zeros(len(X_tr), dtype=float)
    te_pred_xgb = np.zeros(len(X_te), dtype=float)
    te_pred_lgb = np.zeros(len(X_te), dtype=float)

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_tr, y_tr), 1):
        X_tr_f, X_va_f = X_tr.iloc[tr_idx], X_tr.iloc[va_idx]
        y_tr_f, y_va_f = y_tr[tr_idx], y_tr[va_idx]
        w_tr_f, w_va_f = w[tr_idx], w[va_idx]

        # FE + impute
        X_tr_f = fe.transform(X_tr_f)
        X_va_f = fe.transform(X_va_f)
        X_te_f = fe.transform(X_te)

        imp = SimpleImputer(strategy="most_frequent").fit(X_tr_f)
        X_tr_imp = imp.transform(X_tr_f)
        X_va_imp = imp.transform(X_va_f)
        X_te_imp = imp.transform(X_te_f)

        # XGB
        xgb = make_xgb()
        xgb.fit(X_tr_imp, y_tr_f, sample_weight=w_tr_f, verbose=False)
        oof_xgb[va_idx] = xgb.predict_proba(X_va_imp)[:,1]
        te_pred_xgb += xgb.predict_proba(X_te_imp)[:,1] / args.folds

        # LGBM (regressor; clip to [0,1]) - Remove verbose parameter
        lgb = make_lgbm()
        lgb.fit(X_tr_imp, y_tr_f, sample_weight=w_tr_f)  # Removed verbose=False
        oof_lgb[va_idx] = np.clip(lgb.predict(X_va_imp), 0.0, 1.0)
        te_pred_lgb += np.clip(lgb.predict(X_te_imp), 0.0, 1.0) / args.folds

        print(f"[stack] fold {fold}/{args.folds} done")

    # Meta-learner on OOF
    Z_tr = np.column_stack([oof_xgb, oof_lgb])
    meta = LogisticRegression(solver="liblinear", class_weight="balanced", random_state=args.seed)
    meta.fit(Z_tr, y_tr, sample_weight=w)

    # Tune threshold for accuracy on OOF meta probs
    p_tr_meta = meta.predict_proba(Z_tr)[:,1]
    thr, score = tune_threshold(y_tr, p_tr_meta, objective=args.tune_metric)

    # Evaluate on test via averaged base predictions
    Z_te = np.column_stack([te_pred_xgb, te_pred_lgb])
    p_te_meta = meta.predict_proba(Z_te)[:,1]
    rep = metrics(y_te, p_te_meta, thr)

    # Save artifacts
    outdir = Path(args.out)
    joblib.dump({"meta": meta}, outdir / "stack_meta.joblib")
    with open(outdir / "stack_params.json", "w") as f:
        json.dump({"threshold": float(thr), "tune_metric": args.tune_metric,
                   "oof_score": float(score)}, f, indent=2)

    # Save predictions + metrics
    pd.DataFrame({
        "y_true": y_te,
        "p_xgb": te_pred_xgb,
        "p_lgbm": te_pred_lgb,
        "p_meta": p_te_meta,
        "y_pred": (p_te_meta >= thr).astype(int),
    }).to_csv(outdir / "stack_test_predictions.csv", index=False)

    with open(outdir / "metrics_stack.json", "w") as f:
        json.dump(rep, f, indent=2)

    print("=== Stacked (Logistic on OOF XGB+LGBM) ===")
    print(json.dumps(rep, indent=2))
    print(f"Artifacts saved to: {outdir}")

if __name__ == "__main__":
    main()
