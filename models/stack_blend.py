# models/stack_blend.py
from __future__ import annotations  # Enable postponed evaluation of annotations for type hints
import argparse, json  # Command line argument parsing and JSON handling
from pathlib import Path  # Object-oriented filesystem path handling
from typing import Dict, Tuple, List  # Type hints for function signatures

import numpy as np  # Numerical computing library
import pandas as pd  # Data manipulation and analysis library
from sklearn.base import BaseEstimator, TransformerMixin  # Base classes for creating custom transformers
from sklearn.impute import SimpleImputer  # Fill missing values with simple strategies
from sklearn.model_selection import StratifiedKFold, train_test_split  # Cross-validation and data splitting
from sklearn.linear_model import LogisticRegression  # Logistic regression for meta-learning
from sklearn.metrics import (  # Various evaluation metrics
    accuracy_score, precision_score, recall_score, f1_score, brier_score_loss,
    roc_auc_score, average_precision_score, confusion_matrix, balanced_accuracy_score
)
from sklearn.utils.class_weight import compute_class_weight  # Calculate balanced class weights
from xgboost import XGBClassifier  # XGBoost classifier for base learner
from lightgbm import LGBMRegressor  # LightGBM regressor for base learner
import joblib  # Model serialization and persistence

# ----- Project defaults -----
def _paths():
    here = Path(__file__).resolve()  # Get current script path
    root = here.parents[1]  # Navigate to project root (2 levels up)
    data = root / "data" / "excel" / "data_processed.csv"  # Default data file path
    out  = root / "models" / "outputs"  # Default output directory
    return data, out  # Return both paths

# Define the feature columns used for model training
FEATURES = ["depth","width","enamel_cracks","occlusal_load","carious_lesion",
            "opposing_type","adjacent_teeth","age_range","cervical_lesion"]
Y_HARD = "y_majority"  # Hard binary label column name
W_COL  = "weight"  # Sample weight column name

# ---------- Utilities ----------
def metrics(y_true: np.ndarray, prob: np.ndarray, thr: float) -> Dict:
    pred = (prob >= thr).astype(int)  # Convert probabilities to binary predictions using threshold
    out = dict(  # Create metrics dictionary
        threshold=float(thr),  # Store the threshold used
        accuracy=float(accuracy_score(y_true, pred)),  # Overall accuracy
        precision=float(precision_score(y_true, pred, zero_division=0)),  # Precision (TP/(TP+FP))
        recall=float(recall_score(y_true, pred, zero_division=0)),  # Recall (TP/(TP+FN))
        f1=float(f1_score(y_true, pred, zero_division=0)),  # F1 score (harmonic mean of precision/recall)
        brier=float(brier_score_loss(y_true, prob)),  # Brier score (lower is better)
        roc_auc=None,  # Initialize ROC AUC as None
        pr_auc=None,  # Initialize PR AUC as None
        confusion_matrix=confusion_matrix(y_true, pred).tolist(),  # Confusion matrix as list
    )
    if len(np.unique(y_true)) == 2:  # If binary classification (both classes present)
        out["roc_auc"] = float(roc_auc_score(y_true, prob))  # Calculate ROC AUC
        out["pr_auc"]  = float(average_precision_score(y_true, prob))  # Calculate Precision-Recall AUC
    return out  # Return complete metrics dictionary

def tune_threshold(y: np.ndarray, p: np.ndarray, objective: str = "accuracy") -> Tuple[float, float]:
    grid = np.linspace(0.05, 0.95, 181)  # Create threshold grid from 0.05 to 0.95 with 181 points
    best_t, best_s = 0.5, -1.0  # Initialize best threshold and score
    for t in grid:  # Iterate through all threshold values
        yhat = (p >= t).astype(int)  # Convert probabilities to predictions
        if objective == "balanced_accuracy":  # Calculate balanced accuracy
            s = balanced_accuracy_score(y, yhat)
        elif objective == "f1":  # Calculate F1 score
            s = f1_score(y, yhat, zero_division=0)
        else:  # Default to standard accuracy
            s = accuracy_score(y, yhat)
        if s > best_s:  # Update best if current score is better
            best_s, best_t = s, t
    return float(best_t), float(best_s)  # Return best threshold and corresponding score

class DomainFeatures(BaseEstimator, TransformerMixin):
    """Same interactions we used in the boosted models."""
    def __init__(self, cols: List[str]): self.cols = cols  # Store column names
    def fit(self, X, y=None): return self  # Fit method for sklearn compatibility
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # Transform data by adding interaction features
        df = X.copy()  # Create copy to avoid modifying original
        # Create domain-specific interaction features based on dental knowledge
        df["deep_and_thin"]      = ((df["depth"]==1) & (df["width"]==0)).astype(int)  # Deep cavity with thin walls
        df["deep_or_cracks"]     = ((df["depth"]==1) | (df["enamel_cracks"]==1)).astype(int)  # Deep or cracked
        df["load_implant"]       = ((df["occlusal_load"]==1) & (df["opposing_type"]==3)).astype(int)  # High load on implant
        df["risk_plus_cervical"] = ((df["carious_lesion"]==1) & (df["cervical_lesion"]==1)).astype(int)  # High risk + cervical
        df["stable_wall"]        = ((df["width"]==1) & (df["enamel_cracks"]==0) & (df["occlusal_load"]==0)).astype(int)  # Stable structure
        df["depth_x_load"]       = (df["depth"]*df["occlusal_load"]).astype(int)  # Depth-load interaction
        df["depth_x_risk"]       = (df["depth"]*df["carious_lesion"]).astype(int)  # Depth-risk interaction
        return df  # Return transformed dataframe

# ---------- Main training/eval ----------
def main():
    ap = argparse.ArgumentParser(description="Stacked blend: OOF XGB+LGBM -> Logistic meta (accuracy-tuned).")  # Create argument parser
    dflt_data, dflt_out = _paths()  # Get default paths
    ap.add_argument("--data", type=Path, default=dflt_data)  # Data file path argument
    ap.add_argument("--out", type=Path, default=dflt_out)  # Output directory argument
    ap.add_argument("--seed", type=int, default=42)  # Random seed argument
    ap.add_argument("--folds", type=int, default=5)  # Number of cross-validation folds
    ap.add_argument("--consensus-power", type=float, default=0.6, help="Weight shaping exponent.")  # Weight transformation power
    ap.add_argument("--min-weight", type=float, default=0.15, help="Drop train rows with weight below this.")  # Minimum weight threshold
    ap.add_argument("--tune-metric", choices=["accuracy","balanced_accuracy","f1"], default="accuracy")  # Metric for threshold tuning
    args = ap.parse_args()  # Parse command line arguments

    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)  # Create output directory

    df = pd.read_csv(args.data)  # Load dataset from CSV
    assert "split" in df.columns, "Expected 'split' column."  # Ensure split column exists
    train_df = df[df["split"].astype(str).str.lower()=="train"].copy()  # Filter training data
    test_df  = df[df["split"].astype(str).str.lower()=="test"].copy()  # Filter test data

    # drop very ambiguous rows from TRAIN only
    if args.min_weight > 0:  # If minimum weight threshold is set
        n0 = len(train_df)  # Count rows before filtering
        train_df = train_df[train_df[W_COL].fillna(0) >= args.min_weight].copy()  # Filter low-weight rows
        print(f"[stack] dropped {n0-len(train_df)} low-consensus rows (<{args.min_weight})")

    X_tr = train_df[FEATURES].copy()  # Extract training features
    y_tr = train_df[Y_HARD].astype(int).values  # Extract training labels as integers
    w    = train_df[W_COL].fillna(1.0).astype(float).values  # Extract sample weights

    # weight shaping + class balance + normalization
    w = np.power(w, args.consensus_power)  # Apply power transformation to weights
    cls_w = compute_class_weight(class_weight="balanced", classes=np.array([0,1]), y=y_tr)  # Calculate balanced class weights
    w = w * np.where(y_tr==1, cls_w[1], cls_w[0])  # Apply class weights to sample weights
    w = w / (np.mean(w) if np.mean(w)>0 else 1.0)  # Normalize weights to mean=1

    X_te = test_df[FEATURES].copy()  # Extract test features
    y_te = test_df[Y_HARD].astype(int).values  # Extract test labels

    # feature engineering + imputer (per fold)
    fe = DomainFeatures(FEATURES)  # Create feature engineering transformer

    # base learners (same bias as your current strong configs)
    def make_xgb():  # Factory function to create XGBoost classifier
        return XGBClassifier(
            n_estimators=1200, learning_rate=0.03, max_depth=3,  # Basic tree parameters
            min_child_weight=5, gamma=1.0, subsample=0.9, colsample_bytree=0.9,  # Regularization and sampling
            reg_lambda=1.0, reg_alpha=0.5, objective="binary:logistic",  # L1/L2 regularization and objective
            eval_metric="logloss", random_state=args.seed, n_jobs=-1, tree_method="hist",  # Evaluation and training settings
        )
    def make_lgbm():  # Factory function to create LightGBM regressor
        return LGBMRegressor(
            n_estimators=1200, learning_rate=0.03, num_leaves=31, min_child_samples=20,  # Basic model parameters
            subsample=0.8, subsample_freq=1, colsample_bytree=0.9, reg_lambda=1.0, reg_alpha=0.0,  # Sampling and regularization
            random_state=args.seed, n_jobs=-1, verbose=-1  # Reproducibility, parallelization, suppress output
        )

    # OOF arrays and test blending accumulators
    oof_xgb = np.zeros(len(X_tr), dtype=float)  # Out-of-fold predictions for XGBoost
    oof_lgb = np.zeros(len(X_tr), dtype=float)  # Out-of-fold predictions for LightGBM
    te_pred_xgb = np.zeros(len(X_te), dtype=float)  # Test predictions accumulator for XGBoost
    te_pred_lgb = np.zeros(len(X_te), dtype=float)  # Test predictions accumulator for LightGBM

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)  # Create stratified K-fold splitter
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_tr, y_tr), 1):  # Iterate through folds
        X_tr_f, X_va_f = X_tr.iloc[tr_idx], X_tr.iloc[va_idx]  # Split features for this fold
        y_tr_f, y_va_f = y_tr[tr_idx], y_tr[va_idx]  # Split labels for this fold
        w_tr_f, w_va_f = w[tr_idx], w[va_idx]  # Split weights for this fold

        # FE + impute
        X_tr_f = fe.transform(X_tr_f)  # Apply feature engineering to fold training data
        X_va_f = fe.transform(X_va_f)  # Apply feature engineering to fold validation data
        X_te_f = fe.transform(X_te)  # Apply feature engineering to test data

        imp = SimpleImputer(strategy="most_frequent").fit(X_tr_f)  # Fit imputer on fold training data
        X_tr_imp = imp.transform(X_tr_f)  # Impute fold training data
        X_va_imp = imp.transform(X_va_f)  # Impute fold validation data
        X_te_imp = imp.transform(X_te_f)  # Impute test data

        # XGB
        xgb = make_xgb()  # Create XGBoost classifier
        xgb.fit(X_tr_imp, y_tr_f, sample_weight=w_tr_f, verbose=False)  # Train XGBoost on fold data
        oof_xgb[va_idx] = xgb.predict_proba(X_va_imp)[:,1]  # Store out-of-fold predictions
        te_pred_xgb += xgb.predict_proba(X_te_imp)[:,1] / args.folds  # Accumulate test predictions

        # LGBM (regressor; clip to [0,1]) - Remove verbose parameter
        lgb = make_lgbm()  # Create LightGBM regressor
        lgb.fit(X_tr_imp, y_tr_f, sample_weight=w_tr_f)  # Train LightGBM on fold data
        oof_lgb[va_idx] = np.clip(lgb.predict(X_va_imp), 0.0, 1.0)  # Store clipped out-of-fold predictions
        te_pred_lgb += np.clip(lgb.predict(X_te_imp), 0.0, 1.0) / args.folds  # Accumulate clipped test predictions

        print(f"[stack] fold {fold}/{args.folds} done")  # Progress indicator

    # Meta-learner on OOF
    Z_tr = np.column_stack([oof_xgb, oof_lgb])  # Stack out-of-fold predictions as meta-features
    meta = LogisticRegression(solver="liblinear", class_weight="balanced", random_state=args.seed)  # Create meta-learner
    meta.fit(Z_tr, y_tr, sample_weight=w)  # Train meta-learner on stacked predictions

    # Tune threshold for accuracy on OOF meta probs
    p_tr_meta = meta.predict_proba(Z_tr)[:,1]  # Get meta-learner probabilities on training data
    thr, score = tune_threshold(y_tr, p_tr_meta, objective=args.tune_metric)  # Find optimal threshold

    # Evaluate on test via averaged base predictions
    Z_te = np.column_stack([te_pred_xgb, te_pred_lgb])  # Stack averaged test predictions
    p_te_meta = meta.predict_proba(Z_te)[:,1]  # Get meta-learner probabilities on test data
    rep = metrics(y_te, p_te_meta, thr)  # Calculate comprehensive metrics

    # Save artifacts
    outdir = Path(args.out)  # Ensure output directory is Path object
    joblib.dump({"meta": meta}, outdir / "stack_meta.joblib")  # Save meta-learner
    with open(outdir / "stack_params.json", "w") as f:  # Save stacking parameters
        json.dump({"threshold": float(thr), "tune_metric": args.tune_metric,
                   "oof_score": float(score)}, f, indent=2)

    # Save predictions + metrics
    pd.DataFrame({  # Create predictions dataframe
        "y_true": y_te,  # True test labels
        "p_xgb": te_pred_xgb,  # XGBoost test probabilities
        "p_lgbm": te_pred_lgb,  # LightGBM test probabilities
        "p_meta": p_te_meta,  # Meta-learner test probabilities
        "y_pred": (p_te_meta >= thr).astype(int),  # Final binary predictions
    }).to_csv(outdir / "stack_test_predictions.csv", index=False)  # Save to CSV

    with open(outdir / "metrics_stack.json", "w") as f:  # Save metrics to JSON
        json.dump(rep, f, indent=2)

    print("=== Stacked (Logistic on OOF XGB+LGBM) ===")  # Print results header
    print(json.dumps(rep, indent=2))  # Print formatted metrics
    print(f"Artifacts saved to: {outdir}")  # Print save location

if __name__ == "__main__":  # Script entry point
    main()  # Run main function
