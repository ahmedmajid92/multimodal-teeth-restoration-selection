# models/xgboost_model.py
from __future__ import annotations  # Enable postponed evaluation of annotations for type hints
import argparse, json  # Command line argument parsing and JSON handling
from pathlib import Path  # Object-oriented filesystem path handling
from typing import Tuple, Dict, Any  # Type hints for function signatures

import numpy as np  # Numerical computing library
import pandas as pd  # Data manipulation and analysis library
from sklearn.base import BaseEstimator, TransformerMixin  # Base classes for creating custom transformers
from sklearn.impute import SimpleImputer  # Fill missing values with simple strategies
from sklearn.model_selection import train_test_split  # Split data into train/validation sets
from sklearn.metrics import balanced_accuracy_score, f1_score, accuracy_score  # Model evaluation metrics
from sklearn.calibration import CalibratedClassifierCV  # Probability calibration using cross-validation
from sklearn.utils.class_weight import compute_class_weight  # Calculate balanced class weights
from xgboost import XGBClassifier  # Gradient boosting classifier
import joblib  # Model serialization and persistence

# Define the feature columns used for model training
FEATURES = ["depth","width","enamel_cracks","occlusal_load","carious_lesion",
            "opposing_type","adjacent_teeth","age_range","cervical_lesion"]

LABEL_HARD = "y_majority"  # Hard binary label column name (0 or 1)
SAMPLE_WEIGHT = "weight"  # Sample weight column for weighted training

THRESHOLD_FILE = "xgb_threshold.json"  # Filename for saving optimal threshold
MODEL_FILE = "xgb_classifier_pipeline.joblib"  # Filename for saving trained model

# Monotonicity constraints for features (1=positive, -1=negative, 0=no constraint)
# depth(+), width(-), cracks(+), load(+), lesion(+), opposing(?), adjacent(?), age(?), cervical(+)
DEFAULT_MONO = (1, -1, 1, 1, 1, 0, 0, 0, 1)

# -------- Feature engineering --------
class DomainFeatures(BaseEstimator, TransformerMixin):
    """Add a few interpretable interaction features that often help accuracy."""
    def __init__(self, base_cols: list[str]):  # Initialize with base column names
        self.base_cols = base_cols  # Store base columns for reference
    def fit(self, X, y=None):  # Fit method required by sklearn interface
        return self  # Return self for method chaining
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # Transform input data
        df = X.copy()  # Create copy to avoid modifying original data
        # boolean combos - Create binary interaction features
        df["deep_and_thin"] = ((df["depth"]==1) & (df["width"]==0)).astype(int)  # Deep cavity with thin walls
        df["deep_or_cracks"] = ((df["depth"]==1) | (df["enamel_cracks"]==1)).astype(int)  # Deep or has cracks
        df["load_implant"]   = ((df["occlusal_load"]==1) & (df["opposing_type"]==3)).astype(int)  # High load against implant
        df["risk_plus_cervical"] = ((df["carious_lesion"]==1) & (df["cervical_lesion"]==1)).astype(int)  # High risk with cervical lesion
        df["stable_wall"] = ((df["width"]==1) & (df["enamel_cracks"]==0) & (df["occlusal_load"]==0)).astype(int)  # Stable remaining walls
        # numeric interactions (carious_lesion is -1/0/1)
        df["depth_x_load"] = (df["depth"]*df["occlusal_load"]).astype(int)  # Interaction between depth and load
        df["depth_x_risk"] = (df["depth"]*df["carious_lesion"]).astype(int)  # Interaction between depth and caries risk
        return df  # Return transformed dataframe

class DataFrameImputer(BaseEstimator, TransformerMixin):
    def __init__(self, strategy="most_frequent"):  # Initialize imputer with strategy
        self.imputer = SimpleImputer(strategy=strategy)  # Create sklearn imputer
        self.feature_names_ = None  # Store feature names for output
    def fit(self, X, y=None):  # Fit imputer to training data
        self.feature_names_ = X.columns.tolist() if isinstance(X, pd.DataFrame) else None  # Save column names
        self.imputer.fit(X); return self  # Fit imputer and return self
    def transform(self, X):  # Transform data by filling missing values
        X_imp = self.imputer.transform(X)  # Apply imputation
        # Reconstruct column names for output DataFrame
        cols = self.feature_names_ or (X.columns.tolist() if isinstance(X, pd.DataFrame) else [f"f{i}" for i in range(X.shape[1])])
        # Return DataFrame with same structure as input
        return pd.DataFrame(X_imp, columns=cols, index=(X.index if isinstance(X, pd.DataFrame) else None))

class ImputerThenModel:
    def __init__(self, fe: DomainFeatures, imputer: DataFrameImputer, calibrated_model):
        self.fe = fe  # Feature engineering transformer
        self.imputer = imputer  # Missing value imputer
        self.calibrated_model = calibrated_model  # Calibrated model for predictions
    def _prep(self, X: pd.DataFrame) -> pd.DataFrame:  # Prepare data for prediction
        return self.imputer.transform(self.fe.transform(X))  # Apply feature engineering then imputation
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:  # Get prediction probabilities
        return self.calibrated_model.predict_proba(self._prep(X))  # Predict on prepared data
    def predict(self, X: pd.DataFrame) -> np.ndarray:  # Get binary predictions
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)  # Threshold probabilities at 0.5

def load_dataset(data_path: Path) -> pd.DataFrame:
    df = pd.read_csv(data_path)  # Load CSV data
    # Check for required columns and raise error if any are missing
    missing = [c for c in FEATURES + [LABEL_HARD, SAMPLE_WEIGHT] if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")  # Report missing columns
    return df  # Return loaded dataframe

def _find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray, metric="balanced_accuracy") -> Tuple[float, float]:
    grid = np.linspace(0.05, 0.95, 181)  # Create threshold grid from 0.05 to 0.95
    best_t, best_m = 0.5, -1.0  # Initialize best threshold and metric
    for t in grid:  # Iterate through threshold values
        y_pred = (y_prob >= t).astype(int)  # Convert probabilities to binary predictions
        if metric == "balanced_accuracy":  # Calculate balanced accuracy
            m = balanced_accuracy_score(y_true, y_pred)
        elif metric == "f1":  # Calculate F1 score
            m = f1_score(y_true, y_pred, zero_division=0)
        elif metric == "accuracy":  # Calculate accuracy
            m = accuracy_score(y_true, y_pred)
        else:
            raise ValueError("metric must be one of: balanced_accuracy, f1, accuracy")  # Invalid metric error
        if m > best_m:  # Update best if current metric is better
            best_m, best_t = m, t
    return float(best_t), float(best_m)  # Return best threshold and metric value

def train_xgb(
    data_path: Path,  # Path to training data
    output_dir: Path,  # Directory to save model outputs
    random_state: int = 42,  # Random seed for reproducibility
    test_size_val: float = 0.20,  # Fraction of data for validation
    calibration: str = "sigmoid",  # Calibration method
    tune_metric: str = "accuracy",  # Metric to optimize threshold for
    consensus_power: float = 0.7,  # Power to raise consensus weights to
    use_monotone: bool = False,  # Whether to use monotonicity constraints
    min_weight: float = 0.0,  # Minimum weight threshold for training samples
    use_domain_features: bool = True,  # Whether to use engineered features
) -> Tuple[ImputerThenModel, Dict[str, Any]]:
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)  # Create output directory
    df = load_dataset(Path(data_path))  # Load the dataset
    assert "split" in df.columns, "Expected a 'split' column."  # Ensure split column exists
    df_train = df[df["split"].astype(str).str.lower() == "train"].copy()  # Filter to training data only

    # (1) optional: drop highly ambiguous rows from training only
    if min_weight > 0:  # If minimum weight threshold is set
        before = len(df_train)  # Count rows before filtering
        df_train = df_train[df_train[SAMPLE_WEIGHT].fillna(0) >= float(min_weight)].copy()  # Filter low weight rows
        after = len(df_train)  # Count rows after filtering
        print(f"[info] dropped {before - after} low-consensus rows (weight < {min_weight}) from TRAIN only")

    X = df_train[FEATURES].copy()  # Extract feature columns
    y = df_train[LABEL_HARD].astype(int).values  # Extract target labels as integers
    w_consensus = df_train[SAMPLE_WEIGHT].fillna(1.0).astype(float).values  # Extract sample weights

    # (2) weights: soften + class-balance + normalize
    w_consensus = np.power(w_consensus, consensus_power)  # Apply power transformation to soften weights
    cls_w = compute_class_weight(class_weight="balanced", classes=np.array([0,1]), y=y)  # Calculate balanced class weights
    w_final = w_consensus * np.where(y == 1, cls_w[1], cls_w[0])  # Apply class balancing to sample weights
    w_final = w_final / (np.mean(w_final) if np.mean(w_final) > 0 else 1.0)  # Normalize weights to mean=1

    # (3) split
    X_tr, X_val, y_tr, y_val, w_tr, w_val = train_test_split(
        X, y, w_final, test_size=test_size_val, random_state=random_state, stratify=y  # Split maintaining class proportions
    )

    # (4) domain features + impute
    fe = DomainFeatures(FEATURES) if use_domain_features else DomainFeatures([])  # Create feature engineering transformer
    X_tr_fe = fe.transform(X_tr)  # Apply feature engineering to training data
    X_val_fe = fe.transform(X_val)  # Apply feature engineering to validation data

    imputer = DataFrameImputer(strategy="most_frequent").fit(X_tr_fe)  # Fit imputer on training data
    X_tr_imp = imputer.transform(X_tr_fe)  # Impute training data
    X_val_imp = imputer.transform(X_val_fe)  # Impute validation data

    # (5) model
    xgb_kwargs = dict(  # XGBoost hyperparameters
        n_estimators=1200, learning_rate=0.03, max_depth=3,  # Basic tree parameters
        min_child_weight=5, gamma=1.0,  # Regularization parameters
        subsample=0.9, colsample_bytree=0.9,  # Sampling parameters
        reg_lambda=1.0, reg_alpha=0.5,  # L1/L2 regularization
        objective="binary:logistic",  # Binary classification objective
        eval_metric=["logloss", "auc"],  # Evaluation metrics
        random_state=random_state, n_jobs=-1,  # Reproducibility and parallelization
        tree_method="hist", early_stopping_rounds=120,  # Training optimizations
    )
    if use_monotone:  # Add monotonicity constraints if enabled
        xgb_kwargs["monotone_constraints"] = f"({','.join(map(str, DEFAULT_MONO))})"

    xgb = XGBClassifier(**xgb_kwargs)  # Create XGBoost classifier
    xgb.fit(  # Train the model
        X_tr_imp, y_tr,  # Training features and labels
        sample_weight=w_tr,  # Sample weights for training
        eval_set=[(X_val_imp, y_val)],  # Validation set for early stopping
        sample_weight_eval_set=[w_val],  # Validation sample weights
        verbose=False,  # Suppress training output
    )

    # (6) calibration + bundle
    cal = CalibratedClassifierCV(estimator=xgb, method=calibration, cv="prefit")  # Create calibrated classifier
    cal.fit(X_val_imp, y_val, sample_weight=w_val)  # Fit calibration on validation set
    bundle = ImputerThenModel(fe, imputer, cal)  # Bundle preprocessing and model

    # (7) tune threshold for YOUR objective (accuracy by default)
    val_prob = bundle.predict_proba(X_val)[:, 1]  # Get validation probabilities
    best_thr, best_metric = _find_best_threshold(y_val, val_prob, metric=tune_metric)  # Find optimal threshold

    # (8) save
    model_path = output_dir / MODEL_FILE  # Path for saving model
    joblib.dump(bundle, model_path)  # Save the complete pipeline
    thr_path = output_dir / THRESHOLD_FILE  # Path for saving threshold
    with open(thr_path, "w") as f:  # Save threshold and metadata
        json.dump({"threshold": best_thr, "metric": tune_metric, "metric_val": best_metric}, f, indent=2)

    info = {  # Create info dictionary with training details
        "model_path": str(model_path),
        "threshold_path": str(thr_path),
        "best_threshold": best_thr,
        "val_metric": best_metric,
        "val_metric_name": tune_metric,
        "n_train_rows": int(len(df_train)),
        "used_split": True,
    }
    return bundle, info  # Return trained model and info

def _default_paths():
    here = Path(__file__).resolve(); root = here.parents[1]  # Get project root directory
    return root / "data" / "excel" / "data_processed.csv", root / "models" / "outputs"  # Return default paths

if __name__ == "__main__":  # Script entry point
    parser = argparse.ArgumentParser(description="Train enhanced XGBoost on tabular features (accuracy-focused).")  # Create argument parser
    dflt_data, dflt_out = _default_paths()  # Get default paths
    parser.add_argument("--data", type=Path, default=dflt_data)  # Data file path argument
    parser.add_argument("--out", type=Path, default=dflt_out)  # Output directory argument
    parser.add_argument("--seed", type=int, default=42)  # Random seed argument
    parser.add_argument("--calibration", choices=["sigmoid","isotonic"], default="sigmoid")  # Calibration method
    parser.add_argument("--tune-metric", choices=["accuracy","balanced_accuracy","f1"], default="accuracy")  # Tuning metric
    parser.add_argument("--consensus-power", type=float, default=0.7)  # Consensus weight power
    parser.add_argument("--no-monotone", action="store_true")  # Disable monotonicity constraints
    parser.add_argument("--min-weight", type=float, default=0.0)  # Minimum sample weight
    parser.add_argument("--no-domain-features", action="store_true")  # Disable domain features
    args = parser.parse_args()  # Parse command line arguments

    bundle, info = train_xgb(  # Train the model with parsed arguments
        args.data, args.out, random_state=args.seed,
        calibration=args.calibration, tune_metric=args.tune_metric,
        consensus_power=args.consensus_power, use_monotone=not args.no_monotone,
        min_weight=args.min_weight, use_domain_features=not args.no_domain_features,
    )
    print("✅ XGBoost trained + calibrated.")  # Success message
    print(f"   Train rows:          {info['n_train_rows']}")  # Print training statistics
    print(f"   Model saved to:      {info['model_path']}")
    print(f"   Threshold saved to:  {info['threshold_path']}")
    print(f"   Best threshold (val): {info['best_threshold']:.3f} ({info['val_metric_name']}={info['val_metric']:.3f})")
