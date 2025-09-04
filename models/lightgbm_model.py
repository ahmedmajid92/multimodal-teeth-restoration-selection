# models/lightgbm_model.py
from __future__ import annotations  # Enable postponed evaluation of annotations for type hints
import argparse  # Command line argument parsing
from pathlib import Path  # Object-oriented filesystem path handling
from typing import Tuple, Dict, Any  # Type hints for function signatures
import numpy as np  # Numerical computing library
import pandas as pd  # Data manipulation and analysis library
from sklearn.impute import SimpleImputer  # Fill missing values with simple strategies
from sklearn.model_selection import train_test_split  # Split data into train/validation sets
from lightgbm import LGBMRegressor, early_stopping  # LightGBM regressor and early stopping callback
import joblib  # Model serialization and persistence

# Define the feature columns used for model training
FEATURES = ["depth","width","enamel_cracks","occlusal_load","carious_lesion",
            "opposing_type","adjacent_teeth","age_range","cervical_lesion"]

LABEL_SOFT = "p_indirect"     # Soft probability label column name (continuous 0-1)
LABEL_HARD = "y_majority"     # Hard binary label column name (0 or 1) - optional for reporting
SAMPLE_WEIGHT = "weight"  # Sample weight column for weighted training

class DomainFeatures:
    def __init__(self, base_cols: list[str]):  # Initialize with base column names
        self.base_cols = base_cols  # Store base columns for reference
    def fit(self, X, y=None): return self  # Fit method for sklearn compatibility
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # Transform input data by adding interaction features
        df = X.copy()  # Create copy to avoid modifying original data
        # Create meaningful interaction features based on dental domain knowledge
        df["deep_and_thin"] = ((df["depth"]==1) & (df["width"]==0)).astype(int)  # Deep cavity with thin remaining walls
        df["deep_or_cracks"] = ((df["depth"]==1) | (df["enamel_cracks"]==1)).astype(int)  # Deep cavity or presence of cracks
        df["load_implant"]   = ((df["occlusal_load"]==1) & (df["opposing_type"]==3)).astype(int)  # High occlusal load against implant
        df["risk_plus_cervical"] = ((df["carious_lesion"]==1) & (df["cervical_lesion"]==1)).astype(int)  # High caries risk with cervical lesion
        df["stable_wall"] = ((df["width"]==1) & (df["enamel_cracks"]==0) & (df["occlusal_load"]==0)).astype(int)  # Stable remaining tooth structure
        df["depth_x_load"] = (df["depth"]*df["occlusal_load"]).astype(int)  # Interaction between cavity depth and occlusal load
        df["depth_x_risk"] = (df["depth"]*df["carious_lesion"]).astype(int)  # Interaction between depth and caries risk level
        return df  # Return dataframe with added features

class LGBMProbWrapper:
    def __init__(self, fe: DomainFeatures, imputer: SimpleImputer, model: LGBMRegressor):
        self.fe = fe  # Feature engineering transformer
        self.imputer = imputer  # Missing value imputer
        self.model = model  # Trained LightGBM regressor
    def _prep(self, X: pd.DataFrame):  # Prepare data for prediction
        return self.imputer.transform(self.fe.transform(X))  # Apply feature engineering then imputation
    def predict(self, X: pd.DataFrame) -> np.ndarray:  # Get probability predictions
        p = self.model.predict(self._prep(X))  # Get raw model predictions
        return np.clip(p, 0.0, 1.0)  # Clip predictions to valid probability range [0,1]
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:  # Get prediction probabilities in sklearn format
        p = self.predict(X)  # Get clipped probabilities
        return np.column_stack([1.0 - p, p])  # Return as [prob_class_0, prob_class_1] array

def load_dataset(data_path: Path) -> pd.DataFrame:
    df = pd.read_csv(data_path)  # Load CSV data from specified path
    # Check for required columns and raise error if any are missing
    missing = [c for c in FEATURES + [LABEL_SOFT, SAMPLE_WEIGHT] if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")  # Report missing columns
    return df  # Return loaded dataframe

def train_lgbm(
    data_path: Path,  # Path to training data
    output_dir: Path,  # Directory to save model outputs
    random_state: int = 42,  # Random seed for reproducibility
    test_size_val: float = 0.20,  # Fraction of data for validation
    consensus_power: float = 0.5,    # Power to raise consensus weights to (your best performing value)
    min_weight: float = 0.0,         # Minimum weight threshold for including training rows
    use_domain_features: bool = True,  # Whether to use engineered domain features
) -> Tuple[LGBMProbWrapper, Dict[str, Any]]:
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)  # Create output directory
    df = load_dataset(Path(data_path))  # Load the dataset
    assert "split" in df.columns, "Expected a 'split' column."  # Ensure split column exists
    df_train = df[df["split"].astype(str).str.lower() == "train"].copy()  # Filter to training data only

    if min_weight > 0:  # If minimum weight threshold is set
        before = len(df_train)  # Count rows before filtering
        df_train = df_train[df_train[SAMPLE_WEIGHT].fillna(0) >= float(min_weight)].copy()  # Filter low weight rows
        print(f"[info] dropped {before - len(df_train)} low-consensus rows (weight < {min_weight}) from TRAIN only")

    X = df_train[FEATURES].copy()  # Extract feature columns
    y_soft = df_train[LABEL_SOFT].astype(float).values  # Extract soft probability targets
    y_soft = np.clip(y_soft, 1e-3, 1 - 1e-3)  # Clip to avoid exact 0/1 values for numerical stability

    w = df_train[SAMPLE_WEIGHT].fillna(1.0).astype(float).values  # Extract sample weights, fill missing with 1.0
    w = np.power(w, consensus_power)  # Apply power transformation to soften weight differences
    w = w / (np.mean(w) if np.mean(w) > 0 else 1.0)  # Normalize weights to have mean=1

    X_tr, X_val, y_tr, y_val, w_tr, w_val = train_test_split(  # Split data into train and validation sets
        X, y_soft, w, test_size=test_size_val, random_state=random_state  # Use continuous target for regression
    )

    fe = DomainFeatures(FEATURES) if use_domain_features else DomainFeatures([])  # Create feature engineering transformer
    X_tr_fe = fe.transform(X_tr); X_val_fe = fe.transform(X_val)  # Apply feature engineering to both sets

    imputer = SimpleImputer(strategy="most_frequent")  # Create imputer using most frequent value strategy
    imputer.fit(X_tr_fe)  # Fit imputer on training data
    X_tr_imp = imputer.transform(X_tr_fe)  # Impute missing values in training data
    X_val_imp = imputer.transform(X_val_fe)  # Impute missing values in validation data

    model = LGBMRegressor(  # Create LightGBM regressor with optimized hyperparameters
        n_estimators=1200, learning_rate=0.03, num_leaves=31,  # Basic model parameters
        min_child_samples=20, subsample=0.8, subsample_freq=1,  # Regularization parameters
        colsample_bytree=0.9, reg_lambda=1.0, reg_alpha=0.0,  # Feature sampling and regularization
        random_state=random_state, n_jobs=-1,  # Reproducibility and parallelization
    )
    model.fit(  # Train the model
        X_tr_imp, y_tr,  # Training features and continuous targets
        sample_weight=w_tr,  # Sample weights for training
        eval_set=[(X_val_imp, y_val)],  # Validation set for monitoring
        eval_sample_weight=[w_val],  # Validation sample weights
        eval_metric="l2",  # Use L2 loss for regression evaluation
        callbacks=[early_stopping(stopping_rounds=100, verbose=False)],  # Early stopping to prevent overfitting
    )

    bundle = LGBMProbWrapper(fe, imputer, model)  # Bundle preprocessing and model into wrapper

    model_path = Path(output_dir) / "lgbm_regressor_pipeline.joblib"  # Define path for saving model
    joblib.dump(bundle, model_path)  # Save the complete pipeline

    info = {  # Create info dictionary with training details
        "model_path": str(model_path),
        "n_train_rows": int(len(df_train)),
        "used_split": True,
        "consensus_power": consensus_power,
        "min_weight": min_weight,
        "domain_features": bool(use_domain_features),
    }
    return bundle, info  # Return trained model wrapper and training info

def _default_paths():
    here = Path(__file__).resolve(); root = here.parents[1]  # Get project root directory
    return root / "data" / "excel" / "data_processed.csv", root / "models" / "outputs"  # Return default data and output paths

if __name__ == "__main__":  # Script entry point when run directly
    parser = argparse.ArgumentParser(description="Train LightGBM regressor on p_indirect with domain features.")  # Create argument parser
    dflt_data, dflt_out = _default_paths()  # Get default paths
    parser.add_argument("--data", type=Path, default=dflt_data)  # Data file path argument
    parser.add_argument("--out", type=Path, default=dflt_out)  # Output directory argument
    parser.add_argument("--seed", type=int, default=42)  # Random seed argument
    parser.add_argument("--consensus-power", type=float, default=0.5)  # Consensus weight power argument
    parser.add_argument("--min-weight", type=float, default=0.0)  # Minimum sample weight argument
    parser.add_argument("--no-domain-features", action="store_true")  # Flag to disable domain features
    args = parser.parse_args()  # Parse command line arguments

    bundle, info = train_lgbm(  # Train the model with parsed arguments
        args.data, args.out, random_state=args.seed,
        consensus_power=args.consensus_power, min_weight=args.min_weight,
        use_domain_features=not args.no_domain_features,  # Invert the "no" flag
    )
    print("✅ LightGBM trained.")  # Success message
    print(f"   Train rows:        {info['n_train_rows']}")  # Print training statistics
    print(f"   Domain features:   {info['domain_features']}")
    print(f"   Min weight:        {info['min_weight']}")
    print(f"   Model saved to:    {info['model_path']}")
