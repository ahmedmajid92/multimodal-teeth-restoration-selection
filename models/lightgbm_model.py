# models/lightgbm_model.py
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor, early_stopping
import joblib

FEATURES = ["depth","width","enamel_cracks","occlusal_load","carious_lesion",
            "opposing_type","adjacent_teeth","age_range","cervical_lesion"]

LABEL_SOFT = "p_indirect"     # regression target
LABEL_HARD = "y_majority"     # optional for reporting
SAMPLE_WEIGHT = "weight"

class DomainFeatures:
    def __init__(self, base_cols: list[str]):
        self.base_cols = base_cols
    def fit(self, X, y=None): return self
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        df["deep_and_thin"] = ((df["depth"]==1) & (df["width"]==0)).astype(int)
        df["deep_or_cracks"] = ((df["depth"]==1) | (df["enamel_cracks"]==1)).astype(int)
        df["load_implant"]   = ((df["occlusal_load"]==1) & (df["opposing_type"]==3)).astype(int)
        df["risk_plus_cervical"] = ((df["carious_lesion"]==1) & (df["cervical_lesion"]==1)).astype(int)
        df["stable_wall"] = ((df["width"]==1) & (df["enamel_cracks"]==0) & (df["occlusal_load"]==0)).astype(int)
        df["depth_x_load"] = (df["depth"]*df["occlusal_load"]).astype(int)
        df["depth_x_risk"] = (df["depth"]*df["carious_lesion"]).astype(int)
        return df

class LGBMProbWrapper:
    def __init__(self, fe: DomainFeatures, imputer: SimpleImputer, model: LGBMRegressor):
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

def load_dataset(data_path: Path) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    missing = [c for c in FEATURES + [LABEL_SOFT, SAMPLE_WEIGHT] if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")
    return df

def train_lgbm(
    data_path: Path,
    output_dir: Path,
    random_state: int = 42,
    test_size_val: float = 0.20,
    consensus_power: float = 0.5,    # your best
    min_weight: float = 0.0,         # drop low-consensus train rows
    use_domain_features: bool = True,
) -> Tuple[LGBMProbWrapper, Dict[str, Any]]:
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    df = load_dataset(Path(data_path))
    assert "split" in df.columns, "Expected a 'split' column."
    df_train = df[df["split"].astype(str).str.lower() == "train"].copy()

    if min_weight > 0:
        before = len(df_train)
        df_train = df_train[df_train[SAMPLE_WEIGHT].fillna(0) >= float(min_weight)].copy()
        print(f"[info] dropped {before - len(df_train)} low-consensus rows (weight < {min_weight}) from TRAIN only")

    X = df_train[FEATURES].copy()
    y_soft = df_train[LABEL_SOFT].astype(float).values
    y_soft = np.clip(y_soft, 1e-3, 1 - 1e-3)

    w = df_train[SAMPLE_WEIGHT].fillna(1.0).astype(float).values
    w = np.power(w, consensus_power)
    w = w / (np.mean(w) if np.mean(w) > 0 else 1.0)

    X_tr, X_val, y_tr, y_val, w_tr, w_val = train_test_split(
        X, y_soft, w, test_size=test_size_val, random_state=random_state
    )

    fe = DomainFeatures(FEATURES) if use_domain_features else DomainFeatures([])
    X_tr_fe = fe.transform(X_tr); X_val_fe = fe.transform(X_val)

    imputer = SimpleImputer(strategy="most_frequent")
    imputer.fit(X_tr_fe)
    X_tr_imp = imputer.transform(X_tr_fe)
    X_val_imp = imputer.transform(X_val_fe)

    model = LGBMRegressor(
        n_estimators=1200, learning_rate=0.03, num_leaves=31,
        min_child_samples=20, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.9, reg_lambda=1.0, reg_alpha=0.0,
        random_state=random_state, n_jobs=-1,
    )
    model.fit(
        X_tr_imp, y_tr,
        sample_weight=w_tr,
        eval_set=[(X_val_imp, y_val)],
        eval_sample_weight=[w_val],
        eval_metric="l2",
        callbacks=[early_stopping(stopping_rounds=100, verbose=False)],
    )

    bundle = LGBMProbWrapper(fe, imputer, model)

    model_path = Path(output_dir) / "lgbm_regressor_pipeline.joblib"
    joblib.dump(bundle, model_path)

    info = {
        "model_path": str(model_path),
        "n_train_rows": int(len(df_train)),
        "used_split": True,
        "consensus_power": consensus_power,
        "min_weight": min_weight,
        "domain_features": bool(use_domain_features),
    }
    return bundle, info

def _default_paths():
    here = Path(__file__).resolve(); root = here.parents[1]
    return root / "data" / "excel" / "data_processed.csv", root / "models" / "outputs"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LightGBM regressor on p_indirect with domain features.")
    dflt_data, dflt_out = _default_paths()
    parser.add_argument("--data", type=Path, default=dflt_data)
    parser.add_argument("--out", type=Path, default=dflt_out)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--consensus-power", type=float, default=0.5)
    parser.add_argument("--min-weight", type=float, default=0.0)
    parser.add_argument("--no-domain-features", action="store_true")
    args = parser.parse_args()

    bundle, info = train_lgbm(
        args.data, args.out, random_state=args.seed,
        consensus_power=args.consensus_power, min_weight=args.min_weight,
        use_domain_features=not args.no_domain_features,
    )
    print("✅ LightGBM trained.")
    print(f"   Train rows:        {info['n_train_rows']}")
    print(f"   Domain features:   {info['domain_features']}")
    print(f"   Min weight:        {info['min_weight']}")
    print(f"   Model saved to:    {info['model_path']}")
