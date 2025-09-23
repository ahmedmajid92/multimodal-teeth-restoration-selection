# ui/gradio_app/tab_model.py
import json
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd


def _coerce_numeric(x) -> float:
    """Safely coerce text/number like '2', '2.0' to float; NaN on failure."""
    try:
        return float(str(x).strip())
    except Exception:
        return np.nan


def _align_X_to_model(X: pd.DataFrame, model) -> pd.DataFrame:
    """
    Ensure X has exactly the columns (order + presence) the model was trained on.
    Missing columns are created and filled with 0; extra columns are dropped.
    """
    if hasattr(model, "feature_names_in_"):
        cols = list(model.feature_names_in_)
        X = X.reindex(columns=cols, fill_value=0)
    return X


class TabEnsemble:
    """
    Loads a small GradientBoosting (or similar) ensemble trained on tabular features.
    Key improvement: robust feature alignment to the model's training
    one-hot columns using model.feature_names_in_ (sklearn >=1.0).
    """

    # Base feature lists (raw)
    NUM = ["depth", "width"]
    CAT = [
        "enamel_cracks",
        "occlusal_load",
        "carious_lesion",
        "opposing_type",
        "adjacent_teeth",
        "age_range",
        "cervical_lesion",
    ]

    def __init__(self, models: List, meta: Optional[dict] = None):
        self.models = models
        self.num_folds = len(models)
        self.meta = meta or {}

        # Gather required columns from all models (union) to be safe
        req_cols = set()
        for m in self.models:
            cols = getattr(m, "feature_names_in_", None)
            if cols is not None:
                req_cols.update(list(cols))
        self.required_columns = sorted(list(req_cols)) if req_cols else None

    @classmethod
    def from_folder(cls, folder: Path) -> Optional["TabEnsemble"]:
        folder = Path(folder)
        if not folder.exists():
            return None

        models = []
        meta = None
        for p in sorted(folder.glob("tab_fold*.pkl")):
            try:
                models.append(joblib.load(p))
            except Exception:
                pass
        if not models:
            return None

        meta_path = folder / "tab_meta.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                meta = None

        return cls(models=models, meta=meta)

    # -----------------------------
    # Public: predict one sample
    # -----------------------------
    def predict_one(self, features: Dict[str, object]) -> float:
        """
        features: dict with keys in NUM + CAT. Values may be strings like '2.0'.
        Returns calibrated probability of "Indirect".
        """
        # 1) Build single-row DataFrame
        raw = {}
        # numeric
        for k in self.NUM:
            raw[k] = _coerce_numeric(features.get(k, None))
        # categorical — keep as strings so one-hot names match training like "adjacent_teeth_-1"
        for k in self.CAT:
            v = features.get(k, "")
            raw[k] = "" if v is None else str(v).strip()

        df = pd.DataFrame([raw])

        # 2) One-hot encode categoricals
        df_dum = pd.get_dummies(df[self.NUM + self.CAT], columns=self.CAT, dummy_na=False)

        # 3) (Optional) align to the union of training columns for a stable base
        X_base = self._align_to_required(df_dum)

        # 4) Average probs across folds, aligning to EACH model's feature_names_in_
        probs = []
        for m in self.models:
            Xm = _align_X_to_model(X_base.copy(), m)  # <- per-model alignment (fixes name mismatch)
            # Ensure numeric dtype
            for c in Xm.columns:
                Xm[c] = pd.to_numeric(Xm[c], errors="coerce").fillna(0.0)
            p = m.predict_proba(Xm)[:, 1][0]
            probs.append(float(p))
        return float(np.mean(probs))

    # -----------------------------
    # Helpers
    # -----------------------------
    def _align_to_required(self, df_dum: pd.DataFrame) -> pd.DataFrame:
        """
        Make df_dum match the training-time columns (union across folds):
          - add any missing required columns (zeros)
          - drop any unexpected columns
          - order columns exactly like training (union order)
        If feature_names_in_ is not available (older sklearn),
        we fall back to current df_dum columns (best effort).
        """
        if not self.required_columns:
            # Best-effort; at least ensure numeric are present
            return df_dum

        X = df_dum.copy()

        # Add missing columns with zeros
        for col in self.required_columns:
            if col not in X.columns:
                X[col] = 0.0

        # Drop extras
        extras = [c for c in X.columns if c not in self.required_columns]
        if extras:
            X = X.drop(columns=extras)

        # Reorder
        X = X[self.required_columns]

        # Ensure numeric dtype
        for c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0.0)

        return X
