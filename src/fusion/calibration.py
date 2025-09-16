# src/fusion/calibration.py
from dataclasses import dataclass
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

@dataclass
class Calibrator:
    kind: str = "isotonic"  # "isotonic" or "platt"
    model: object = None

    def fit(self, p: np.ndarray, y: np.ndarray):
        p = np.asarray(p).reshape(-1)
        y = np.asarray(y).astype(int).reshape(-1)
        if self.kind == "isotonic":
            self.model = IsotonicRegression(out_of_bounds="clip")
            self.model.fit(p, y)
        elif self.kind == "platt":
            # simple Platt scaling on probabilities (or logits if provided)
            self.model = LogisticRegression(solver="lbfgs", max_iter=1000)
            self.model.fit(p.reshape(-1,1), y)
        else:
            raise ValueError("Unknown calibrator")
        return self

    def transform(self, p: np.ndarray):
        p = np.asarray(p).reshape(-1)
        if self.model is None:
            return p
        if self.kind == "isotonic":
            return self.model.predict(p)
        return self.model.predict_proba(p.reshape(-1,1))[:,1]
