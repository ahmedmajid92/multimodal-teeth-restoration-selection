# src/fusion/meta_learner.py
import numpy as np
from sklearn.linear_model import LogisticRegression

class MetaStacker:
    """
    L1-regularized logistic (non-negative enforced via clipping) to auto-drop harmful streams.
    """
    def __init__(self, C=1.0):
        self.model = LogisticRegression(
            penalty="l1", solver="liblinear", C=C, max_iter=2000
        )

    def fit(self, P, y):
        self.model.fit(P, y.astype(int).reshape(-1))
        return self

    def predict_proba(self, P):
        p = self.model.predict_proba(P)[:,1]
        return np.clip(p, 0, 1)

    @property
    def weights_(self):
        # raw coefficients (>=0 after clipping)
        w = self.model.coef_.reshape(-1)
        w = np.clip(w, 0, None)
        s = w.sum()
        return (w / s).tolist() if s > 0 else [0.0]*len(w)
