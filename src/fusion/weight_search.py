# src/fusion/weight_search.py
import numpy as np
from itertools import product
from .metrics import evaluate, tune_threshold

def grid_simplex(num_models, step=0.05):
    # non-negative weights that sum to 1
    vals = np.arange(0, 1+1e-9, step)
    for w in product(vals, repeat=num_models):
        if abs(sum(w) - 1.0) < 1e-9:
            yield np.array(w, dtype=float)

def blended_prob(weights, P):  # P: shape [n_samples, n_models]
    w = np.asarray(weights).reshape(1,-1)
    return np.clip((P * w).sum(axis=1), 0, 1)

def search_weights(P_val, y_val, metric="f1", step=0.1, threshold_mode="tune"):
    best = {"weights": None, "threshold": 0.5, "score": -1.0}
    m = P_val.shape[1]
    for w in grid_simplex(m, step=step):
        p = blended_prob(w, P_val)
        t, _ = (0.5, None) if threshold_mode=="fixed" else tune_threshold(y_val, p, "f1")
        score = evaluate(y_val, p, t)["f1"] if metric=="f1" else evaluate(y_val, p, t)["pr_auc"]
        if score > best["score"]:
            best.update({"weights": w.tolist(), "threshold": float(t), "score": float(score)})
    return best
