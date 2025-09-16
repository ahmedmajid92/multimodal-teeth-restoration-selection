# src/fusion/metrics.py
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score, brier_score_loss,
    confusion_matrix
)

def evaluate(y_true, p, threshold=0.5):
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    p = np.clip(np.asarray(p).reshape(-1), 0, 1)
    y_hat = (p >= threshold).astype(int)

    out = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_hat)),
        "precision": float(precision_score(y_true, y_hat, zero_division=0)),
        "recall": float(recall_score(y_true, y_hat)),
        "f1": float(f1_score(y_true, y_hat)),
        "brier": float(brier_score_loss(y_true, p)),
        "roc_auc": float(roc_auc_score(y_true, p)),
        "pr_auc": float(average_precision_score(y_true, p)),
        "confusion_matrix": confusion_matrix(y_true, y_hat).tolist()
    }
    return out

def tune_threshold(y_true, p, metric="f1"):
    # search fine grid; you can switch to argmax of PR/ROC if needed
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    p = np.clip(np.asarray(p).reshape(-1), 0, 1)
    best_t, best_m = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 181):  # step ~0.005
        y_hat = (p >= t).astype(int)
        if metric == "f1":
            m = f1_score(y_true, y_hat)
        elif metric == "youden":
            from sklearn.metrics import roc_curve
            fpr, tpr, thr = roc_curve(y_true, p)
            m = np.max(tpr - fpr)
        else:
            from sklearn.metrics import average_precision_score
            m = average_precision_score(y_true, p)
        if m > best_m:
            best_m, best_t = m, t
    return float(best_t), float(best_m)
