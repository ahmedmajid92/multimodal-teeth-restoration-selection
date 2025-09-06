# models/vision/utils.py
import math, os, time, random
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score

class AvgMeter:
    def __init__(self): self.sum = 0.0; self.n = 0
    def update(self, val, k=1): self.sum += val * k; self.n += k
    @property
    def avg(self): return self.sum / max(1, self.n)

def seed_all(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed); torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def binary_metrics_from_logits_2class(logits, targets):
    # logits: [N,2], targets: [N]  0=Direct,1=Indirect
    probs = torch.softmax(logits, dim=1)[:,1].detach().cpu().numpy()
    preds = (probs >= 0.5).astype(int)
    t = targets.detach().cpu().numpy()
    acc = accuracy_score(t, preds)
    f1  = f1_score(t, preds)
    prec = precision_score(t, preds, zero_division=0)
    rec  = recall_score(t, preds, zero_division=0)
    try:
        auc = roc_auc_score(t, probs)
    except Exception:
        auc = float("nan")
    return dict(acc=acc, f1=f1, prec=prec, rec=rec, auc=auc)

def binary_metrics_from_logits_single(logits, targets):
    # logits: [N,1], targets: [N,1] prob of INDIRECT
    probs = torch.sigmoid(logits).detach().cpu().numpy().ravel()
    y = targets.detach().cpu().numpy().ravel()
    # For reporting accuracy, threshold at 0.5 to get 0/1
    preds = (probs >= 0.5).astype(int)
    hard = (y >= 0.5).astype(int)
    acc = accuracy_score(hard, preds)
    try:
        auc = roc_auc_score(hard, probs)
    except Exception:
        auc = float("nan")
    brier = np.mean((probs - y) ** 2)
    mae = np.mean(np.abs(probs - y))
    return dict(acc=acc, auc=auc, brier=brier, mae=mae)

def save_checkpoint(state, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
