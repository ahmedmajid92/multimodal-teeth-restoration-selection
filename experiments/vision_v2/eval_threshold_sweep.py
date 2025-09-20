# experiments/vision_v2/eval_threshold_sweep.py
import argparse, os, re, glob, json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

try:
    import timm
except Exception as e:
    raise RuntimeError("This script needs 'timm'. pip install timm") from e

try:
    from sklearn.metrics import roc_auc_score, roc_curve
except Exception:
    roc_auc_score = None
    roc_curve = None


# ---------------- Utilities ----------------
def set_seed(seed: int = 42):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_fold_id_from_name(path: str, default: int = -1):
    m = re.search(r"fold(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else default

def mean_std(arr):
    arr = np.array(arr, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=0))


# ---------------- Data ----------------
class ImgDS(Dataset):
    def __init__(self, df, root, img_size=384, mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)):
        self.df = df.reset_index(drop=True).copy()
        self.root = Path(root)
        self.img_size = img_size
        from torchvision import transforms as T
        self.tf = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
        self.name_col = "image_path" if "image_path" in self.df.columns else "image_name"
        assert self.name_col in self.df.columns, "Missing image column (image_name or image_path)"
        assert "y_majority" in self.df.columns, "Missing y_majority column"
        self.labels = self.df["y_majority"].astype(int).values

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        name = str(self.df.loc[i, self.name_col])
        p = self.root / name
        x = Image.open(p).convert("RGB")
        x = self.tf(x)
        y = torch.tensor(self.labels[i], dtype=torch.float32)
        return x, y


# ---------------- Model ----------------
class BinaryHead(nn.Module):
    def __init__(self, in_features, p=0.2):
        super().__init__()
        self.drop = nn.Dropout(p)
        self.fc = nn.Linear(in_features, 1)
    def forward(self, feats):
        return self.fc(self.drop(feats)).squeeze(1)

def build_model(model_name: str, pretrained: bool = False):
    # num_classes=0 => pooled features; we attach our head
    backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool="avg")
    head = BinaryHead(backbone.num_features, p=0.2)
    return nn.Sequential(backbone, head)


# ---------------- Evaluation & Calibration ----------------
@torch.no_grad()
def forward_logits(model, loader, device, amp=False):
    model.eval()
    zs, ys = [], []
    bce = nn.BCEWithLogitsLoss(reduction="mean")
    losses = []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        with torch.amp.autocast("cuda", enabled=amp):
            z = model(xb)
            loss = bce(z, yb)
        losses.append(loss.item())
        zs.append(z.detach().cpu().numpy())
        ys.append(yb.detach().cpu().numpy())
    z = np.concatenate(zs)
    y = np.concatenate(ys)
    p = 1.0/(1.0 + np.exp(-z))
    auc = float("nan")
    if roc_auc_score is not None:
        try: auc = roc_auc_score(y, p)
        except Exception: pass
    return {"loss": float(np.mean(losses)), "auc": float(auc)}, y, z

def fit_temperature(z_val, y_val):
    """Fit temperature T >= 1e-3 minimizing BCE-with-logits on val."""
    z = torch.tensor(z_val, dtype=torch.float32)
    y = torch.tensor(y_val, dtype=torch.float32)
    T = torch.ones(1, requires_grad=True)
    opt = torch.optim.LBFGS([T], lr=0.1, max_iter=50, line_search_fn="strong_wolfe")
    bce = nn.BCEWithLogitsLoss()

    def closure():
        opt.zero_grad()
        loss = bce(z / T.clamp_min(1e-3), y)
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        T.clamp_(min=1e-3)
    return float(T.item())

def probs_from_logits(z, T=1.0):
    z = np.array(z, dtype=np.float32) / float(max(T, 1e-3))
    return 1.0/(1.0 + np.exp(-z))

def metrics_at_threshold(y, p, t):
    yhat = (p >= t).astype(int)
    tp = int(((yhat==1)&(y==1)).sum())
    fp = int(((yhat==1)&(y==0)).sum())
    fn = int(((yhat==0)&(y==1)).sum())
    tn = int(((yhat==0)&(y==0)).sum())
    acc = (tp+tn) / max(1, len(y))
    rec = tp / max(1, tp+fn)
    prec = tp / max(1, tp+fp)
    f1 = 2*prec*rec / max(1e-8, prec+rec)
    return {"acc": float(acc), "rec": float(rec), "prec": float(prec), "f1": float(f1),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}

def sweep_metrics(y, p, t_min=0.0, t_max=1.0, steps=1001):
    ts = np.linspace(t_min, t_max, steps)
    accs, precs, recs, f1s = [], [], [], []
    for t in ts:
        m = metrics_at_threshold(y, p, t)
        accs.append(m["acc"]); precs.append(m["prec"]); recs.append(m["rec"]); f1s.append(m["f1"])
    return ts, np.array(accs), np.array(precs), np.array(recs), np.array(f1s)

def pick_threshold(y, p, objective="max_acc", t_min=0.0, t_max=1.0, steps=1001):
    ts = np.linspace(t_min, t_max, steps)
    if objective == "max_acc":
        best = None
        for t in ts:
            m = metrics_at_threshold(y, p, t)
            key = (m["acc"], t)  # prefer larger t on ties
            if best is None or key > best[0]:
                best = (key, t, m)
        _, t_star, m_star = best
        return float(t_star), m_star

    if objective.startswith("recall>="):
        # e.g., "recall>=0.90|max_f1" or "|max_acc"
        rest = objective.split(">=")[1]
        target = float(rest.split("|")[0])
        tie = rest.split("|")[1] if "|" in rest else "max_f1"
        key_name = {"max_f1":"f1","max_acc":"acc","max_prec":"prec"}[tie]
        cand = []
        for t in ts:
            m = metrics_at_threshold(y, p, t)
            if m["rec"] >= target:
                cand.append((m[key_name], t, m))
        if cand:
            cand.sort(key=lambda x: (x[0], x[1]))
            t_star = cand[-1][1]
            m_star = cand[-1][2]
            return float(t_star), m_star
        # fallback: best F1 overall
        return pick_threshold(y, p, objective="max_f1", t_min=t_min, t_max=t_max, steps=steps)

    if objective == "max_f1":
        best = None
        for t in ts:
            m = metrics_at_threshold(y, p, t)
            key = (m["f1"], t)
            if best is None or key > best[0]:
                best = (key, t, m)
        _, t_star, m_star = best
        return float(t_star), m_star

    raise ValueError(f"Unknown objective: {objective}")


# ---------------- Plotting ----------------
def plot_metrics(ts, accs, precs, recs, f1s, t_star, title, outfile):
    plt.figure(figsize=(7.5, 5.5))
    plt.plot(ts, accs, label="Accuracy")
    plt.plot(ts, precs, label="Precision")
    plt.plot(ts, recs, label="Recall")
    plt.plot(ts, f1s, label="F1")
    plt.axvline(t_star, linestyle="--")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    Path(os.path.dirname(outfile) or ".").mkdir(parents=True, exist_ok=True)
    plt.savefig(outfile, dpi=160)
    plt.close()

def plot_roc(y, p, title, outfile):
    if roc_curve is None:
        return
    fpr, tpr, _ = roc_curve(y, p)
    plt.figure(figsize=(5.5, 5.0))
    plt.plot(fpr, tpr)
    plt.plot([0,1],[0,1], linestyle="--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    Path(os.path.dirname(outfile) or ".").mkdir(parents=True, exist_ok=True)
    plt.savefig(outfile, dpi=160)
    plt.close()


# ---------------- Fold runner ----------------
def run_fold(ckpt_path, train_csv, val_csv, test_csv, image_root, model_arg=None, img_size_arg=None, batch_size=32, num_workers=4, amp=False, objective="max_acc", temp_scaling=True, plot_dir=None, steps=1001):
    device = get_device()

    # Load data
    df_val = pd.read_csv(val_csv)
    df_test = pd.read_csv(test_csv) if (test_csv and os.path.exists(test_csv)) else None

    # Rebuild model
    ckpt = torch.load(ckpt_path, map_location="cpu")
    args_in = ckpt.get("args", {})
    model_name = model_arg or args_in.get("model", "tf_efficientnet_b4_ns")
    img_size = int(img_size_arg or args_in.get("img_size", 384))
    model = build_model(model_name, pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Dataloaders
    val_ds = ImgDS(df_val, image_root, img_size=img_size)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = None
    if df_test is not None:
        test_ds = ImgDS(df_test, image_root, img_size=img_size)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # Inference (logits)
    val_metrics, y_val, z_val = forward_logits(model, val_loader, device, amp=amp)
    if test_loader is not None:
        _, y_te, z_te = forward_logits(model, test_loader, device, amp=amp)
    else:
        y_te = z_te = None

    # Temperature scaling (fit on VAL)
    T = 1.0
    if temp_scaling:
        T = fit_temperature(z_val, y_val)

    p_val = probs_from_logits(z_val, T)
    t_star, val_cls = pick_threshold(y_val, p_val, objective=objective, steps=steps)

    test_cls = None
    test_auc = None
    if z_te is not None:
        p_te = probs_from_logits(z_te, T)
        test_cls = metrics_at_threshold(y_te, p_te, t_star)
        if roc_auc_score is not None:
            try:
                test_auc = float(roc_auc_score(y_te, p_te))
            except Exception:
                test_auc = None

    # Plots
    if plot_dir:
        fold_id = parse_fold_id_from_name(ckpt_path, default=-1)
        # VAL metrics vs threshold
        ts, accs, precs, recs, f1s = sweep_metrics(y_val, p_val, steps=steps)
        plot_metrics(ts, accs, precs, recs, f1s, t_star,
                     title=f"VAL Metrics vs Threshold (fold {fold_id})",
                     outfile=os.path.join(plot_dir, f"fold{fold_id}_VAL_metrics.png"))
        plot_roc(y_val, p_val, title=f"VAL ROC (fold {fold_id}, AUC={val_metrics['auc']:.3f})",
                 outfile=os.path.join(plot_dir, f"fold{fold_id}_VAL_ROC.png"))
        # TEST metrics vs threshold (if available)
        if z_te is not None:
            p_te = probs_from_logits(z_te, T)
            ts, accs, precs, recs, f1s = sweep_metrics(y_te, p_te, steps=steps)
            plot_metrics(ts, accs, precs, recs, f1s, t_star,
                         title=f"TEST Metrics vs Threshold (fold {fold_id})",
                         outfile=os.path.join(plot_dir, f"fold{fold_id}_TEST_metrics.png"))
            if roc_curve is not None:
                try:
                    auc_te = roc_auc_score(y_te, p_te) if roc_auc_score is not None else float("nan")
                except Exception:
                    auc_te = float("nan")
                plot_roc(y_te, p_te, title=f"TEST ROC (fold {fold_id}, AUC={auc_te:.3f})",
                         outfile=os.path.join(plot_dir, f"fold{fold_id}_TEST_ROC.png"))

    out = {
        "ckpt": ckpt_path,
        "model": model_name,
        "img_size": img_size,
        "val_auc": float(val_metrics["auc"]),
        "test_auc": test_auc,
        "T": float(T),
        "t_star": float(t_star),
        "val_cls": val_cls,
        "test_cls": test_cls,
    }
    return out


# ---------------- Main ----------------
def main(args):
    set_seed(args.seed)
    ckpt_paths = sorted(glob.glob(args.ckpt_glob))
    assert ckpt_paths, f"No checkpoints match: {args.ckpt_glob}"

    # map by fold id if present in names
    fold_map = {}
    for p in ckpt_paths:
        k = parse_fold_id_from_name(p, default=-1)
        fold_map[k] = p
    if all(k == -1 for k in fold_map.keys()):
        fold_map = {i: p for i, p in enumerate(ckpt_paths)}

    results = []
    for k in sorted(fold_map.keys()):
        ckpt = fold_map[k]
        train_csv = os.path.join(args.folds_dir, f"train_fold{k}.csv")
        val_csv   = os.path.join(args.folds_dir, f"val_fold{k}.csv")
        test_csv  = os.path.join(args.folds_dir, "test.csv") if os.path.exists(os.path.join(args.folds_dir, "test.csv")) else None

        res = run_fold(
            ckpt, train_csv, val_csv, test_csv,
            image_root=args.image_root,
            model_arg=args.model,
            img_size_arg=args.img_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            amp=args.amp,
            objective=args.objective,
            temp_scaling=not args.no_temp,
            plot_dir=args.plot_dir,
            steps=args.steps
        )
        res["fold"] = int(k)
        results.append(res)
        # pretty print fold result
        print(
            f"[Fold {k}] val_auc={res['val_auc']:.4f}  T={res['T']:.3f}  t*={res['t_star']:.3f}  "
            f"VAL(acc={res['val_cls']['acc']:.3f}, prec={res['val_cls']['prec']:.3f}, rec={res['val_cls']['rec']:.3f}, f1={res['val_cls']['f1']:.3f})"
            + ("" if res['test_cls'] is None else
               f"  | TEST(auc={(res['test_auc'] if res['test_auc'] is not None else float('nan')):.4f}, "
               f"acc={res['test_cls']['acc']:.3f}, prec={res['test_cls']['prec']:.3f}, rec={res['test_cls']['rec']:.3f}, f1={res['test_cls']['f1']:.3f})")
        )

    # Aggregate
    val_aucs, test_aucs = [], []
    val_accs, val_precs, val_recs, val_f1s = [], [], [], []
    te_accs, te_precs, te_recs, te_f1s = [], [], [], []
    for r in results:
        val_aucs.append(r["val_auc"])
        vc = r["val_cls"]; val_accs.append(vc["acc"]); val_precs.append(vc["prec"]); val_recs.append(vc["rec"]); val_f1s.append(vc["f1"])
        if r["test_cls"] is not None:
            if r.get("test_auc") is not None:
                test_aucs.append(r["test_auc"])
            tc = r["test_cls"]; te_accs.append(tc["acc"]); te_precs.append(tc["prec"]); te_recs.append(tc["rec"]); te_f1s.append(tc["f1"])

    print("\n=== OVERALL (VAL) ===")
    mauc, sauc = mean_std(val_aucs)
    print(f"AUC  mean±std: {mauc:.3f} ± {sauc:.3f}")
    ma, sa = mean_std(val_accs); mp, sp = mean_std(val_precs); mr, sr = mean_std(val_recs); mf, sf = mean_std(val_f1s)
    print(f"ACC  mean±std: {ma:.3f} ± {sa:.3f}")
    print(f"PREC mean±std: {mp:.3f} ± {sp:.3f}")
    print(f"REC  mean±std: {mr:.3f} ± {sr:.3f}")
    print(f"F1   mean±std: {mf:.3f} ± {sf:.3f}")

    if te_accs:
        print("\n=== OVERALL (TEST) ===")
        if test_aucs:
            mauc, sauc = mean_std(test_aucs)
            print(f"AUC  mean±std: {mauc:.3f} ± {sauc:.3f}")
        ma, sa = mean_std(te_accs); mp, sp = mean_std(te_precs); mr, sr = mean_std(te_recs); mf, sf = mean_std(te_f1s)
        print(f"ACC  mean±std: {ma:.3f} ± {sa:.3f}")
        print(f"PREC mean±std: {mp:.3f} ± {sp:.3f}")
        print(f"REC  mean±std: {mr:.3f} ± {sr:.3f}")
        print(f"F1   mean±std: {mf:.3f} ± {sf:.3f}")

    # Save outputs
    if args.save_json:
        Path(os.path.dirname(args.save_json) or ".").mkdir(parents=True, exist_ok=True)
        with open(args.save_json, "w") as f:
            json.dump(results, f, indent=2)
    if args.save_csv:
        rows = []
        for r in results:
            row = {
                "fold": r["fold"], "ckpt": r["ckpt"], "model": r["model"], "img_size": r["img_size"],
                "val_auc": r["val_auc"], "T": r["T"], "t_star": r["t_star"],
                "val_acc": r["val_cls"]["acc"], "val_prec": r["val_cls"]["prec"], "val_rec": r["val_cls"]["rec"], "val_f1": r["val_cls"]["f1"]
            }
            if r["test_cls"] is not None:
                row.update({
                    "test_auc": r.get("test_auc"),
                    "test_acc": r["test_cls"]["acc"],
                    "test_prec": r["test_cls"]["prec"],
                    "test_rec": r["test_cls"]["rec"],
                    "test_f1": r["test_cls"]["f1"],
                })
            rows.append(row)
        df = pd.DataFrame(rows)
        Path(os.path.dirname(args.save_csv) or ".").mkdir(parents=True, exist_ok=True)
        df.to_csv(args.save_csv, index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser("Threshold sweep + temperature scaling evaluator (with plots)")
    ap.add_argument("--image-root", type=str, required=True)
    ap.add_argument("--folds-dir", type=str, required=True)
    ap.add_argument("--ckpt-glob", type=str, required=True)
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--img-size", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--objective", type=str, default="max_acc", help='Options: "max_acc", "max_f1", or "recall>=0.90|max_f1" etc.')
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--no-temp", action="store_true", help="Disable temperature scaling (default is enabled).")
    ap.add_argument("--steps", type=int, default=1001, help="Number of thresholds in the sweep.")
    ap.add_argument("--plot-dir", type=str, default=None, help="Directory to save per-fold plots (VAL/TEST curves and ROC).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save-json", type=str, default=None)
    ap.add_argument("--save-csv", type=str, default=None)
    args = ap.parse_args()
    main(args)
