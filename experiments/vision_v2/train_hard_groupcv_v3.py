# experiments/vision_v2/train_hard_groupcv_v3.py
import argparse, os, json, random
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import timm

try:
    from sklearn.metrics import roc_auc_score
except Exception:
    roc_auc_score = None


# ---------------- Utils ----------------
def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------- Simple binary-safe MixUp / CutMix ----------
def _rand_beta(alpha: float, size: int, device):
    # Sample lam in (0,1); symmetric Beta
    lam = np.random.beta(alpha, alpha, size) if alpha > 0 else np.ones(size)
    lam = torch.tensor(lam, dtype=torch.float32, device=device)
    return torch.clamp(lam, 0.0 + 1e-6, 1.0 - 1e-6)

def apply_mixup_cutmix(x, y, p_mixup=0.0, p_cutmix=0.0, mixup_alpha=0.2, cutmix_alpha=0.2):
    """
    x: (N,C,H,W), y: (N,) float in {0,1}
    Returns (x', y') same batch size N.
    """
    N, _, H, W = x.shape
    device = x.device
    do_mix = (np.random.rand() < p_mixup)
    do_cut = (np.random.rand() < p_cutmix)

    if not do_mix and not do_cut:
        return x, y

    # Pair indices
    perm = torch.randperm(N, device=device)
    x2 = x[perm]
    y2 = y[perm]

    if do_mix and not do_cut:
        lam = _rand_beta(mixup_alpha, 1, device).item()
        x_m = lam * x + (1 - lam) * x2
        y_m = lam * y + (1 - lam) * y2
        return x_m, y_m

    if do_cut and not do_mix:
        lam = _rand_beta(cutmix_alpha, 1, device).item()
        # cutout box
        cut_w = int(W * (1 - lam) ** 0.5)
        cut_h = int(H * (1 - lam) ** 0.5)
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        x1 = np.clip(cx - cut_w // 2, 0, W)
        x2c = np.clip(cx + cut_w // 2, 0, W)
        y1 = np.clip(cy - cut_h // 2, 0, H)
        y2c = np.clip(cy + cut_h // 2, 0, H)
        x_clone = x.clone()
        x_clone[:, :, y1:y2c, x1:x2c] = x2[:, :, y1:y2c, x1:x2c]
        lam_eff = 1 - ((x2c - x1) * (y2c - y1) / (W * H + 1e-6))
        y_m = lam_eff * y + (1 - lam_eff) * y[perm]
        return x_clone, y_m

    # If both are triggered, apply MixUp then CutMix on mixed tensor
    xm, ym = apply_mixup_cutmix(x, y, p_mixup=1.0, p_cutmix=0.0, mixup_alpha=mixup_alpha, cutmix_alpha=cutmix_alpha)
    xm, ym = apply_mixup_cutmix(xm, ym, p_mixup=0.0, p_cutmix=1.0, mixup_alpha=mixup_alpha, cutmix_alpha=cutmix_alpha)
    return xm, ym


# --------------- Dataset ---------------
class ImgDS(Dataset):
    def __init__(self, df, root, img_size=384, augment=False, mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)):
        self.df = df.reset_index(drop=True).copy()
        self.root = Path(root)
        self.img_size = img_size
        self.augment = augment
        from torchvision import transforms as T
        self.tf_train = T.Compose([
            T.Resize((img_size, img_size)),
            T.RandomHorizontalFlip(0.5),
            T.RandomVerticalFlip(0.1),
            T.RandomRotation(10),
            T.ColorJitter(0.15, 0.15, 0.1, 0.02),
            T.ToTensor(), T.Normalize(mean, std),
        ])
        self.tf_eval = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(), T.Normalize(mean, std),
        ])
        self.name_col = "image_path" if "image_path" in self.df.columns else "image_name"
        assert self.name_col in self.df.columns
        assert "y_majority" in self.df.columns, "Need y_majority column"
        self.labels = self.df["y_majority"].astype(int).values

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        name = str(self.df.loc[i, self.name_col])
        p = self.root / name
        img = Image.open(p).convert("RGB")
        x = (self.tf_train if self.augment else self.tf_eval)(img)
        y = torch.tensor(self.labels[i], dtype=torch.float32)
        return x, y


# --------------- Model -----------------
class BinaryHead(nn.Module):
    def __init__(self, in_features, p=0.2):
        super().__init__()
        self.drop = nn.Dropout(p)
        self.fc = nn.Linear(in_features, 1)
    def forward(self, feats):
        return self.fc(self.drop(feats)).squeeze(1)

def build_model(model_name, pretrained=True):
    backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool="avg")
    head = BinaryHead(backbone.num_features, p=0.2)
    return nn.Sequential(backbone, head)


# ------------- Metrics & thresholds -------------
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    bce = nn.BCEWithLogitsLoss(reduction="mean")
    losses, ys, ps = [], [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = bce(logits, yb)
        losses.append(loss.item())
        ps.append(torch.sigmoid(logits).cpu().numpy())
        ys.append(yb.cpu().numpy())
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    auc = float("nan")
    if roc_auc_score is not None:
        try: auc = roc_auc_score(y, p)
        except Exception: pass
    return {"loss": float(np.mean(losses)), "auc": float(auc)}, y, p

def tune_threshold(y, p, objective="recall>=0.90|max_f1", t_min=0.0, t_max=1.0):
    import numpy as np
    ts = np.linspace(t_min, t_max, 1001)
    def metrics_at(t):
        yhat = (p >= t).astype(int)
        tp = ((yhat==1)&(y==1)).sum()
        fp = ((yhat==1)&(y==0)).sum()
        fn = ((yhat==0)&(y==1)).sum()
        tn = ((yhat==0)&(y==0)).sum()
        rec = tp / max(1, tp+fn)
        prec = tp / max(1, tp+fp)
        acc = (tp+tn) / max(1, len(y))
        f1 = 2*prec*rec / max(1e-8, prec+rec)
        return {"t": float(t), "acc":acc, "f1":f1, "prec":prec, "rec":rec}

    if objective.startswith("recall>="):
        # parse like "recall>=0.90|max_f1" or "recall>=0.90|max_acc" or "recall>=0.90|max_prec"
        rest = objective.split(">=")[1]
        target = float(rest.split("|")[0])
        tie = rest.split("|")[1] if "|" in rest else "max_f1"
        cand = [m for t in ts for m in [metrics_at(t)] if m["rec"] >= target]
        if cand:
            key = {"max_f1":"f1","max_acc":"acc","max_prec":"prec"}[tie]
            best = max(cand, key=lambda m: (m[key], m["t"]))  # prefer larger t if tie
            return best["t"], {k:best[k] for k in ("acc","f1","prec","rec")}
        # fallback if target recall is impossible:
        # pick the globally best F1 (or acc) without the constraint
        key = {"max_f1":"f1","max_acc":"acc","max_prec":"prec"}[tie]
        best = max((metrics_at(t) for t in ts), key=lambda m: m[key])
        return best["t"], {k:best[k] for k in ("acc","f1","prec","rec")}

    if objective == "max_acc":
        best = max((metrics_at(t) for t in ts), key=lambda m: m["acc"])
        return best["t"], {k:best[k] for k in ("acc","f1","prec","rec")}
    if objective == "max_f1":
        best = max((metrics_at(t) for t in ts), key=lambda m: m["f1"])
        return best["t"], {k:best[k] for k in ("acc","f1","prec","rec")}
    return 0.5, {"acc":0,"f1":0,"prec":0,"rec":0}


# ------------- Train loop -------------
def train_one_fold(args, fold_id, df_train, df_val, df_test):
    device = get_device()
    set_seed(args.seed + fold_id)

    tr_ds = ImgDS(df_train, args.image_root, img_size=args.img_size, augment=True)
    va_ds = ImgDS(df_val,   args.image_root, img_size=args.img_size, augment=False)
    te_ds = ImgDS(df_test,  args.image_root, img_size=args.img_size, augment=False) if df_test is not None else None

    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,  num_workers=args.num_workers, pin_memory=True)
    va_loader = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    te_loader = DataLoader(te_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True) if te_ds else None

    model = build_model(args.model, pretrained=True).to(device)
    backbone = model[0]
    for p in backbone.parameters(): p.requires_grad = False

    head_params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(head_params, lr=args.lr_head, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.freeze_epochs)

    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
    ema = timm.utils.ModelEmaV2(model, decay=0.999) if args.ema else None
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(args.pos_weight).to(device) if args.pos_weight>0 else None)

    best_auc, best_path = -1.0, Path(args.outdir) / f"{args.run_name}_fold{fold_id}.pt"
    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    # Phase 1: frozen backbone
    for epoch in range(1, args.freeze_epochs+1):
        model.train(); losses=[]
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=args.amp):
                xb2, yb2 = apply_mixup_cutmix(
                    xb, yb,
                    p_mixup=args.mixup_prob,
                    p_cutmix=args.cutmix_prob,
                    mixup_alpha=args.mixup_alpha,
                    cutmix_alpha=args.cutmix_alpha,
                )
                logits = model(xb2)
                loss = bce(logits, yb2)
            scaler.scale(loss).backward()
            scaler.step(optim); scaler.update()
            if ema: ema.update(model)
            losses.append(loss.item())
        sched.step()
        eval_model = ema.module if ema else model
        val_metrics, _, _ = evaluate(eval_model, va_loader, device)
        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]; torch.save({"model": eval_model.state_dict(), "args": vars(args)}, best_path)
        print(f"[Fold {fold_id}] Epoch {epoch}/{args.freeze_epochs} (frozen) "
              f"train_loss={np.mean(losses):.4f} val_auc={val_metrics['auc']:.4f}")

    # Phase 2: unfreeze whole backbone
    for p in backbone.parameters(): p.requires_grad = True
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr_all, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.unfreeze_epochs)

    patience_ctr = 0
    for epoch in range(1, args.unfreeze_epochs+1):
        model.train(); losses=[]
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=args.amp):
                xb2, yb2 = apply_mixup_cutmix(
                    xb, yb,
                    p_mixup=args.mixup_prob,
                    p_cutmix=args.cutmix_prob,
                    mixup_alpha=args.mixup_alpha,
                    cutmix_alpha=args.cutmix_alpha,
                )
                logits = model(xb2)
                loss = bce(logits, yb2)
            scaler.scale(loss).backward()
            if args.grad_clip>0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optim); scaler.update()
            if ema: ema.update(model)
            losses.append(loss.item())
        sched.step()

        eval_model = ema.module if ema else model
        val_metrics, _, _ = evaluate(eval_model, va_loader, device)
        improved = val_metrics["auc"] > best_auc + 1e-6
        if improved:
            best_auc = val_metrics["auc"]; patience_ctr = 0
            torch.save({"model": eval_model.state_dict(), "args": vars(args)}, best_path)
        else:
            patience_ctr += 1
        print(f"[Fold {fold_id}] Epoch {epoch}/{args.unfreeze_epochs} train_loss={np.mean(losses):.4f} "
              f"val_auc={val_metrics['auc']:.4f} best_auc={best_auc:.4f} patience={patience_ctr}/{args.patience}")
        if patience_ctr >= args.patience: break

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])

    val_metrics, y_val, p_val = evaluate(model, va_loader, device)
    t_star, val_cls = tune_threshold(y_val, p_val, objective=args.threshold_objective)
    results = {"fold": fold_id, "val_auc": val_metrics["auc"], "t_star": t_star, "val_cls": val_cls}

    if te_loader is not None:
        _, y_te, p_te = evaluate(model, te_loader, device)
        yhat_te = (p_te >= t_star).astype(int)
        acc = (yhat_te == y_te).mean()
        prec = ((yhat_te==1)&(y_te==1)).sum()/max(1,(yhat_te==1).sum())
        rec = ((yhat_te==1)&(y_te==1)).sum()/max(1,(y_te==1).sum())
        f1  = 2*prec*rec/max(1e-8,prec+rec)
        results["test_cls"] = {"acc": float(acc), "f1": float(f1), "prec": float(prec), "rec": float(rec)}
    return results, str(Path(args.outdir)/f"{args.run_name}_fold{fold_id}.pt")


def load_split_files(args):
    tr = pd.read_csv(args.train_csv)
    va = pd.read_csv(args.val_csv)
    te = pd.read_csv(args.test_csv) if args.test_csv and os.path.exists(args.test_csv) else None
    return tr, va, te

def main(args):
    set_seed(args.seed)
    res_all, best_paths = [], []
    if args.via_folds_dir:
        for k in range(args.folds):
            args.train_csv = os.path.join(args.folds_dir, f"train_fold{k}.csv")
            args.val_csv   = os.path.join(args.folds_dir, f"val_fold{k}.csv")
            if os.path.exists(os.path.join(args.folds_dir, "test.csv")):
                args.test_csv  = os.path.join(args.folds_dir, "test.csv")
            tr, va, te = load_split_files(args)
            fold_res, best_path = train_one_fold(args, k, tr, va, te)
            res_all.append(fold_res); best_paths.append(best_path)
    else:
        tr, va, te = load_split_files(args)
        fold_res, best_path = train_one_fold(args, args.fold_id, tr, va, te)
        res_all.append(fold_res); best_paths.append(best_path)

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.outdir)/f"{args.run_name}_results.json", "w") as f:
        json.dump(res_all, f, indent=2)
    print(json.dumps(res_all, indent=2))
    print("Best checkpoints:", *best_paths, sep="\n - ")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    # data
    p.add_argument("--image-root", type=str, required=True)
    p.add_argument("--train-csv", type=str, default=None)
    p.add_argument("--val-csv", type=str, default=None)
    p.add_argument("--test-csv", type=str, default=None)
    p.add_argument("--via-folds-dir", action="store_true")
    p.add_argument("--folds-dir", type=str, default="data/splits_grouped")
    p.add_argument("--folds", type=int, default=5)

    # training
    p.add_argument("--model", type=str, default="tf_efficientnet_b4_ns")
    p.add_argument("--img-size", type=int, default=384)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--freeze-epochs", type=int, default=3)
    p.add_argument("--unfreeze-epochs", type=int, default=30)
    p.add_argument("--lr-head", type=float, default=8e-4)
    p.add_argument("--lr-all", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--pos-weight", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--ema", action="store_true")

    # our custom mixup/cutmix
    p.add_argument("--mixup-prob", type=float, default=0.4)
    p.add_argument("--cutmix-prob", type=float, default=0.2)
    p.add_argument("--mixup-alpha", type=float, default=0.2)
    p.add_argument("--cutmix-alpha", type=float, default=0.2)

    # early stop / thresholding
    p.add_argument("--label-smoothing", type=float, default=0.05)  # unused now; retained for CLI compatibility
    p.add_argument("--patience", type=int, default=7)
    p.add_argument("--threshold-objective", type=str, default="recall>=0.90")

    # misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--outdir", type=str, default="weights/v3")
    p.add_argument("--run-name", type=str, default="hard_groupcv_v3")
    p.add_argument("--fold-id", type=int, default=0)
    args = p.parse_args()
    main(args)
