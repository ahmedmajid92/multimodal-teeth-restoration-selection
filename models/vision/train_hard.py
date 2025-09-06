# models/vision/train_hard.py
import argparse, os, sys, time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, GroupShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.amp import autocast

# Add project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from models.vision.datasets import TeethImageDataset
from models.vision.model_factory import create_model
from models.vision.utils import AvgMeter, seed_all, binary_metrics_from_logits_2class, save_checkpoint

def _split_train_val(train_df: pd.DataFrame, val_frac: float, seed: int, group_col: str = None):
    if group_col and group_col in train_df.columns:
        print(f"🔒 Grouped split on '{group_col}'")
        gss = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
        tr_idx, va_idx = next(gss.split(train_df, groups=train_df[group_col].values))
        return train_df.iloc[tr_idx].reset_index(drop=True), train_df.iloc[va_idx].reset_index(drop=True)
    if "origin_id" in train_df.columns:
        print("🔒 Grouped split on 'origin_id'")
        gss = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
        tr_idx, va_idx = next(gss.split(train_df, groups=train_df["origin_id"].values))
        return train_df.iloc[tr_idx].reset_index(drop=True), train_df.iloc[va_idx].reset_index(drop=True)
    print("🧪 Fallback stratified split on y_majority")
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
    tr_idx, va_idx = next(sss.split(train_df, train_df["y_majority"]))
    return train_df.iloc[tr_idx].reset_index(drop=True), train_df.iloc[va_idx].reset_index(drop=True)

def make_loaders(csv_path, images_root, img_size, batch_size, num_workers, val_frac=0.12, seed=42, group_col=None):
    df = pd.read_csv(csv_path)
    has_val = (df["split"].astype(str).str.lower() == "val").any()
    if has_val:
        train_df = df[df["split"].str.lower() == "train"].copy()
        val_df   = df[df["split"].str.lower() == "val"].copy()
        test_df  = df[df["split"].str.lower() == "test"].copy()
        print(f"📈 Total={len(df)} | Train={len(train_df)} | Val={len(val_df)} | Test={len(test_df)}")
        train_sub, val_sub = train_df.reset_index(drop=True), val_df.reset_index(drop=True)
    else:
        train_df = df[df["split"].str.lower() == "train"].copy()
        test_df  = df[df["split"].str.lower() == "test"].copy()
        print(f"📈 Total={len(df)} | Train={len(train_df)} | Test={len(test_df)}")
        train_sub, val_sub = _split_train_val(train_df, val_frac, seed, group_col)

    def dist(d):
        vc = d["y_majority"].astype(int).value_counts().sort_index()
        return {int(k): int(v) for k, v in vc.items()}
    print(f"🏷️  Train dist: {dist(train_sub)}")
    print(f"🏷️  Val   dist: {dist(val_sub)}")

    ds_train = TeethImageDataset(csv_path, images_root, split="train", task="hard",
                                 img_size=img_size, aug=True, df_override=train_sub)
    ds_val   = TeethImageDataset(csv_path, images_root, split="train", task="hard",
                                 img_size=img_size, aug=False, df_override=val_sub)
    ds_test  = TeethImageDataset(csv_path, images_root, split="test", task="hard",
                                 img_size=img_size, aug=False)

    # Weighted sampler for imbalance
    class_counts = np.bincount(train_sub["y_majority"].astype(int).values, minlength=2)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = class_weights[train_sub["y_majority"].astype(int).values]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    print(f"⚖️  Class weights: {class_weights}")

    dl_train = DataLoader(ds_train, batch_size=batch_size, sampler=sampler,
                          num_workers=num_workers, pin_memory=True, drop_last=True)
    dl_val   = DataLoader(ds_val, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=True)
    dl_test  = DataLoader(ds_test, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=True)
    print(f"🔄 Batches | Train: {len(dl_train)}  Val: {len(dl_val)}  Test: {len(dl_test)}")
    return dl_train, dl_val, dl_test

def train_one_epoch(model, loader, optimizer, scaler, device, criterion, epoch, total_epochs):
    model.train()
    meter = AvgMeter()
    t0 = time.time()
    for bi, (imgs, ys) in enumerate(loader):
        imgs = imgs.to(device, non_blocking=True); ys = ys.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast('cuda'):
            logits = model(imgs)
            loss = criterion(logits, ys)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        meter.update(loss.item(), imgs.size(0))
        if (bi+1) % 10 == 0 or (bi+1) == len(loader):
            print(f"    [{bi+1:3d}/{len(loader):3d}] loss {loss.item():.4f}  avg {meter.avg:.4f}  lr {optimizer.param_groups[0]['lr']:.2e}")
    print(f"✅ Epoch {epoch}/{total_epochs}  time {time.time()-t0:.1f}s  avg_loss {meter.avg:.4f}")
    return meter.avg

@torch.no_grad()
def evaluate(model, loader, device, criterion, name="Val", tta=False, return_arrays=False):
    """
    If return_arrays=True, also returns (probs1, targets) arrays for custom-threshold metrics.
    probs1 are P(class=1) from softmax over logits.
    """
    model.eval()
    meter = AvgMeter()
    all_logits, all_targets = [], []
    t0 = time.time()
    for bi, (imgs, ys) in enumerate(loader):
        imgs = imgs.to(device, non_blocking=True); ys = ys.to(device, non_blocking=True)
        with autocast('cuda'):
            logits = model(imgs)
            if tta:
                logits_flip = model(torch.flip(imgs, dims=[3]))  # hflip TTA
                logits = (logits + logits_flip) / 2
            loss = criterion(logits, ys)
        meter.update(loss.item(), imgs.size(0))
        all_logits.append(logits); all_targets.append(ys)
        if (bi+1) % 10 == 0 or (bi+1) == len(loader):
            print(f"    {name} [{bi+1:3d}/{len(loader):3d}] loss {loss.item():.4f}")
    logits = torch.cat(all_logits); targets = torch.cat(all_targets)
    metrics = binary_metrics_from_logits_2class(logits, targets)  # default threshold=0.5 inside
    metrics["loss"] = meter.avg
    print(f"✅ {name} done in {time.time()-t0:.1f}s")
    if return_arrays:
        probs1 = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        y_true = targets.detach().cpu().numpy()
        return metrics, probs1, y_true
    return metrics

def grid_best_threshold(probs, y, lo=0.05, hi=0.95, step=0.005):
    ths = np.arange(lo, hi + 1e-9, step)
    best = (0.5, -1.0)
    for t in ths:
        yp = (probs >= t).astype(int)
        f1 = f1_score(y, yp)
        if f1 > best[1]:
            best = (t, f1)
    return best  # (threshold, f1_at_threshold)

def metrics_with_threshold(probs, y, thr):
    yp = (probs >= thr).astype(int)
    return {
        "acc":  accuracy_score(y, yp),
        "f1":   f1_score(y, yp),
        "prec": precision_score(y, yp),
        "rec":  recall_score(y, yp),
        "auc":  roc_auc_score(y, probs),
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv-path", default="data/excel/data_dl_augmented.csv")
    p.add_argument("--images-root", default="data/augmented")
    p.add_argument("--model-name", default="tf_efficientnet_b3_ns")
    p.add_argument("--img-size", type=int, default=512)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=12)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--val-frac", type=float, default=0.12)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="weights/vision_hard_best.pt")
    p.add_argument("--group-col", default=None, help="Optional column for grouped split (e.g., origin_id)")
    p.add_argument("--label-smoothing", type=float, default=0.05)
    # NEW:
    p.add_argument("--tune-threshold", action="store_true", help="Find best F1 threshold on Val and use it on Test")
    p.add_argument("--tta", action="store_true", help="Use simple TTA (hflip) during evaluation")
    p.add_argument("--thr-range", default="0.05,0.95,0.005", help="lo,hi,step for threshold grid search")
    args = p.parse_args()

    print("="*80)
    print("🦷 HARD LABELS TRAINING")
    for k,v in vars(args).items(): print(f"{k:>18}: {v}")
    print("="*80)

    seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Device: {device}")
    if device == "cuda":
        print(f"🎮 GPU: {torch.cuda.get_device_name()} "
              f"({torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)")

    dl_train, dl_val, dl_test = make_loaders(
        args.csv_path, args.images_root, args.img_size,
        args.batch_size, args.num_workers, args.val_frac, args.seed, args.group_col
    )

    print("\n🏗️  Building model…")
    model = create_model(args.model_name, num_classes=2, pretrained=True)
    model.to(device)
    print(f"📊 Params: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler()

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, dl_train, optimizer, scaler, device, criterion, epoch, args.epochs)
        val_metrics = evaluate(model, dl_val, device, criterion, "Val", tta=args.tta)
        scheduler.step()

        print(f"\n📊 EPOCH {epoch:02d}/{args.epochs}")
        print(f"    train_loss {train_loss:.4f}")
        print(f"    val_loss   {val_metrics['loss']:.4f}  "
              f"acc {val_metrics['acc']:.4f}  f1 {val_metrics['f1']:.4f}  "
              f"prec {val_metrics['prec']:.4f}  rec {val_metrics['rec']:.4f}  auc {val_metrics['auc']:.4f}")

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            print(f"    🏆 New best (val_loss={best_val:.4f}) → {args.out}")
            save_checkpoint({"model": model.state_dict(),
                             "model_name": args.model_name,
                             "img_size": args.img_size}, args.out)
        print("-"*80)

    print("\n🧪 Loading best model for VAL & TEST…")
    ckpt = torch.load(args.out, map_location=device)
    model.load_state_dict(ckpt["model"])

    # Optional: find best threshold on Val
    best_thr = 0.5
    if args.tune_threshold:
        lo, hi, step = map(float, args.thr_range.split(","))
        val_metrics, val_probs, val_y = evaluate(model, dl_val, device, criterion, "Val", tta=args.tta, return_arrays=True)
        best_thr, best_f1 = grid_best_threshold(val_probs, val_y, lo, hi, step)
        tuned = metrics_with_threshold(val_probs, val_y, best_thr)
        print(f"\n🎯 Tuned threshold on Val: t*={best_thr:.3f} (F1={best_f1:.4f})")
        print(f"    Val@t*: acc {tuned['acc']:.4f}  f1 {tuned['f1']:.4f}  prec {tuned['prec']:.4f}  rec {tuned['rec']:.4f}  auc {tuned['auc']:.4f}")

    # Final Test evaluation (with same TTA and tuned threshold if requested)
    test_metrics, test_probs, test_y = evaluate(model, dl_test, device, criterion, "Test", tta=args.tta, return_arrays=True)
    if args.tune_threshold:
        test_tuned = metrics_with_threshold(test_probs, test_y, best_thr)
        print("\n🏁 FINAL TEST (threshold-tuned)")
        print(f"    acc  {test_tuned['acc']:.4f}")
        print(f"    f1   {test_tuned['f1']:.4f}")
        print(f"    prec {test_tuned['prec']:.4f}")
        print(f"    rec  {test_tuned['rec']:.4f}")
        print(f"    auc  {test_tuned['auc']:.4f}")
    else:
        print("\n🏁 FINAL TEST")
        print(f"    acc  {test_metrics['acc']:.4f}")
        print(f"    f1   {test_metrics['f1']:.4f}")
        print(f"    prec {test_metrics['prec']:.4f}")
        print(f"    rec  {test_metrics['rec']:.4f}")
        print(f"    auc  {test_metrics['auc']:.4f}")
    print("="*80)

if __name__ == "__main__":
    main()
