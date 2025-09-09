# models/vision/train_soft.py
import argparse, os, sys, time
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.amp import autocast

# --- Make repo imports work regardless of CWD ---
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
# ------------------------------------------------

from models.vision.datasets import TeethImageDataset
from models.vision.model_factory import create_model
from models.vision.utils import (
    AvgMeter, seed_all, save_checkpoint,
    binary_metrics_from_logits_single,  # returns acc@0.5, auc, brier, mae
)

def _split_train_val(train_df: pd.DataFrame, val_frac: float, seed: int, group_col: str | None = None):
    """
    Prefer grouped split (family leakage safe). If no group column, random split (labels are continuous).
    """
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

    print("🧪 No group column found; using random split (ShuffleSplit).")
    ss = ShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
    tr_idx, va_idx = next(ss.split(train_df))
    return train_df.iloc[tr_idx].reset_index(drop=True), train_df.iloc[va_idx].reset_index(drop=True)

def make_loaders(csv_path, images_root, img_size, batch_size, num_workers, val_frac=0.12, seed=42, group_col=None):
    df = pd.read_csv(csv_path)

    has_val = (df["split"].astype(str).str.lower() == "val").any()
    if has_val:
        train_df = df[df["split"].str.lower() == "train"].copy()
        val_df   = df[df["split"].str.lower() == "val"].copy()
        test_df  = df[df["split"].str.lower() == "test"].copy()
        print(f"📈 Total={len(df)} | Train={len(train_df)} | Val={len(val_df)} | Test={len(test_df)}")
        tr_sub, va_sub = train_df.reset_index(drop=True), val_df.reset_index(drop=True)
    else:
        train_df = df[df["split"].str.lower() == "train"].copy()
        test_df  = df[df["split"].str.lower() == "test"].copy()
        print(f"📈 Total={len(df)} | Train={len(train_df)} | Test={len(test_df)}")
        tr_sub, va_sub = _split_train_val(train_df, val_frac, seed, group_col)

    ds_train = TeethImageDataset(csv_path, images_root, split="train", task="soft",
                                 img_size=img_size, aug=True,  df_override=tr_sub)
    ds_val   = TeethImageDataset(csv_path, images_root, split="train", task="soft",
                                 img_size=img_size, aug=False, df_override=va_sub)
    ds_test  = TeethImageDataset(csv_path, images_root, split="test",  task="soft",
                                 img_size=img_size, aug=False)

    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True, drop_last=True)
    dl_val   = DataLoader(ds_val,   batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=True)
    dl_test  = DataLoader(ds_test,  batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=True)
    print(f"🔄 Batches | Train: {len(dl_train)}  Val: {len(dl_val)}  Test: {len(dl_test)}")
    return dl_train, dl_val, dl_test

def train_one_epoch(model, loader, optimizer, scaler, device, pos_weight=None):
    """
    Soft labels training: BCEWithLogits with optional per-sample weights from CSV ('weight' column).
    """
    model.train()
    loss_meter = AvgMeter()
    bce = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)
    t0 = time.time()

    for bi, (imgs, y, w) in enumerate(loader):
        imgs = imgs.to(device, non_blocking=True)   # [B,3,H,W]
        y    = y.to(device, non_blocking=True)      # [B,1]
        w    = w.to(device, non_blocking=True)      # [B,1]

        optimizer.zero_grad(set_to_none=True)
        with autocast('cuda'):
            logits = model(imgs)                    # [B,1]
            loss_vec = bce(logits, y)              # [B,1]
            loss = (loss_vec * w).mean()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_meter.update(loss.item(), imgs.size(0))
        if (bi + 1) % 10 == 0 or (bi + 1) == len(loader):
            print(f"    [{bi+1:3d}/{len(loader):3d}] loss {loss.item():.4f}  avg {loss_meter.avg:.4f}  lr {optimizer.param_groups[0]['lr']:.2e}")

    print(f"✅ Train epoch time {time.time()-t0:.1f}s  avg_loss {loss_meter.avg:.4f}")
    return loss_meter.avg

@torch.no_grad()
def evaluate(model, loader, device, tta=False):
    """
    Returns metrics dict with: loss (BCE mean), acc@0.5, auc, brier, mae.
    """
    model.eval()
    loss_meter = AvgMeter()
    bce = nn.BCEWithLogitsLoss(reduction="mean")

    all_logits, all_targets = [], []
    t0 = time.time()
    for bi, batch in enumerate(loader):
        # batch may be (imgs,y,w) for val/test; ignore w here
        imgs, y = batch[0], batch[1]
        imgs = imgs.to(device, non_blocking=True)
        y    = y.to(device, non_blocking=True)
        with autocast('cuda'):
            logits = model(imgs)
            if tta:
                logits_flip = model(torch.flip(imgs, dims=[3]))  # hflip TTA
                logits = (logits + logits_flip) / 2
            loss = bce(logits, y)
        loss_meter.update(loss.item(), imgs.size(0))
        all_logits.append(logits)
        all_targets.append(y)

        if (bi + 1) % 10 == 0 or (bi + 1) == len(loader):
            print(f"    Eval [{bi+1:3d}/{len(loader):3d}] loss {loss.item():.4f}")

    logits = torch.cat(all_logits)   # [N,1]
    targets = torch.cat(all_targets) # [N,1]
    metrics = binary_metrics_from_logits_single(logits, targets)
    metrics["loss"] = loss_meter.avg
    print(f"✅ Eval done in {time.time()-t0:.1f}s")
    return metrics

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
    p.add_argument("--out", default="weights/vision_soft_best.pt")
    p.add_argument("--group-col", default="origin_id", help="Family column to avoid leakage (default: origin_id)")
    p.add_argument("--tta", action="store_true", help="Use simple TTA (hflip) during evaluation")
    p.add_argument("--pos-weight", type=float, default=None,
                   help="Optional BCE pos_weight (e.g., 1.2). None disables it.")
    args = p.parse_args()

    print("=" * 80)
    print("🦷 SOFT LABELS TRAINING — Tooth Restoration (probabilistic)")
    for k, v in vars(args).items():
        print(f"{k:>18}: {v}")
    print("=" * 80)

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
    model = create_model(args.model_name, num_classes=1, pretrained=True)  # single logit
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_train  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Params: total={n_params:,}  trainable={n_train:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler()

    pos_weight_tensor = None
    if args.pos_weight is not None:
        pos_weight_tensor = torch.tensor([float(args.pos_weight)], device=device)

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        print(f"\n🚀 Epoch {epoch}/{args.epochs}")
        train_loss = train_one_epoch(model, dl_train, optimizer, scaler, device, pos_weight=pos_weight_tensor)
        val_metrics = evaluate(model, dl_val, device, tta=args.tta)
        scheduler.step()

        print(f"\n📊 EPOCH {epoch:02d}/{args.epochs} SUMMARY")
        print(f"    train_loss {train_loss:.4f}")
        print(f"    val_loss   {val_metrics['loss']:.4f}  "
              f"acc@0.5 {val_metrics['acc']:.4f}  auc {val_metrics['auc']:.4f}  "
              f"brier {val_metrics['brier']:.4f}  mae {val_metrics['mae']:.4f}")

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            print(f"    🏆 New best (val_loss={best_val:.4f}) → {args.out}")
            save_checkpoint({"model": model.state_dict(),
                             "model_name": args.model_name,
                             "img_size": args.img_size}, args.out)
        print("-" * 80)

    print("\n🧪 Loading best model for TEST…")
    ckpt = torch.load(args.out, map_location=device)
    model.load_state_dict(ckpt["model"])
    test_metrics = evaluate(model, dl_test, device, tta=args.tta)

    print("\n🏁 FINAL TEST")
    print(f"    acc@0.5 {test_metrics['acc']:.4f}")
    print(f"    auc     {test_metrics['auc']:.4f}")
    print(f"    brier   {test_metrics['brier']:.4f}")
    print(f"    mae     {test_metrics['mae']:.4f}")
    print("=" * 80)

if __name__ == "__main__":
    main()
