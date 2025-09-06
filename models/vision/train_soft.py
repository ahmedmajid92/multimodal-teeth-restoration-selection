# models/vision/train_soft.py
import argparse, time
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit
import torch
from torch import nn
from torch.utils.data import DataLoader

from torch.amp import autocast
from models.vision.datasets import TeethImageDataset
from models.vision.model_factory import create_model
from models.vision.utils import AvgMeter, seed_all, binary_metrics_from_logits_single, save_checkpoint

def split_train_val(train_df: pd.DataFrame, val_frac: float, seed: int):
    """Prefer grouped split on origin_id; else random split (labels are continuous)."""
    if "origin_id" in train_df.columns:
        print("🔒 Using grouped split on 'origin_id'")
        gss = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
        tr_idx, va_idx = next(gss.split(train_df, groups=train_df["origin_id"].values))
        return train_df.iloc[tr_idx].reset_index(drop=True), train_df.iloc[va_idx].reset_index(drop=True)

    print("🧪 origin_id not available; using random split.")
    ss = ShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
    tr_idx, va_idx = next(ss.split(train_df))
    return train_df.iloc[tr_idx].reset_index(drop=True), train_df.iloc[va_idx].reset_index(drop=True)

def make_loaders(csv_path, images_root, img_size, batch_size, num_workers, val_frac=0.12, seed=42):
    print(f"📊 Loading data from: {csv_path}")
    print(f"🖼️  Images root: {images_root}")
    df = pd.read_csv(csv_path)

    if (df["split"].astype(str).str.lower() == "val").any():
        train_df = df[df["split"].astype(str).str.lower() == "train"].copy()
        val_df   = df[df["split"].astype(str).str.lower() == "val"].copy()
        test_df  = df[df["split"].astype(str).str.lower() == "test"].copy()
        print(f"📈 Total={len(df)} | Train={len(train_df)} | Val={len(val_df)} | Test={len(test_df)}")
        tr_sub, va_sub = train_df.reset_index(drop=True), val_df.reset_index(drop=True)
    else:
        train_df = df[df["split"].astype(str).str.lower() == "train"].copy()
        test_df  = df[df["split"].astype(str).str.lower() == "test"].copy()
        print(f"📈 Total={len(df)} | Train={len(train_df)} | Test={len(test_df)}")
        tr_sub, va_sub = split_train_val(train_df, val_frac, seed)

    ds_train = TeethImageDataset(csv_path, images_root, split="train", task="soft",
                                 img_size=img_size, aug=True, df_override=tr_sub)
    ds_val   = TeethImageDataset(csv_path, images_root, split="train", task="soft",
                                 img_size=img_size, aug=False, df_override=va_sub)
    ds_test  = TeethImageDataset(csv_path, images_root, split="test", task="soft",
                                 img_size=img_size, aug=False)

    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True, drop_last=True)
    dl_val   = DataLoader(ds_val, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=True)
    dl_test  = DataLoader(ds_test, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=True)
    print(f"🔄 Train batches: {len(dl_train)} | Val batches: {len(dl_val)} | Test batches: {len(dl_test)}")
    return dl_train, dl_val, dl_test

def train_one_epoch(model, loader, optimizer, scaler, device):
    model.train()
    loss_meter = AvgMeter()
    bce = nn.BCEWithLogitsLoss(reduction="none")
    t0 = time.time()
    for batch_idx, (imgs, y, w) in enumerate(loader):
        imgs = imgs.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        w = w.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast('cuda'):
            logits = model(imgs)
            loss_vec = bce(logits, y)
            loss = (loss_vec * w).mean()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loss_meter.update(loss.item(), imgs.size(0))

        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(loader):
            print(f"    Batch [{batch_idx+1:3d}/{len(loader):3d}] | Loss: {loss.item():.4f} | Avg: {loss_meter.avg:.4f}")

    print(f"✅ Train epoch in {time.time()-t0:.2f}s | Avg Loss: {loss_meter.avg:.4f}")
    return loss_meter.avg

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    bce = nn.BCEWithLogitsLoss(reduction="mean")
    loss_meter = AvgMeter()
    all_logits, all_targets = [], []
    t0 = time.time()
    for batch_idx, (imgs, y, _w) in enumerate(loader):
        imgs = imgs.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
        with autocast('cuda'):
            logits = model(imgs)
            loss = bce(logits, y)
        loss_meter.update(loss.item(), imgs.size(0))
        all_logits.append(logits); all_targets.append(y)
        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(loader):
            print(f"    Eval Batch [{batch_idx+1:3d}/{len(loader):3d}] | Loss: {loss.item():.4f}")

    logits = torch.cat(all_logits); targets = torch.cat(all_targets)
    metrics = binary_metrics_from_logits_single(logits, targets)
    metrics["loss"] = loss_meter.avg
    print(f"✅ Eval done in {time.time()-t0:.2f}s")
    return metrics

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-path", default="data/excel/data_dl_augmented.csv")
    ap.add_argument("--images-root", default="data/augmented")
    ap.add_argument("--model-name", default="convnext_tiny")
    ap.add_argument("--img-size", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--val-frac", type=float, default=0.12)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="weights/vision_soft_best.pt")
    args = ap.parse_args()

    print("=" * 80)
    print("🦷 SOFT LABELS TRAINING - Tooth Restoration (probabilistic)")
    print("=" * 80)
    for k, v in vars(args).items():
        print(f"{k:>18}: {v}")
    print("=" * 80)

    seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Device: {device}")
    if device == "cuda":
        print(f"🎮 GPU: {torch.cuda.get_device_name()} ({torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)")

    dl_train, dl_val, dl_test = make_loaders(
        args.csv_path, args.images_root, args.img_size,
        args.batch_size, args.num_workers, args.val_frac, args.seed
    )

    print(f"\n🏗️  Creating model: {args.model_name}")
    model = create_model(args.model_name, num_classes=1, pretrained=True)  # single logit
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Total parameters: {total_params:,} | Trainable: {trainable_params:,}")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler()

    best_val = float("inf")
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        print(f"\n🚀 Epoch {epoch}/{args.epochs}")
        train_loss = train_one_epoch(model, dl_train, optimizer, scaler, device)
        val_metrics = evaluate(model, dl_val, device)
        scheduler.step()

        print(f"\n📊 EPOCH {epoch:02d}/{args.epochs} SUMMARY:")
        print(f"    📉 Train Loss: {train_loss:.4f}")
        print(f"    📉 Val Loss:   {val_metrics['loss']:.4f}")
        print(f"    🎯 Val Acc@0.5:{val_metrics['acc']:.4f} | AUC: {val_metrics['auc']:.4f} "
              f"| Brier: {val_metrics['brier']:.4f} | MAE: {val_metrics['mae']:.4f}")

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            print(f"    🏆 NEW BEST (Val Loss: {best_val:.4f}) → {args.out}")
            save_checkpoint({"model": model.state_dict(),
                             "model_name": args.model_name,
                             "img_size": args.img_size}, args.out)
        print("-" * 80)

    print(f"\n🎉 Training completed in {(time.time()-t0)/60:.1f} min")

    # Final test evaluation
    print("\n🧪 Loading best model for TEST...")
    ckpt = torch.load(args.out, map_location=device)
    model.load_state_dict(ckpt["model"])
    test_metrics = evaluate(model, dl_test, device)

    print(f"\n🏁 FINAL TEST RESULTS:")
    print(f"    🎯 Acc@0.5: {test_metrics['acc']:.4f}")
    print(f"    📈 AUC:     {test_metrics['auc']:.4f}")
    print(f"    📉 Brier:   {test_metrics['brier']:.4f}")
    print(f"    📉 MAE:     {test_metrics['mae']:.4f}")
    print("=" * 80)

if __name__ == "__main__":
    main()
