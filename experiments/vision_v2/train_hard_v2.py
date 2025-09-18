# experiments/vision_v2/train_hard_v2.py
import os, sys, time, argparse
import numpy as np
import pandas as pd
from pathlib import Path
from math import ceil

import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.amp import autocast

# --- Make repo imports work regardless of CWD ---
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
# ------------------------------------------------

from sklearn.model_selection import StratifiedShuffleSplit, GroupShuffleSplit

from models.vision.datasets import TeethImageDataset
from models.vision.model_factory import create_model
from models.vision.utils import AvgMeter, seed_all, binary_metrics_from_logits_2class, save_checkpoint


def _split_train_val(train_df: pd.DataFrame, val_frac: float, seed: int, group_col: str | None):
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


def make_loaders(csv_path, images_root, img_size, batch_size, num_workers, val_frac, seed, group_col,
                 use_weighted_sampler=True):
    df = pd.read_csv(csv_path)
    has_val = (df["split"].astype(str).str.lower() == "val").any()
    if has_val:
        train_df = df[df["split"].str.lower() == "train"].copy()
        val_df   = df[df["split"].str.lower() == "val"].copy()
        test_df  = df[df["split"].str.lower() == "test"].copy()
        train_sub, val_sub = train_df.reset_index(drop=True), val_df.reset_index(drop=True)
    else:
        train_df = df[df["split"].str.lower() == "train"].copy()
        test_df  = df[df["split"].str.lower() == "test"].copy()
        train_sub, val_sub = _split_train_val(train_df, val_frac, seed, group_col)

    def dist(d):
        vc = d["y_majority"].astype(int).value_counts().sort_index()
        return {int(k): int(v) for k, v in vc.items()}

    print(f"📈 Total={len(df)} | Train={len(train_df)} | Val={len(val_sub)} | Test={len(test_df)}")
    print(f"🏷️ Train dist: {dist(train_sub)} | Val dist: {dist(val_sub)}")

    ds_train = TeethImageDataset(csv_path, images_root, split="train", task="hard",
                                 img_size=img_size, aug=True,  df_override=train_sub)
    ds_val   = TeethImageDataset(csv_path, images_root, split="train", task="hard",
                                 img_size=img_size, aug=False, df_override=val_sub)
    ds_test  = TeethImageDataset(csv_path, images_root, split="test",  task="hard",
                                 img_size=img_size, aug=False)

    # Class counts for CE weighting
    class_counts = np.bincount(train_sub["y_majority"].astype(int).values, minlength=2)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = class_weights[train_sub["y_majority"].astype(int).values]
    sampler = None
    if use_weighted_sampler:
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        print(f"⚖️ Sampler class weights: {class_weights}")
    else:
        print("⚖️ Sampler disabled → using shuffle=True")

    dl_train = DataLoader(ds_train, batch_size=batch_size,
                          sampler=sampler if sampler is not None else None,
                          shuffle=(sampler is None),
                          num_workers=num_workers, pin_memory=True, drop_last=True)
    dl_val   = DataLoader(ds_val,   batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=True)
    dl_test  = DataLoader(ds_test,  batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=True)
    return dl_train, dl_val, dl_test, train_sub, class_counts


def _init_head_bias(model: nn.Module, pos_rate: float):
    """Bias init so the head starts near the base rate (prevents 'all-negative' regime)."""
    prior = np.clip(pos_rate, 1e-4, 1 - 1e-4)
    prior_bias = float(np.log(prior / (1 - prior)))
    head = None
    # common head attribute names
    for attr in ["classifier", "head", "fc", "last_linear"]:
        mod = model
        try:
            mod = getattr(mod, attr)
            head = mod
            break
        except Exception:
            continue
    if head is not None and hasattr(head, "bias") and head.bias is not None:
        with torch.no_grad():
            b = head.bias.data
            if b.numel() == 2:
                b[1] = prior_bias / 2.0
                b[0] = -prior_bias / 2.0
            else:
                b[...] = prior_bias
        print(f"🧭 Head bias initialized to prior={prior:.4f} (logit={prior_bias:.3f})")
    else:
        print("ℹ️ Could not find a standard head bias to initialize; skipping.")


def train_one_epoch(model, loader, optimizer, scaler, device, criterion, epoch, total_epochs, lr, warmup_epochs):
    model.train()
    meter = AvgMeter()
    t0 = time.time()

    # Linear warmup
    if epoch <= warmup_epochs:
        warmup_factor = epoch / float(max(1, warmup_epochs))
        for pg in optimizer.param_groups:
            pg["lr"] = lr * warmup_factor

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
def evaluate(model, loader, device, criterion, name="Val"):
    model.eval()
    meter = AvgMeter()
    all_logits, all_targets = [], []
    t0 = time.time()
    for bi, (imgs, ys) in enumerate(loader):
        imgs = imgs.to(device, non_blocking=True); ys = ys.to(device, non_blocking=True)
        with autocast('cuda'):
            logits = model(imgs)
            loss = criterion(logits, ys)
        meter.update(loss.item(), imgs.size(0))
        all_logits.append(logits); all_targets.append(ys)
        if (bi+1) % 10 == 0 or (bi+1) == len(loader):
            print(f"    {name} [{bi+1:3d}/{len(loader):3d}] loss {loss.item():.4f}")
    logits = torch.cat(all_logits); targets = torch.cat(all_targets)
    metrics = binary_metrics_from_logits_2class(logits, targets)
    metrics["loss"] = meter.avg
    print(f"✅ {name} done in {time.time()-t0:.1f}s")
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-path", default="data/excel/data_dl_augmented.csv")
    ap.add_argument("--images-root", default="data/augmented")
    ap.add_argument("--model-name", default="tf_efficientnet_b4_ns")
    ap.add_argument("--stages", default="384,512", help="Progressive sizes, e.g. 384,512")
    ap.add_argument("--epochs", default="15,10", help="Epochs per stage, same length as --stages")
    ap.add_argument("--batch-sizes", default="16,8", help="Batch sizes per stage")
    ap.add_argument("--lrs", default="3e-4,1.5e-4", help="Learning rates per stage")
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--val-frac", type=float, default=0.12)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--label-smoothing", type=float, default=0.10)
    ap.add_argument("--group-col", default="origin_id")
    ap.add_argument("--save-dir", default="weights/v2")
    ap.add_argument("--run-name", default="hard_b4_prog")
    ap.add_argument("--no-weighted-sampler", action="store_true", help="Disable WeightedRandomSampler")
    ap.add_argument("--warmup-epochs", type=int, default=5, help="Linear LR warmup epochs per stage")
    args = ap.parse_args()

    # Parse lists
    sizes  = [int(x) for x in args.stages.split(",")]
    epochs = [int(x) for x in args.epochs.split(",")]
    bss    = [int(x) for x in args.batch_sizes.split(",")]
    lrs    = [float(x) for x in args.lrs.split(",")]
    assert len(sizes) == len(epochs) == len(bss) == len(lrs), "stages/epochs/batch-sizes/lrs must have same length"
    seeds  = [int(s) for s in args.seeds.split(",")]

    os.makedirs(args.save_dir, exist_ok=True)

    print("="*92)
    print("🦷 HARD v2 PROGRESSIVE TRAINING (multi-seed supported)")
    for k,v in vars(args).items(): print(f"{k:>22}: {v}")
    print("="*92)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Device: {device}")
    if device == "cuda":
        print(f"🎮 GPU: {torch.cuda.get_device_name()} "
              f"({torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)")

    for seed in seeds:
        print(f"\n====================  SEED {seed}  ====================")
        seed_all(seed)
        model = create_model(args.model_name, num_classes=2, pretrained=True)
        model.to(device)

        prev_ckpt_path = None
        for stage_i, (img_size, E, bs, lr) in enumerate(zip(sizes, epochs, bss, lrs), start=1):
            print(f"\n---- Stage {stage_i}/{len(sizes)}  |  img_size={img_size}  epochs={E}  bs={bs}  lr={lr} ----")

            dl_train, dl_val, dl_test, train_sub, class_counts = make_loaders(
                args.csv_path, args.images_root, img_size,
                bs, args.num_workers, args.val_frac, seed, args.group_col,
                use_weighted_sampler=not args.no_weighted_sampler
            )

            # resume from previous stage best
            if prev_ckpt_path and Path(prev_ckpt_path).exists():
                print(f"↪️  Resuming from previous stage checkpoint: {prev_ckpt_path}")
                ckpt_prev = torch.load(prev_ckpt_path, map_location=device)
                model.load_state_dict(ckpt_prev["model"], strict=True)

            # Head bias init (once at start of stage 1 only)
            if stage_i == 1:
                pos_rate = float(train_sub["y_majority"].astype(int).mean())
                _init_head_bias(model, pos_rate)

            # Class-balanced CE weights
            neg, pos = int(class_counts[0]), int(class_counts[1])
            total = float(max(1, neg + pos))
            w0 = total / (2.0 * max(neg, 1))
            w1 = total / (2.0 * max(pos, 1))
            ce_weights = torch.tensor([w0, w1], dtype=torch.float, device=device)

            criterion = nn.CrossEntropyLoss(weight=ce_weights, label_smoothing=args.label_smoothing)
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=E)
            scaler = torch.cuda.amp.GradScaler()

            best_val = float("inf")
            stage_tag = f"{args.run_name}_seed{seed}_stage{stage_i}_{img_size}"
            save_path = os.path.join(args.save_dir, f"{stage_tag}.pt")

            warmup_epochs = min(args.warmup_epochs, E)

            for epoch in range(1, E + 1):
                train_loss = train_one_epoch(model, dl_train, optimizer, scaler, device, criterion,
                                             epoch, E, lr, warmup_epochs)
                val_metrics = evaluate(model, dl_val, device, criterion, "Val")
                scheduler.step()

                print(f"\n📊 STAGE {stage_i} EPOCH {epoch:02d}/{E}")
                print(f"    train_loss {train_loss:.4f}")
                print(f"    val_loss   {val_metrics['loss']:.4f}  "
                      f"acc {val_metrics['acc']:.4f}  f1 {val_metrics['f1']:.4f}  "
                      f"prec {val_metrics['prec']:.4f}  rec {val_metrics['rec']:.4f}  auc {val_metrics['auc']:.4f}")

                if val_metrics["loss"] < best_val:
                    best_val = val_metrics["loss"]
                    print(f"    🏆 New best (val_loss={best_val:.4f}) → {save_path}")
                    save_checkpoint({"model": model.state_dict(),
                                     "model_name": args.model_name,
                                     "img_size": img_size,
                                     "seed": seed,
                                     "stage": stage_i}, save_path)
                print("-"*80)

            prev_ckpt_path = save_path

        print(f"✅ Finished SEED {seed}. Best final ckpt: {prev_ckpt_path}")


if __name__ == "__main__":
    main()
