# models/vision/train_hard.py
import argparse, os, time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
import torch
from torch import autocast, nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.ops import misc as misc_ops
from models.vision.datasets import TeethImageDataset
from models.vision.model_factory import create_model
from models.vision.utils import AvgMeter, seed_all, binary_metrics_from_logits_2class, save_checkpoint

def make_loaders(csv_path, images_root, img_size, batch_size, num_workers, val_frac=0.12, seed=42):
    print(f"📊 Loading data from: {csv_path}")
    print(f"🖼️  Images root: {images_root}")
    
    df = pd.read_csv(csv_path)
    train_df = df[df["split"] == "train"].copy()
    test_df  = df[df["split"] == "test"].copy()
    
    print(f"📈 Total samples: {len(df)} | Train: {len(train_df)} | Test: {len(test_df)}")

    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
    train_idx, val_idx = next(sss.split(train_df, train_df["y_majority"]))
    train_sub = train_df.iloc[train_idx].reset_index(drop=True)
    val_sub   = train_df.iloc[val_idx].reset_index(drop=True)
    
    print(f"📊 Train split: {len(train_sub)} | Validation split: {len(val_sub)}")
    
    # Print class distribution
    train_counts = train_sub["y_majority"].value_counts().sort_index()
    val_counts = val_sub["y_majority"].value_counts().sort_index()
    print(f"🏷️  Train class distribution: {dict(train_counts)}")
    print(f"🏷️  Val class distribution: {dict(val_counts)}")

    ds_train = TeethImageDataset(csv_path, images_root, split="train", task="hard",
                                 img_size=img_size, aug=True, df_override=train_sub)
    ds_val   = TeethImageDataset(csv_path, images_root, split="train", task="hard",
                                 img_size=img_size, aug=False, df_override=val_sub)
    ds_test  = TeethImageDataset(csv_path, images_root, split="test", task="hard",
                                 img_size=img_size, aug=False)

    # Handle imbalance with weighted sampling
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
    
    print(f"🔄 Train batches: {len(dl_train)} | Val batches: {len(dl_val)} | Test batches: {len(dl_test)}")
    return dl_train, dl_val, dl_test

def train_one_epoch(model, loader, optimizer, scaler, device, criterion, epoch, total_epochs):
    model.train()
    loss_meter = AvgMeter()
    
    print(f"\n🚀 Training Epoch {epoch}/{total_epochs}")
    start_time = time.time()
    
    for batch_idx, (imgs, ys) in enumerate(loader):
        batch_start = time.time()
        
        imgs = imgs.to(device, non_blocking=True)
        ys = ys.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast('cuda'):
            logits = model(imgs)
            loss = criterion(logits, ys)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loss_meter.update(loss.item(), imgs.size(0))
        
        batch_time = time.time() - batch_start
        
        # Print progress every 10 batches or at the end
        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(loader):
            print(f"    Batch [{batch_idx+1:3d}/{len(loader):3d}] | "
                  f"Loss: {loss.item():.4f} | Avg Loss: {loss_meter.avg:.4f} | "
                  f"Time: {batch_time:.2f}s | LR: {optimizer.param_groups[0]['lr']:.2e}")
    
    epoch_time = time.time() - start_time
    print(f"✅ Epoch {epoch} completed in {epoch_time:.2f}s | Avg Loss: {loss_meter.avg:.4f}")
    return loss_meter.avg

@torch.no_grad()
def evaluate(model, loader, device, criterion, split_name="Val"):
    model.eval()
    loss_meter = AvgMeter()
    all_logits, all_targets = [], []
    
    print(f"\n🔍 Evaluating on {split_name} set...")
    start_time = time.time()
    
    for batch_idx, (imgs, ys) in enumerate(loader):
        imgs = imgs.to(device, non_blocking=True)
        ys = ys.to(device, non_blocking=True)
        with autocast('cuda'):
            logits = model(imgs)
            loss = criterion(logits, ys)
        loss_meter.update(loss.item(), imgs.size(0))
        all_logits.append(logits)
        all_targets.append(ys)
        
        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(loader):
            print(f"    {split_name} Batch [{batch_idx+1:3d}/{len(loader):3d}] | Loss: {loss.item():.4f}")
    
    logits = torch.cat(all_logits)
    targets = torch.cat(all_targets)
    metrics = binary_metrics_from_logits_2class(logits, targets)
    metrics["loss"] = loss_meter.avg
    
    eval_time = time.time() - start_time
    print(f"✅ {split_name} evaluation completed in {eval_time:.2f}s")
    return metrics

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv-path", default="data/excel/data_dl_augmented.csv")
    p.add_argument("--images-root", default="data/augmented")
    p.add_argument("--model-name", default="tf_efficientnet_b3_ns")
    p.add_argument("--img-size", type=int, default=384)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=24)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--val-frac", type=float, default=0.12)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="weights/vision_hard_best.pt")
    args = p.parse_args()

    print("=" * 80)
    print("🦷 HARD LABELS TRAINING - Tooth Restoration Classification")
    print("=" * 80)
    print(f"🎯 Model: {args.model_name}")
    print(f"📏 Image size: {args.img_size}x{args.img_size}")
    print(f"📊 Batch size: {args.batch_size}")
    print(f"🔄 Epochs: {args.epochs}")
    print(f"📈 Learning rate: {args.lr}")
    print(f"🌱 Seed: {args.seed}")
    print(f"💾 Output: {args.out}")
    print("=" * 80)

    seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Device: {device}")
    
    if device == "cuda":
        print(f"🎮 GPU: {torch.cuda.get_device_name()}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    dl_train, dl_val, dl_test = make_loaders(
        args.csv_path, args.images_root, args.img_size,
        args.batch_size, args.num_workers, args.val_frac, args.seed
    )

    print(f"\n🏗️  Creating model: {args.model_name}")
    model = create_model(args.model_name, num_classes=2, pretrained=True)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Total parameters: {total_params:,}")
    print(f"📊 Trainable parameters: {trainable_params:,}")
    
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler()

    print(f"\n🏃 Starting training for {args.epochs} epochs...")
    print("=" * 80)
    
    best_val = float("inf")
    training_start = time.time()
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        
        train_loss = train_one_epoch(model, dl_train, optimizer, scaler, device, criterion, epoch, args.epochs)
        val_metrics = evaluate(model, dl_val, device, criterion, "Validation")
        scheduler.step()
        
        epoch_time = time.time() - epoch_start
        
        print(f"\n📊 EPOCH {epoch:02d}/{args.epochs} SUMMARY:")
        print(f"    ⏱️  Time: {epoch_time:.2f}s")
        print(f"    📉 Train Loss: {train_loss:.4f}")
        print(f"    📉 Val Loss: {val_metrics['loss']:.4f}")
        print(f"    🎯 Val Accuracy: {val_metrics['acc']:.4f}")
        print(f"    📈 Val AUC: {val_metrics['auc']:.4f}")
        print(f"    🎯 Val F1: {val_metrics['f1']:.4f}")
        print(f"    🎯 Val Precision: {val_metrics['prec']:.4f}")
        print(f"    🎯 Val Recall: {val_metrics['rec']:.4f}")
        print(f"    📈 Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            print(f"    🏆 NEW BEST MODEL! (Val Loss: {best_val:.4f})")
            save_checkpoint({"model": model.state_dict(),
                             "model_name": args.model_name,
                             "img_size": args.img_size}, args.out)
        
        print("-" * 80)

    total_training_time = time.time() - training_start
    print(f"\n🎉 Training completed in {total_training_time:.2f}s ({total_training_time/60:.1f} minutes)")

    # Final test eval
    print("\n🧪 Loading best model for final test evaluation...")
    ckpt = torch.load(args.out, map_location=device)
    model.load_state_dict(ckpt["model"])
    test_metrics = evaluate(model, dl_test, device, criterion, "Test")
    
    print(f"\n🏁 FINAL TEST RESULTS:")
    print(f"    🎯 Accuracy: {test_metrics['acc']:.4f}")
    print(f"    🎯 F1-Score: {test_metrics['f1']:.4f}")
    print(f"    🎯 Precision: {test_metrics['prec']:.4f}")
    print(f"    🎯 Recall: {test_metrics['rec']:.4f}")
    print(f"    📈 AUC: {test_metrics['auc']:.4f}")
    print("=" * 80)

if __name__ == "__main__":
    main()
