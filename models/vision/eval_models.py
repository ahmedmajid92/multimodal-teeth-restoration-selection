# models/vision/eval_models.py
import argparse, os, sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             roc_auc_score, confusion_matrix, classification_report,
                             brier_score_loss, log_loss)

# Add project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from models.vision.datasets import TeethImageDataset
from models.vision.model_factory import create_model

def _dl(csv_path, images_root, task, img_size=384, batch_size=32, num_workers=4):
    ds = TeethImageDataset(csv_path, images_root, split="test", task=task,
                           img_size=img_size, aug=False)
    return ds, DataLoader(ds, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=True)

@torch.no_grad()
def eval_hard(ckpt_path, csv_path, images_root, batch_size, num_workers, out_csv=None):
    print("🔍 Starting hard model evaluation...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Device: {device}")
    
    print(f"📥 Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    model_name = ckpt.get("model_name", "tf_efficientnet_b3_ns")
    img_size   = ckpt.get("img_size", 384)
    
    print(f"🏗️  Model: {model_name}, Image size: {img_size}")

    print(f"📊 Loading test data...")
    ds, dl = _dl(csv_path, images_root, "hard", img_size, batch_size, num_workers)
    print(f"📈 Test samples: {len(ds)}, Batches: {len(dl)}")
    
    model = create_model(model_name, num_classes=2, pretrained=False)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    
    print(f"🚀 Running inference...")

    all_probs, all_preds, all_targets, all_names = [], [], [], []
    for batch_idx, (imgs, y) in enumerate(dl):
        imgs = imgs.to(device, non_blocking=True)
        logits = model(imgs)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()  # P(indirect)
        preds = (probs >= 0.5).astype(int)
        all_probs.append(probs)
        all_preds.append(preds)
        all_targets.append(y.numpy())
        # keep filenames
        idx0 = len(all_names)
        batch_names = ds.df.iloc[idx0:idx0+len(y)]["image_name"].tolist()  # safe because shuffle=False
        all_names.extend(batch_names)
        
        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(dl):
            print(f"    Processed batch [{batch_idx+1:3d}/{len(dl):3d}]")

    probs  = np.concatenate(all_probs)
    preds  = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)

    acc  = accuracy_score(y_true, preds)
    f1   = f1_score(y_true, preds)
    prec = precision_score(y_true, preds, zero_division=0)
    rec  = recall_score(y_true, preds, zero_division=0)
    try: auc = roc_auc_score(y_true, probs)
    except Exception: auc = float("nan")

    cm = confusion_matrix(y_true, preds)
    report = classification_report(y_true, preds, digits=4, zero_division=0)

    print("\n" + "="*60)
    print("🦷 HARD MODEL EVALUATION RESULTS (2-class)")
    print("="*60)
    print(f"🎯 Accuracy:  {acc:.4f}")
    print(f"🎯 F1-Score:  {f1:.4f}")
    print(f"🎯 Precision: {prec:.4f}")
    print(f"🎯 Recall:    {rec:.4f}")
    print(f"📈 AUC:       {auc:.4f}")
    print(f"\n📊 Confusion Matrix (TN FP / FN TP):")
    print(cm)
    print(f"\n📋 Classification Report:")
    print(report)

    if out_csv:
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        out = pd.DataFrame({
            "image_name": ds.df["image_name"].tolist(),
            "y_true": y_true,
            "p_indirect_pred": probs,
            "y_pred": preds
        })
        out.to_csv(out_csv, index=False)
        print(f"💾 Saved predictions to: {out_csv}")

def _safe_auc(y_true, p):
    try: return roc_auc_score(y_true, p)
    except Exception: return float("nan")

@torch.no_grad()
def eval_soft(ckpt_path, csv_path, images_root, batch_size, num_workers, out_csv=None):
    print("🔍 Starting soft model evaluation...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Device: {device}")
    
    print(f"📥 Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    model_name = ckpt.get("model_name", "convnext_tiny")
    img_size   = ckpt.get("img_size", 384)
    
    print(f"🏗️  Model: {model_name}, Image size: {img_size}")

    print(f"📊 Loading test data...")
    ds, dl = _dl(csv_path, images_root, "soft", img_size, batch_size, num_workers)
    print(f"📈 Test samples: {len(ds)}, Batches: {len(dl)}")
    
    model = create_model(model_name, num_classes=1, pretrained=False)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    
    print(f"🚀 Running inference...")

    probs_list, ysoft_list, names = [], [], []
    for batch_idx, (imgs, y, _w) in enumerate(dl):
        imgs = imgs.to(device, non_blocking=True)
        logits = model(imgs)
        probs = torch.sigmoid(logits).cpu().numpy().ravel()
        probs_list.append(probs)
        ysoft_list.append(y.numpy().ravel())
        idx0 = len(names)
        batch_names = ds.df.iloc[idx0:idx0+len(y)]["image_name"].tolist()
        names.extend(batch_names)
        
        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(dl):
            print(f"    Processed batch [{batch_idx+1:3d}/{len(dl):3d}]")

    p = np.concatenate(probs_list)
    y = np.concatenate(ysoft_list)

    # "hard" version of soft labels for AUC/Acc
    y_hard = (y >= 0.5).astype(int)
    acc = accuracy_score(y_hard, (p >= 0.5).astype(int))
    auc = _safe_auc(y_hard, p)
    brier = brier_score_loss(y, p)
    try:
        ll = log_loss(y, np.c_[1-p, p], labels=[0,1])
    except Exception:
        ll = float("nan")
    mae = np.mean(np.abs(p - y))

    print("\n" + "="*60)
    print("🦷 SOFT MODEL EVALUATION RESULTS (probabilistic)")
    print("="*60)
    print(f"🎯 Accuracy@0.5: {acc:.4f}")
    print(f"📈 AUC:          {auc:.4f}")
    print(f"📊 Brier Score:  {brier:.4f}")
    print(f"📊 Log Loss:     {ll:.4f}")
    print(f"📊 MAE:          {mae:.4f}")

    if out_csv:
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        out = pd.DataFrame({
            "image_name": names if len(names)==len(p) else ds.df["image_name"].tolist(),
            "p_indirect_true": y,
            "p_indirect_pred": p
        })
        out.to_csv(out_csv, index=False)
        print(f"💾 Saved predictions to: {out_csv}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-path", default="data/excel/data_dl_augmented.csv")
    ap.add_argument("--images-root", default="data/augmented")
    ap.add_argument("--which", choices=["hard", "soft", "both"], required=True)
    ap.add_argument("--hard-ckpt", default="weights/vision_hard_best.pt")
    ap.add_argument("--soft-ckpt", default="weights/vision_soft_best.pt")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--out-csv", default=None,
                    help="Optional path to save predictions (for 'both', will append _hard/_soft).")
    args = ap.parse_args()

    print("="*80)
    print("🦷 TOOTH RESTORATION AI - MODEL EVALUATION")
    print("="*80)
    print(f"📊 CSV Path: {args.csv_path}")
    print(f"🖼️  Images Root: {args.images_root}")
    print(f"🎯 Evaluating: {args.which}")
    print(f"📦 Batch Size: {args.batch_size}")
    print("="*80)

    if args.which in ("hard", "both"):
        out_h = (args.out_csv.replace(".csv", "_hard.csv") if args.out_csv and args.out_csv.endswith(".csv")
                 else (args.out_csv+"_hard.csv" if args.out_csv else None))
        eval_hard(args.hard_ckpt, args.csv_path, args.images_root, args.batch_size, args.num_workers, out_h)

    if args.which in ("soft", "both"):
        out_s = (args.out_csv.replace(".csv", "_soft.csv") if args.out_csv and args.out_csv.endswith(".csv")
                 else (args.out_csv+"_soft.csv" if args.out_csv else None))
        eval_soft(args.soft_ckpt, args.csv_path, args.images_root, args.batch_size, args.num_workers, out_s)

    print("\n🎉 Evaluation completed!")

if __name__ == "__main__":
    main()
