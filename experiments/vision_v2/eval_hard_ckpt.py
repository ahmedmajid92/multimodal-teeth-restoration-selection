# experiments/vision_v2/eval_hard_ckpt.py
import os, sys, argparse, time
import numpy as np
import pandas as pd
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.amp import autocast

# repo path fix
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit

from models.vision.datasets import TeethImageDataset
from models.vision.model_factory import create_model
from models.vision.utils import seed_all

@torch.no_grad()
def _probs_single(model, loader, device, tta=False):
    model.eval()
    all = []
    for imgs, ys in loader:
        imgs = imgs.to(device, non_blocking=True)
        with autocast('cuda'):
            logits = model(imgs)
            if tta:
                logits = (logits + model(torch.flip(imgs, dims=[3]))) / 2
            p = torch.softmax(logits, dim=1)[:,1]
        all.append(p.detach().cpu())
    return torch.cat(all).numpy()

def _grid_best_threshold(probs, y, lo=0.05, hi=0.95, step=0.005):
    ths = np.arange(lo, hi + 1e-9, step)
    best = (0.5, -1.0)
    for t in ths:
        f1 = f1_score(y, (probs >= t).astype(int))
        if f1 > best[1]:
            best = (t, f1)
    return best

def _metrics(probs, y, thr):
    yp = (probs >= thr).astype(int)
    return dict(
        acc=accuracy_score(y, yp),
        f1=f1_score(y, yp),
        prec=precision_score(y, yp),
        rec=recall_score(y, yp),
        auc=roc_auc_score(y, probs),
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-path", default="data/excel/data_dl_augmented.csv")
    ap.add_argument("--images-root", default="data/augmented")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--val-frac", type=float, default=0.12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--group-col", default="origin_id")
    ap.add_argument("--tta", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_all(args.seed)

    # Load ckpt
    ck = torch.load(args.ckpt, map_location=device)
    model = create_model(ck.get("model_name", "tf_efficientnet_b4_ns"), num_classes=2, pretrained=False)
    model.load_state_dict(ck["model"], strict=True)
    model.to(device)
    img_size = int(ck.get("img_size", 512))
    print(f"Loaded {os.path.basename(args.ckpt)} (img_size={img_size})")

    df = pd.read_csv(args.csv_path)
    has_val = (df["split"].astype(str).str.lower()=="val").any()
    if not has_val:
        tr_df = df[df["split"].str.lower()=="train"].reset_index(drop=True)
        if args.group_col in tr_df.columns:
            gss = GroupShuffleSplit(n_splits=1, test_size=args.val_frac, random_state=args.seed)
            tr_idx, va_idx = next(gss.split(tr_df, groups=tr_df[args.group_col].values))
            val_df = tr_df.iloc[va_idx].reset_index(drop=True)
        else:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=args.val_frac, random_state=args.seed)
            tr_idx, va_idx = next(sss.split(tr_df, tr_df["y_majority"]))
            val_df = tr_df.iloc[va_idx].reset_index(drop=True)
    else:
        val_df = df[df["split"].str.lower()=="val"].reset_index(drop=True)
    test_df = df[df["split"].str.lower()=="test"].reset_index(drop=True)

    ds_val  = TeethImageDataset(args.csv_path, args.images_root, split="train", task="hard",
                                img_size=img_size, aug=False, df_override=val_df)
    ds_test = TeethImageDataset(args.csv_path, args.images_root, split="test", task="hard",
                                img_size=img_size, aug=False)
    dl_val  = DataLoader(ds_val,  batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    dl_test = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    y_val  = val_df["y_majority"].astype(int).values
    y_test = test_df["y_majority"].astype(int).values

    print("🔮 Predicting on VAL…")
    p_val = _probs_single(model, dl_val, device, tta=args.tta)
    print("🔮 Predicting on TEST…")
    p_test= _probs_single(model, dl_test, device, tta=args.tta)

    t_star, f1_star = _grid_best_threshold(p_val, y_val)
    mv = _metrics(p_val,  y_val,  t_star)
    mt = _metrics(p_test, y_test, t_star)

    print(f"\n🎯 Tuned threshold on Val: t*={t_star:.3f} (F1={f1_star:.4f})")
    print(f"VAL  @t*: acc {mv['acc']:.4f}  f1 {mv['f1']:.4f}  prec {mv['prec']:.4f}  rec {mv['rec']:.4f}  auc {mv['auc']:.4f}")
    print(f"TEST @t*: acc {mt['acc']:.4f}  f1 {mt['f1']:.4f}  prec {mt['prec']:.4f}  rec {mt['rec']:.4f}  auc {mt['auc']:.4f}")

if __name__ == "__main__":
    main()
