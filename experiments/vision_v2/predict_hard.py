# experiments/vision_v2/predict_hard.py
import os, sys, argparse, glob
import numpy as np
import pandas as pd
from pathlib import Path

import torch
from torch.amp import autocast
from torch.utils.data import DataLoader

# repo path fix
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.vision.datasets import TeethImageDataset
from models.vision.model_factory import create_model
from models.vision.utils import seed_all

def _load_models(ckpt_paths, device):
    models, sizes = [], []
    for p in ckpt_paths:
        ck = torch.load(p, map_location=device)
        m = create_model(ck.get("model_name", "tf_efficientnet_b4_ns"), num_classes=2, pretrained=False)
        m.load_state_dict(ck["model"], strict=True)
        m.to(device); m.eval()
        models.append(m); sizes.append(int(ck.get("img_size", 512)))
    assert len(set(sizes)) == 1, "Different img_size across ckpts"
    return models, sizes[0]

@torch.no_grad()
def _probs(models, loader, device, tta=False):
    n = len(models); out = []
    for imgs, ys in loader:
        imgs = imgs.to(device, non_blocking=True)
        with autocast('cuda'):
            logits_sum = None
            for m in models:
                o = m(imgs)
                if tta:
                    o = (o + m(torch.flip(imgs, dims=[3]))) / 2
                logits_sum = o if logits_sum is None else logits_sum + o
            probs = torch.softmax(logits_sum / n, dim=1)[:,1].detach().cpu().numpy()
        out.append(probs)
    return np.concatenate(out, axis=0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-path", default="data/excel/data_dl_augmented.csv")
    ap.add_argument("--images-root", default="data/augmented")
    ap.add_argument("--ckpts", default="weights/v2/hard_b4_prog_seed*_stage2_512.pt")
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-frac", type=float, default=0.12)
    ap.add_argument("--group-col", default="origin_id")
    ap.add_argument("--out-csv", default="out/preds/vis_hard_preds.csv")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_all(args.seed)
    os.makedirs(Path(args.out_csv).parent, exist_ok=True)

    ckpt_paths = sorted(glob.glob(args.ckpts))
    assert len(ckpt_paths) >= 1, "No checkpoints found"
    models, img_size = _load_models(ckpt_paths, device)

    df = pd.read_csv(args.csv_path)
    has_val = (df["split"].astype(str).str.lower()=="val").any()
    if not has_val:
        # Create a deterministic grouped-val split identical across scripts
        from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit
        tr_df = df[df["split"].str.lower()=="train"].reset_index(drop=True)
        if args.group_col in tr_df.columns:
            gss = GroupShuffleSplit(n_splits=1, test_size=args.val_frac, random_state=args.seed)
            tr_idx, va_idx = next(gss.split(tr_df, groups=tr_df[args.group_col].values))
            df.loc[tr_df.index[va_idx], "split"] = "val"

    # Build loaders
    val_df  = df[df["split"].str.lower()=="val"].reset_index(drop=True)
    test_df = df[df["split"].str.lower()=="test"].reset_index(drop=True)

    ds_val  = TeethImageDataset(args.csv_path, args.images_root, split="train", task="hard",
                                img_size=img_size, aug=False, df_override=val_df)
    ds_test = TeethImageDataset(args.csv_path, args.images_root, split="test",  task="hard",
                                img_size=img_size, aug=False)

    dl_val  = DataLoader(ds_val,  batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    dl_test = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # Predict
    p_val  = _probs(models, dl_val,  device, tta=args.tta)
    p_test = _probs(models, dl_test, device, tta=args.tta)

    out_val = val_df[["image_name","origin_id","split","y_majority"]].copy()
    out_val["prob_vis_hard"] = p_val
    out_test = test_df[["image_name","origin_id","split","y_majority"]].copy()
    out_test["prob_vis_hard"] = p_test

    out = pd.concat([out_val, out_test], ignore_index=True)
    out.to_csv(args.out_csv, index=False)
    print(f"✅ Saved: {args.out_csv}  (rows={len(out)})")

if __name__ == "__main__":
    main()
