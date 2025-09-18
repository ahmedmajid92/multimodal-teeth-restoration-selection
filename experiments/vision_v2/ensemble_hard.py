# experiments/vision_v2/ensemble_hard.py
import os, sys, argparse, glob, numpy as np, pandas as pd
from pathlib import Path
from contextlib import nullcontext

import torch
from torch.amp import autocast
from torch.utils.data import DataLoader

# repo path fix
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit

from models.vision.datasets import TeethImageDataset
from models.vision.model_factory import create_model
from models.vision.utils import seed_all

def _parse_ckpts(arg: str):
    if any(ch in arg for ch in "*?[]"):
        return sorted(glob.glob(arg))
    if "," in arg:
        return [p.strip() for p in arg.split(",") if p.strip()]
    return [arg]

def _load_models(ckpt_paths, device):
    models, img_sizes, names = [], [], []
    for p in ckpt_paths:
        print(f"Loading: {p}")
        ck = torch.load(p, map_location=device)
        mname = ck.get("model_name", "tf_efficientnet_b4_ns")
        m = create_model(mname, num_classes=2, pretrained=False)
        m.load_state_dict(ck["model"], strict=True)
        m.to(device); m.eval()
        models.append(m); img_sizes.append(int(ck.get("img_size", 512))); names.append(os.path.basename(p))
    assert len(set(img_sizes)) == 1, "All checkpoints must share the same img_size"
    return models, img_sizes[0], names

def _build_val_test_frames(csv_path, val_frac, seed, group_col):
    df = pd.read_csv(csv_path)
    has_val = (df["split"].astype(str).str.lower() == "val").any()
    if has_val:
        train_df = df[df["split"].str.lower()=="train"].reset_index(drop=True)
        val_df   = df[df["split"].str.lower()=="val"].reset_index(drop=True)
    else:
        train_df = df[df["split"].str.lower()=="train"].reset_index(drop=True)
        if group_col and (group_col in train_df.columns):
            gss = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
            tr_idx, va_idx = next(gss.split(train_df, groups=train_df[group_col].values))
            val_df = train_df.iloc[va_idx].reset_index(drop=True)
        else:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
            tr_idx, va_idx = next(sss.split(train_df, train_df["y_majority"]))
            val_df = train_df.iloc[va_idx].reset_index(drop=True)
    test_df = df[df["split"].str.lower()=="test"].reset_index(drop=True)
    print(f"📊 Data splits: Train={len(train_df)} | Val={len(val_df)} | Test={len(test_df)}")
    def _dist(d):
        vc = d["y_majority"].astype(int).value_counts().sort_index()
        neg, pos = int(vc.get(0,0)), int(vc.get(1,0))
        return f"0:{neg} 1:{pos} (pos_rate={pos/max(len(d),1):.3f})"
    print(f"🏷️ Val dist:  {_dist(val_df)}")
    print(f"🏷️ Test dist: {_dist(test_df)}")
    return val_df, test_df

@torch.no_grad()
def _predict_probs(models, loader, device, tta=True, amp=True, names=None):
    n_models = len(models)
    probs_all, nan_idx = [], []
    seen = 0
    for bi, (imgs, ys) in enumerate(loader):
        imgs = imgs.to(device, non_blocking=True)
        ctx = autocast('cuda') if (amp and device == "cuda") else nullcontext()
        with ctx:
            logits_sum = None
            for m in models:
                out = m(imgs)
                if tta:
                    out = (out + m(torch.flip(imgs, dims=[3]))) / 2  # hflip TTA
                logits_sum = out if logits_sum is None else logits_sum + out
            logits_mean = logits_sum / n_models
            logits_mean = torch.nan_to_num(logits_mean, nan=0.0, posinf=30.0, neginf=-30.0)
            probs = torch.softmax(logits_mean, dim=1)[:, 1]
        probs_np = probs.detach().cpu().numpy()
        if np.isnan(probs_np).any():
            idxs = np.where(np.isnan(probs_np))[0]
            for j in idxs:
                global_idx = seen + j
                fname = names[global_idx] if (names is not None and global_idx < len(names)) else f"idx{global_idx}"
                nan_idx.append((global_idx, fname))
        probs_all.append(probs_np)
        if (bi+1) % 10 == 0 or (bi+1) == len(loader):
            print(f"    Processed {bi+1}/{len(loader)} batches")
        seen += imgs.size(0)
    return np.concatenate(probs_all, axis=0), nan_idx

def _grid_best_threshold(probs, y, metric="f1", lo=0.05, hi=0.95, step=0.005):
    ths = np.arange(lo, hi + 1e-9, step)
    best_t, best_v = 0.5, -1.0
    for t in ths:
        yp = (probs >= t).astype(int)
        if metric == "acc":
            v = accuracy_score(y, yp)
        else:  # default F1
            v = f1_score(y, yp)
        if v > best_v:
            best_t, best_v = t, v
    return best_t, best_v

def _metrics_from_probs(probs, y, thr=0.5):
    yp = (probs >= thr).astype(int)
    return dict(
        acc = accuracy_score(y, yp),
        f1  = f1_score(y, yp),
        prec= precision_score(y, yp),
        rec = recall_score(y, yp),
        auc = roc_auc_score(y, probs),
    )

def _per_model_aucs(models, loader, y_true, device, tta=True, amp=True):
    out = []
    for i, m in enumerate(models):
        allp = []
        with torch.no_grad():
            for imgs, _ in loader:
                imgs = imgs.to(device, non_blocking=True)
                ctx = autocast('cuda') if (amp and device == "cuda") else nullcontext()
                with ctx:
                    o = m(imgs)
                    if tta: o = (o + m(torch.flip(imgs, dims=[3]))) / 2
                    p = torch.softmax(o, dim=1)[:,1]
                allp.append(p.detach().cpu())
        p = torch.cat(allp).numpy()
        out.append(dict(idx=i+1, auc=roc_auc_score(y_true, p)))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-path", default="data/excel/data_dl_augmented.csv")
    ap.add_argument("--images-root", default="data/augmented")
    ap.add_argument("--ckpts", default="weights/v2/hard_b4_prog_seed*_stage2_512.pt",
                    help="Glob OR comma-separated list of ckpt paths")
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--tta", dest="tta", action="store_true")
    ap.add_argument("--no-tta", dest="tta", action="store_false")
    ap.set_defaults(tta=True)
    ap.add_argument("--amp", dest="amp", action="store_true")
    ap.add_argument("--no-amp", dest="amp", action="store_false")
    ap.set_defaults(amp=True)
    ap.add_argument("--opt-metric", choices=["f1","acc"], default="f1",
                    help="Optimize threshold on Val for this metric")
    ap.add_argument("--val-frac", type=float, default=0.12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--group-col", default="origin_id")
    ap.add_argument("--per-model", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_all(args.seed)

    ckpt_paths = _parse_ckpts(args.ckpts)
    assert len(ckpt_paths) >= 1, "No checkpoints found."
    models, img_size, names = _load_models(ckpt_paths, device)
    print(f"Loaded {len(models)} models @img_size={img_size}:\n  - " + "\n  - ".join(names))

    # Val/Test frames
    val_df, test_df = _build_val_test_frames(args.csv_path, args.val_frac, args.seed, args.group_col)

    # Loaders
    ds_val  = TeethImageDataset(args.csv_path, args.images_root, split="train", task="hard",
                                img_size=img_size, aug=False, df_override=val_df)
    ds_test = TeethImageDataset(args.csv_path, args.images_root, split="test",  task="hard",
                                img_size=img_size, aug=False)
    dl_val  = DataLoader(ds_val,  batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    dl_test = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    y_val  = val_df["y_majority"].astype(int).values
    y_test = test_df["y_majority"].astype(int).values
    names_val  = val_df["image_name"].tolist()
    names_test = test_df["image_name"].tolist()

    # Optional per-model AUCs (TEST) to spot a weak seed
    if args.per_model and len(models) > 1:
        per = _per_model_aucs(models, dl_test, y_test, device, tta=args.tta, amp=args.amp)
        for m in per:
            print(f"  - Model {m['idx']}: AUC {m['auc']:.4f}")

    # Predict ensemble
    print("🔮 Predicting ensemble on VAL…")
    p_val,  nan_val  = _predict_probs(models, dl_val,  device, tta=args.tta, amp=args.amp, names=names_val)
    print("🔮 Predicting ensemble on TEST…")
    p_test, nan_test = _predict_probs(models, dl_test, device, tta=args.tta, amp=args.amp, names=names_test)

    print(f"Val predictions - Min: {np.nanmin(p_val):.4f}, Max: {np.nanmax(p_val):.4f}, NaN count: {np.isnan(p_val).sum()}")
    print(f"Test predictions - Min: {np.nanmin(p_test):.4f}, Max: {np.nanmax(p_test):.4f}, NaN count: {np.isnan(p_test).sum()}")

    if nan_val or nan_test:
        print("⚠️ NaNs detected in predictions:")
        for idx, fname in nan_val:  print(f"  • VAL NaN:  {fname}")
        for idx, fname in nan_test: print(f"  • TEST NaN: {fname}")
        p_val  = np.nan_to_num(p_val,  nan=0.5)
        p_test = np.nan_to_num(p_test, nan=0.5)

    # Tune threshold on Val for chosen metric
    t_best, v_best = _grid_best_threshold(p_val, y_val, metric=args.opt_metric)
    mv  = _metrics_from_probs(p_val,  y_val,  thr=t_best)
    mt  = _metrics_from_probs(p_test, y_test, thr=t_best)

    print(f"\n🎯 Best threshold (Val, {args.opt_metric}): {t_best:.4f}  (score={v_best:.4f})")
    print(f"📊 VAL  (thr={t_best:.4f}): acc={mv['acc']:.4f} f1={mv['f1']:.4f} prec={mv['prec']:.4f} rec={mv['rec']:.4f} auc={mv['auc']:.4f}")
    print(f"📊 TEST (thr={t_best:.4f}): acc={mt['acc']:.4f} f1={mt['f1']:.4f} prec={mt['prec']:.4f} rec={mt['rec']:.4f} auc={mt['auc']:.4f}")

if __name__ == "__main__":
    main()
