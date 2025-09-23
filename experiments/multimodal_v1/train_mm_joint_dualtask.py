# experiments/multimodal_v1/train_mm_joint_dualtask.py
"""
Multimodal (Image + Tabular) end‑to‑end training with a dual‑task head.

- Shared CNN encoder (timm) for images
- MLP encoder for tabular features (9 numerical features from data_*.xlsx)
- Fusion by concatenation + dropout
- Two heads:
    * cls_head -> binary classification for "Indirect" (hard labels y_majority)
    * reg_head -> regression for p_indirect (soft label in [0, 1])
- Loss = alpha * BCEWithLogitsLoss(hard) + beta * BCEWithLogitsLoss(soft)  # soft targets are allowed
- GroupKFold by origin_id
- Temperature scaling (learned on each fold's validation set) for calibration
- Test‑time augmentation (horizontal/vertical flips) at evaluation
- Saves:
    * weights per fold (.pt)
    * a JSON summary and OOF/test csvs

Usage (example):
python experiments/multimodal_v1/train_mm_joint_dualtask.py \
    --xlsx data/data_dl_augmented.xlsx \
    --image-root data/augmented \
    --outdir weights/mm_dualtask_v1 \
    --backbone tf_efficientnet_b4_ns \
    --epochs 30 --batch-size 32 --lr 3e-4 --alpha 1.0 --beta 0.3 --folds 5 --amp
"""
import argparse, os, json, math, time, random
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import warnings

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast

from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler

# Suppress sklearn feature name warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")

try:
    import timm
except Exception as e:
    raise SystemExit("This script requires timm. Install with: pip install timm albumentations opencv-python-headless")

from PIL import Image

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.JPG', '.JPEG', '.PNG'}

TAB_FEATURES = [
    'depth','width','enamel_cracks','occlusal_load','carious_lesion',
    'opposing_type','adjacent_teeth','age_range','cervical_lesion'
]

def set_seed(seed: int = 1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def fast_round(x: float, n: int = 4) -> float:
    return float(np.round(x, n))

def timm_transforms(img_size: int, train: bool):
    # Use timm native transforms for reproducible aug
    if train:
        return timm.data.create_transform(
            input_size=img_size,
            is_training=True,
            auto_augment='rand-m9-mstd0.5-inc1',
            interpolation='bicubic',
            re_prob=0.2,
            re_mode='pixel',
            re_count=1,
            mean=(0.485,0.456,0.406),
            std=(0.229,0.224,0.225),
        )
    else:
        return timm.data.create_transform(
            input_size=img_size,
            is_training=False,
            interpolation='bicubic',
            mean=(0.485,0.456,0.406),
            std=(0.229,0.224,0.225),
        )

class TeethMM(Dataset):
    def __init__(self, df: pd.DataFrame, image_root: str, transform, scaler: Optional[StandardScaler]=None):
        self.df = df.reset_index(drop=True)
        self.image_root = Path(image_root)
        self.transform = transform
        self.scaler = scaler
        # Fill NaNs; treat -1 as a valid category (leave as -1)
        self.df[TAB_FEATURES] = self.df[TAB_FEATURES].fillna(0)

    def __len__(self): return len(self.df)

    def _load_image(self, fname: str) -> Image.Image:
        p = self.image_root / fname
        if not p.suffix:
            # sometimes names are like "123" without ext – find any existing
            for ext in IMG_EXTS:
                cand = self.image_root / (fname + ext)
                if cand.exists():
                    p = cand
                    break
        if not p.exists():
            raise FileNotFoundError(f"Image not found: {p}")
        img = Image.open(p).convert('RGB')
        return img

    def __getitem__(self, i: int):
        r = self.df.iloc[i]
        img = self._load_image(str(r['image_name']))
        img = self.transform(img)
        # tabular
        tab = r[TAB_FEATURES].astype(np.float32).values
        if self.scaler is not None:
            tab = self.scaler.transform([tab])[0]
        tab = torch.tensor(tab, dtype=torch.float32)
        # targets
        y_hard = torch.tensor(float(r['y_majority']), dtype=torch.float32)  # 0/1
        y_soft = torch.tensor(float(r['p_indirect']), dtype=torch.float32)  # in [0,1]
        w = torch.tensor(float(r.get('weight', 1.0)), dtype=torch.float32)
        return img, tab, y_hard, y_soft, w, r['image_name']

class MMJointDualHead(nn.Module):
    def __init__(self, backbone='tf_efficientnet_b4_ns', tab_in=9, tab_hidden=64, drop=0.2):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=True, num_classes=0, global_pool='avg')
        feat_dim = self.backbone.num_features
        self.tab = nn.Sequential(
            nn.Linear(tab_in, tab_hidden),
            nn.BatchNorm1d(tab_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=drop),
            nn.Linear(tab_hidden, tab_hidden),
            nn.ReLU(inplace=True),
        )
        fusion_dim = feat_dim + tab_hidden
        self.fusion = nn.Sequential(nn.Dropout(p=drop))
        self.cls_head = nn.Linear(fusion_dim, 1)
        self.reg_head = nn.Linear(fusion_dim, 1)

    def forward(self, x_img, x_tab):
        f_img = self.backbone(x_img)          # (B, F)
        f_tab = self.tab(x_tab)               # (B, H)
        f = torch.cat([f_img, f_tab], dim=1)  # (B, F+H)
        f = self.fusion(f)
        logit = self.cls_head(f).squeeze(1)
        reg   = self.reg_head(f).squeeze(1)
        return logit, reg

class TemperatureScaler(nn.Module):
    """Platt scaling implemented as a single temperature parameter."""
    def __init__(self):
        super().__init__()
        self.log_T = nn.Parameter(torch.zeros(1))  # T = exp(log_T)

    def forward(self, logits):
        T = self.log_T.exp()
        return logits / T

    @torch.no_grad()
    def temperature(self):
        return float(self.log_T.exp().item())

def bce_logits_with_soft_targets(logits, targets, weight=None):
    # BCEWithLogitsLoss supports float targets in [0,1]
    loss = F.binary_cross_entropy_with_logits(logits, targets, weight=weight, reduction='mean')
    return loss

def compute_metrics(y_true, y_prob, thr=0.5) -> Dict[str, float]:
    y_pred = (y_prob >= thr).astype(int)
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float('nan')
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    return {'auc': fast_round(auc), 'acc': fast_round(acc), 'prec': fast_round(prec), 'rec': fast_round(rec), 'f1': fast_round(f1)}

def run_fold(fold:int, df_trainval: pd.DataFrame, df_test: pd.DataFrame, args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Split by GroupKFold on origin_id; if not available, fall back to image_id
    groups = df_trainval.get('origin_id', df_trainval.get('image_id', None))
    if groups is None: 
        raise ValueError("No 'origin_id' or 'image_id' column in xlsx.")
    gkf = GroupKFold(n_splits=args.folds)
    idxs = list(gkf.split(df_trainval, groups=groups))
    tr_idx, va_idx = idxs[fold]
    tr_df = df_trainval.iloc[tr_idx].reset_index(drop=True)
    va_df = df_trainval.iloc[va_idx].reset_index(drop=True)

    # Standardize tabular features - fit on numpy array to avoid feature name warnings
    scaler = StandardScaler()
    train_features = tr_df[TAB_FEATURES].fillna(0).values  # Convert to numpy array
    scaler.fit(train_features)

    # Datasets / loaders
    t_train = timm_transforms(args.img_size, train=True)
    t_eval  = timm_transforms(args.img_size, train=False)
    ds_tr = TeethMM(tr_df, args.image_root, t_train, scaler)
    ds_va = TeethMM(va_df, args.image_root, t_eval, scaler)
    ds_te = TeethMM(df_test, args.image_root, t_eval, scaler)
    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size*2, shuffle=False, num_workers=args.workers, pin_memory=True)
    dl_te = DataLoader(ds_te, batch_size=args.batch_size*2, shuffle=False, num_workers=args.workers, pin_memory=True)

    # Model / opt
    model = MMJointDualHead(args.backbone, tab_in=len(TAB_FEATURES), tab_hidden=args.tab_hidden, drop=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    iters_per_epoch = max(1, len(dl_tr))
    # Cosine schedule
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * iters_per_epoch)
    scaler_amp = GradScaler(device, enabled=args.amp)

    # For class imbalance, weight positives inversely
    pos_weight = None
    if args.pos_weight is not None:
        pos_weight = torch.tensor([args.pos_weight], device=device)

    # Train
    best_auc = -1.0
    best_state = None
    history = {'epoch': [], 'tr_loss': [], 'va_auc': [], 'va_f1': []}

    for epoch in range(1, args.epochs+1):
        model.train()
        loss_meter = []
        for (x_img, x_tab, y_h, y_s, w, _) in dl_tr:
            x_img = x_img.to(device); x_tab = x_tab.to(device)
            y_h = y_h.to(device); y_s = y_s.to(device)
            w = w.to(device)

            opt.zero_grad(set_to_none=True)
            with autocast(device, enabled=args.amp):
                logit, reg = model(x_img, x_tab)
                loss_h = bce_logits_with_soft_targets(logit, y_h, weight=None if args.use_sample_weights==0 else w)
                # For soft targets, use BCE as well
                loss_s = bce_logits_with_soft_targets(reg, y_s, weight=None if args.use_sample_weights==0 else w)
                loss = args.alpha * loss_h + args.beta * loss_s
            scaler_amp.scale(loss).backward()
            if args.grad_clip > 0:
                scaler_amp.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler_amp.step(opt)
            scaler_amp.update()
            scheduler.step()

            loss_meter.append(loss.item())

        # Eval on val
        model.eval()
        va_logits, va_probs, va_y = [], [], []
        with torch.no_grad():
            for (x_img, x_tab, y_h, y_s, w, names) in dl_va:
                x_img = x_img.to(device); x_tab = x_tab.to(device)
                logit, reg = model(x_img, x_tab)
                prob = torch.sigmoid(logit)
                va_logits.append(logit.float().cpu().numpy())
                va_probs.append(prob.float().cpu().numpy())
                va_y.append(y_h.cpu().numpy())
        va_logits = np.concatenate(va_logits); va_probs = np.concatenate(va_probs); va_y = np.concatenate(va_y)
        # Temperature scaling on-the-fly (few steps) to align calibration
        scaler_T = TemperatureScaler().to(device)
        opt_T = torch.optim.LBFGS(scaler_T.parameters(), lr=0.1, max_iter=50)
        logits_tensor = torch.tensor(va_logits, dtype=torch.float32, device=device)
        targets_tensor = torch.tensor(va_y, dtype=torch.float32, device=device)
        def _closure():
            opt_T.zero_grad()
            logits_adj = scaler_T(logits_tensor)
            loss_T = F.binary_cross_entropy_with_logits(logits_adj, targets_tensor)
            loss_T.backward()
            return loss_T
        try:
            opt_T.step(_closure)
        except Exception:
            pass
        with torch.no_grad():
            adj_logits = scaler_T(logits_tensor).cpu().numpy()
            va_probs = 1/(1+np.exp(-adj_logits))

        # Metrics at best threshold on val (sweep)
        best_thr, best_f1 = 0.5, -1.0
        for t in np.linspace(0.2, 0.8, 61):
            m = compute_metrics(va_y, va_probs, thr=t)
            if m['f1'] > best_f1:
                best_f1 = m['f1']; best_thr = float(t)
        m_va = compute_metrics(va_y, va_probs, thr=best_thr)
        print(f"[Fold {fold}][Epoch {epoch}] tr_loss={np.mean(loss_meter):.4f}  val_auc={m_va['auc']:.4f} f1={m_va['f1']:.3f} thr*={best_thr:.3f}  T={scaler_T.temperature():.3f}")

        history['epoch'].append(epoch); history['tr_loss'].append(np.mean(loss_meter)); history['va_auc'].append(m_va['auc']); history['va_f1'].append(m_va['f1'])

        if m_va['auc'] > best_auc:
            best_auc = m_va['auc']
            best_state = {
                'model': model.state_dict(),
                'scaler_mean': getattr(ds_tr.scaler, 'mean_', None),
                'scaler_scale': getattr(ds_tr.scaler, 'scale_', None),
                'thr': best_thr,
                'T': scaler_T.temperature(),
                'args': vars(args),
                'epoch': epoch,
            }
            # save checkpoint
            outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
            torch.save(best_state, outdir / f"mm_dualtask_fold{fold}.pt")

    # Prepare OOF preds for val and preds for test
    # Reload best model & calibrator
    ckpt = best_state
    model.load_state_dict(ckpt['model'])
    T = ckpt['T']; thr = ckpt['thr']

    def _predict(dl: DataLoader):
        logits_all, probs_all, yh_all, names_all = [], [], [], []
        with torch.no_grad():
            for (x_img, x_tab, y_h, y_s, w, names) in dl:
                x_img = x_img.to(device); x_tab = x_tab.to(device)
                # TTA (H/V flip)
                logits_list = []
                for flip in [None, 'h', 'v']:
                    xi = x_img.clone()
                    if flip == 'h':
                        xi = torch.flip(xi, dims=[3])
                    elif flip == 'v':
                        xi = torch.flip(xi, dims=[2])
                    logit, _ = model(xi, x_tab)
                    logits_list.append(logit)
                logit = torch.stack(logits_list, dim=0).mean(0)  # avg TTA
                prob = torch.sigmoid(logit / T)
                logits_all.append(logit.float().cpu().numpy())
                probs_all.append(prob.float().cpu().numpy())
                yh_all.append(y_h.cpu().numpy())
                names_all.extend(list(names))
        logits_all = np.concatenate(logits_all)
        probs_all = np.concatenate(probs_all)
        yh_all    = np.concatenate(yh_all)
        return logits_all, probs_all, yh_all, names_all

    va_logits, va_probs, va_y, va_names = _predict(dl_va)
    te_logits, te_probs, te_y, te_names = _predict(dl_te)

    va_m = compute_metrics(va_y, va_probs, thr=thr)
    te_m = compute_metrics(te_y, te_probs, thr=thr)
    return {
        'fold': fold,
        'thr': thr,
        'T': T,
        'val_metrics': va_m,
        'test_metrics': te_m,
        'val_oof': pd.DataFrame({'image_name': va_names, 'y': va_y, 'prob': va_probs}),
        'test_pred': pd.DataFrame({'image_name': te_names, 'y': te_y, 'prob': te_probs}),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--xlsx', type=str, required=True, help='Path to data_dl_augmented.xlsx (or similar)')
    parser.add_argument('--image-root', type=str, required=True, help='Root folder with images (augmented)')
    parser.add_argument('--outdir', type=str, required=True)
    parser.add_argument('--backbone', type=str, default='tf_efficientnet_b4_ns')
    parser.add_argument('--img-size', type=int, default=380)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--alpha', type=float, default=1.0, help='weight for BCE hard label')
    parser.add_argument('--beta', type=float, default=0.3, help='weight for BCE soft label')
    parser.add_argument('--tab-hidden', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--pos-weight', type=float, default=None)
    parser.add_argument('--use-sample-weights', type=int, default=0, choices=[0,1])
    parser.add_argument('--amp', action='store_true')
    args = parser.parse_args()
    set_seed(42)

    if args.xlsx.endswith('.csv'):
        df = pd.read_csv(args.xlsx)
    else:
        df = pd.read_excel(args.xlsx)

    # Expect columns: ['image_name', 'origin_id', features..., 'y_majority', 'p_indirect', 'split']
    assert 'image_name' in df.columns, "xlsx must have 'image_name'"
    if 'origin_id' not in df.columns:
        # fallback: use image_id
        df['origin_id'] = df.get('image_id', df.index)

    # Use provided splits: train/val/test in xlsx
    df_trainval = df[df['split'].isin(['train','val'])].copy().reset_index(drop=True)
    df_test = df[df['split']=='test'].copy().reset_index(drop=True)

    print(f"Train+Val: {len(df_trainval)}, Test: {len(df_test)}")

    all_fold_results = []
    oof_dfs = []
    test_dfs = []
    for fold in range(args.folds):
        res = run_fold(fold, df_trainval, df_test, args)
        all_fold_results.append({'fold': fold, **res['val_metrics'], **{f'test_{k}':v for k,v in res['test_metrics'].items()}})
        oof_dfs.append(res['val_oof'])
        test_dfs.append(res['test_pred'])

    oof_all = pd.concat(oof_dfs, axis=0).reset_index(drop=True)
    test_all = pd.concat(test_dfs, axis=0).reset_index(drop=True)

    # Summaries
    def _agg(rows, keys):
        return {k: fast_round(np.mean([r[k] for r in rows])) for k in keys}
    keys = ['auc','acc','prec','rec','f1']
    # all_fold_results already stores flattened metrics, e.g. 'auc', 'test_auc', etc.
    val_mean  = {k: fast_round(np.mean([r[k] for r in all_fold_results])) for k in keys}
    test_mean = {k: fast_round(np.mean([r[f'test_{k}'] for r in all_fold_results])) for k in keys}


    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    oof_all.to_csv(outdir / "oof_val.csv", index=False)
    test_all.to_csv(outdir / "pred_test.csv", index=False)

    summary = {
        'val_mean': val_mean,
        'test_mean': test_mean,
        'fold_details': all_fold_results
    }
    with open(outdir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("=== VAL (mean) ===", val_mean)
    print("=== TEST (mean) ===", test_mean)

if __name__ == '__main__':
    main()
