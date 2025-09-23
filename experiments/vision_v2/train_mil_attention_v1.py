# experiments/vision_v2/train_mil_attention_v1.py
"""
Attention-MIL on image patches (bags of instances).

Motivation:
- Your 2D tooth images contain small local cues (discolorations, margins, defects).
- With few images, training on full frames under-utilizes data. MIL grows the
  effective sample count by extracting K patches per image and learns to attend
  to the discriminative ones.

This script:
- Splits each image into K random resized crops of size img_size
- Uses a pretrained CNN (timm) to embed each patch
- Aggregates instance features with the Ilse et al. (2018) attention-MIL pooling
- Trains a binary classifier on bag-level labels (y_majority)
- GroupKFold by origin_id

Usage:
python experiments/vision_v2/train_mil_attention_v1.py \
  --csv data/excel/data_dl_augmented.csv --image-root data/augmented \
  --outdir weights/mil_v1 --backbone tf_efficientnet_b0_ns \
  --img-size 320 --instances 12 --epochs 30 --batch-size 6 --lr 2e-4 --amp
"""
import argparse, os, json, random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support

try:
    import timm
except Exception:
    raise SystemExit("Install timm first: pip install timm")

from PIL import Image
import torchvision.transforms as T

def set_seed(seed=123):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def compute_metrics(y_true, y_prob, thr=0.5):
    y_pred = (y_prob >= thr).astype(int)
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true))>1 else float('nan')
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    return {'auc': float(np.round(auc,4)), 'acc': float(np.round(acc,4)),
            'prec': float(np.round(prec,4)), 'rec': float(np.round(rec,4)), 'f1': float(np.round(f1,4))}

def make_transforms(img_size, train=True):
    if train:
        return T.Compose([
            T.RandomResizedCrop(img_size, scale=(0.4, 1.0), interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.ColorJitter(0.15,0.15,0.15,0.05),
            T.ToTensor(),
            T.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ])
    else:
        return T.Compose([
            T.Resize(int(img_size*1.14), interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(img_size),
            T.ToTensor(),
            T.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ])

class TeethMILBag(Dataset):
    def __init__(self, df, image_root, img_size=320, instances=12, train=True):
        self.df = df.reset_index(drop=True)
        self.image_root = Path(image_root)
        self.train = train
        self.instances = instances
        self.tf_train = make_transforms(img_size, train=True)
        self.tf_eval  = make_transforms(img_size, train=False)

    def __len__(self): return len(self.df)

    def _open(self, name):
        p = self.image_root / name
        if not p.exists():
            # try case-insensitive search
            cand = list(self.image_root.glob(name.split('.')[0]+'*'))
            if len(cand): p = cand[0]
        img = Image.open(p).convert('RGB')
        return img

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = self._open(r['image_name'])
        y = torch.tensor(float(r['y_majority']), dtype=torch.float32)
        # Build a bag of K crops
        crops = []
        tf = self.tf_train if self.train else self.tf_eval
        for _ in range(self.instances):
            crops.append(tf(img))
        bag = torch.stack(crops, dim=0)  # (K,C,H,W)
        return bag, y, r['image_name']

def mil_collate(batch):
    # Each element: (bag[K,C,H,W], y, name)
    bags, ys, names = zip(*batch)
    bags = torch.stack(bags, 0)  # (B,K,C,H,W)
    ys = torch.stack(ys, 0)      # (B,)
    return bags, ys, names

class AttentionMIL(nn.Module):
    """Attention-based MIL pooling (Ilse et al., 2018) with gated attention."""
    def __init__(self, in_dim, hid=128):
        super().__init__()
        self.attention_V = nn.Linear(in_dim, hid)
        self.attention_U = nn.Linear(in_dim, hid)
        self.attention_w = nn.Linear(hid, 1)
    def forward(self, H):  # H: (B,K,D)
        A_V = torch.tanh(self.attention_V(H))
        A_U = torch.sigmoid(self.attention_U(H))
        A = self.attention_w(A_V * A_U).squeeze(-1)   # (B,K)
        A = torch.softmax(A, dim=1)                   # attention weights
        M = torch.einsum('bkd,bk->bd', H, A)          # bag representation
        return M, A

class MILNet(nn.Module):
    def __init__(self, backbone='tf_efficientnet_b0_ns', drop=0.2):
        super().__init__()
        self.encoder = timm.create_model(backbone, pretrained=True, num_classes=0, global_pool='avg')
        d = self.encoder.num_features
        self.mil = AttentionMIL(d, hid=128)
        self.drop = nn.Dropout(p=drop)
        self.head = nn.Linear(d, 1)
    def forward(self, x):  # x: (B,K,C,H,W)
        B,K,C,H,W = x.shape
        x = x.view(B*K, C, H, W)
        feats = self.encoder(x)                       # (B*K, D)
        feats = feats.view(B, K, -1)                  # (B,K,D)
        bag, A = self.mil(feats)                      # (B,D)
        bag = self.drop(bag)
        logit = self.head(bag).squeeze(1)             # (B,)
        return logit, A

def run_fold(fold, df_trainval, df_test, args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    groups = df_trainval.get('origin_id', df_trainval.get('image_id'))
    gkf = GroupKFold(n_splits=args.folds)
    tr_idx, va_idx = list(gkf.split(df_trainval, groups=groups))[fold]
    tr_df = df_trainval.iloc[tr_idx].reset_index(drop=True)
    va_df = df_trainval.iloc[va_idx].reset_index(drop=True)

    ds_tr = TeethMILBag(tr_df, args.image_root, img_size=args.img_size, instances=args.instances, train=True)
    ds_va = TeethMILBag(va_df, args.image_root, img_size=args.img_size, instances=args.instances, train=False)
    ds_te = TeethMILBag(df_test, args.image_root, img_size=args.img_size, instances=args.instances, train=False)
    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                       pin_memory=True, collate_fn=mil_collate)
    dl_va = DataLoader(ds_va, batch_size=max(1,args.batch_size//2), shuffle=False, num_workers=args.workers,
                       pin_memory=True, collate_fn=mil_collate)
    dl_te = DataLoader(ds_te, batch_size=max(1,args.batch_size//2), shuffle=False, num_workers=args.workers,
                       pin_memory=True, collate_fn=mil_collate)

    model = MILNet(args.backbone, drop=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * max(1,len(dl_tr)))
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    best_auc = -1.0
    for epoch in range(1, args.epochs+1):
        model.train()
        losses = []
        for bags, y, _ in dl_tr:
            bags = bags.to(device)  # (B,K,C,H,W)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp):
                logit, A = model(bags)
                loss = F.binary_cross_entropy_with_logits(logit, y)
            scaler.scale(loss).backward()
            if args.grad_clip>0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt); scaler.update(); scheduler.step()
            losses.append(loss.item())
        # val
        model.eval()
        def _pred(dl):
            ys, probs = [], []
            with torch.no_grad():
                for bags, y, _ in dl:
                    p_list = []
                    # TTA: original + horizontal flip
                    for flip in [False, True]:
                        b = bags
                        if flip:
                            b = torch.flip(b, dims=[3])  # flip W
                        logit, _ = model(b.to(device))
                        p_list.append(torch.sigmoid(logit).cpu().numpy())
                    p = np.mean(np.stack(p_list,0),0)
                    ys.append(y.numpy()); probs.append(p)
            ys = np.concatenate(ys); probs = np.concatenate(probs)
            return ys, probs
        yv, pv = _pred(dl_va)
        best_thr, best_f1 = 0.5, -1.0
        for t in np.linspace(0.2,0.8,61):
            m = compute_metrics(yv, pv, thr=t)
            if m['f1'] > best_f1:
                best_f1, best_thr = m['f1'], float(t)
        mv = compute_metrics(yv, pv, thr=best_thr)
        print(f"[Fold {fold}][Epoch {epoch}] tr_loss={np.mean(losses):.4f}  val_auc={mv['auc']:.4f} f1={mv['f1']:.3f} thr*={best_thr:.3f}")
        if mv['auc'] > best_auc:
            best_auc = mv['auc']
            ckpt = {
                'model': model.state_dict(),
                'args': vars(args),
                'thr': best_thr,
                'epoch': epoch
            }
            Path(args.outdir).mkdir(parents=True, exist_ok=True)
            torch.save(ckpt, Path(args.outdir)/f"mil_v1_fold{fold}.pt")

    # reload best
    model.load_state_dict(torch.load(Path(args.outdir)/f"mil_v1_fold{fold}.pt", map_location=device)['model'])
    # final predictions
    def _predict(dl):
        ys, probs, names = [], [], []
        with torch.no_grad():
            for bags, y, n in dl:
                p_list = []
                for flip in [False, True]:
                    b = bags
                    if flip:
                        b = torch.flip(b, dims=[3])
                    logit, _ = model(b.to(device))
                    p_list.append(torch.sigmoid(logit).cpu().numpy())
                p = np.mean(np.stack(p_list,0),0)
                ys.append(y.numpy()); probs.append(p); names += list(n)
        ys = np.concatenate(ys); probs = np.concatenate(probs)
        return ys, probs, names
    yv, pv, names_v = _predict(dl_va)
    yt, pt, names_t = _predict(dl_te)
    mval = compute_metrics(yv, pv, thr=torch.load(Path(args.outdir)/f"mil_v1_fold{fold}.pt", map_location='cpu')['thr'])
    mtest = compute_metrics(yt, pt, thr=torch.load(Path(args.outdir)/f"mil_v1_fold{fold}.pt", map_location='cpu')['thr'])
    return {'val': mval, 'test': mtest,
            'oof': pd.DataFrame({'image_name': names_v, 'y': yv, 'prob': pv}),
            'pred_test': pd.DataFrame({'image_name': names_t, 'y': yt, 'prob': pt})}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, required=True, help='Path to CSV file')
    parser.add_argument('--image-root', type=str, required=True)
    parser.add_argument('--outdir', type=str, required=True)
    parser.add_argument('--backbone', type=str, default='tf_efficientnet_b0_ns')
    parser.add_argument('--img-size', type=int, default=320)
    parser.add_argument('--instances', type=int, default=12)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--batch-size', type=int, default=6)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--grad-clip', type=float, default=1.0)
    args = parser.parse_args()

    set_seed(2025)
    df = pd.read_csv(args.csv)
    if 'origin_id' not in df.columns:
        df['origin_id'] = df.get('image_id', df.index)
    df_trainval = df[df['split'].isin(['train','val'])].reset_index(drop=True)
    df_test = df[df['split']=='test'].reset_index(drop=True)

    all_val, all_test = [], []
    oofs, preds = [], []
    for fold in range(args.folds):
        res = run_fold(fold, df_trainval, df_test, args)
        all_val.append(res['val']); all_test.append(res['test'])
        oofs.append(res['oof']); preds.append(res['pred_test'])

    def _mean(dlist):
        keys = list(dlist[0].keys())
        return {k: float(np.round(np.mean([d[k] for d in dlist]),4)) for k in keys}
    val_mean = _mean(all_val); test_mean = _mean(all_test)

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    pd.concat(oofs).to_csv(Path(args.outdir)/"oof_val.csv", index=False)
    pd.concat(preds).to_csv(Path(args.outdir)/"pred_test.csv", index=False)
    with open(Path(args.outdir)/"summary.json", "w") as f:
        json.dump({'val_mean': val_mean, 'test_mean': test_mean}, f, indent=2)
    print("=== VAL (mean) ===", val_mean)
    print("=== TEST (mean) ===", test_mean)

if __name__ == '__main__':
    main()
