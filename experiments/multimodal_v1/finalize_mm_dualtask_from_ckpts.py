# experiments/multimodal_v1/finalize_mm_dualtask_from_ckpts.py
import argparse, json
import warnings
from pathlib import Path
import numpy as np, pandas as pd
import torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from PIL import Image
import timm

# Suppress sklearn feature name warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")

TAB_FEATURES = [
    'depth','width','enamel_cracks','occlusal_load','carious_lesion',
    'opposing_type','adjacent_teeth','age_range','cervical_lesion'
]

def timm_tf(img_size, train):
    return timm.data.create_transform(
        input_size=img_size,
        is_training=train,
        auto_augment='rand-m9-mstd0.5-inc1' if train else None,
        interpolation='bicubic',
        re_prob=0.2 if train else 0.0,
        mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225),
    )

class TeethMM(Dataset):
    def __init__(self, df, root, transform, scaler):
        self.df=df.reset_index(drop=True); self.root=Path(root); self.t=transform; self.scaler=scaler
        self.df[TAB_FEATURES]=self.df[TAB_FEATURES].fillna(0)
    def __len__(self): return len(self.df)
    def __getitem__(self,i):
        r=self.df.iloc[i]
        p=self.root/str(r['image_name'])
        if not p.exists():
            candidates=list(self.root.glob(str(r['image_name']).split('.')[0]+'*'))
            if candidates: p=candidates[0]
        img=Image.open(p).convert('RGB'); x=self.t(img)
        tab=r[TAB_FEATURES].astype(np.float32).values
        if self.scaler is not None: tab=self.scaler.transform([tab])[0]
        return x, torch.tensor(tab,dtype=torch.float32), torch.tensor(float(r['y_majority']),dtype=torch.float32), r['image_name']

class Model(nn.Module):
    def __init__(self, backbone='tf_efficientnet_b4_ns', tab_in=9, tab_h=64, drop=0.2):
        super().__init__()
        self.backbone=timm.create_model(backbone,pretrained=False,num_classes=0,global_pool='avg')
        d=self.backbone.num_features
        self.tab=nn.Sequential(nn.Linear(tab_in,tab_h), nn.BatchNorm1d(tab_h), nn.ReLU(), nn.Dropout(drop),
                               nn.Linear(tab_h,tab_h), nn.ReLU())
        self.fuse=nn.Dropout(drop)
        # Use the same names as in the saved checkpoint
        self.cls_head=nn.Linear(d+tab_h,1)
        self.reg_head=nn.Linear(d+tab_h,1)
    def forward(self, xi, xt):
        fi=self.backbone(xi); ft=self.tab(xt); f=self.fuse(torch.cat([fi,ft],1))
        return self.cls_head(f).squeeze(1), self.reg_head(f).squeeze(1)

def metrics(y,p,thr):
    from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support
    yhat=(p>=thr).astype(int)
    auc=roc_auc_score(y,p) if len(np.unique(y))>1 else float('nan')
    acc=accuracy_score(y,yhat)
    pr,rc,f1,_=precision_recall_fscore_support(y,yhat,average='binary',zero_division=0)
    r=lambda z: float(np.round(z,4))
    return dict(auc=r(auc),acc=r(acc),prec=r(pr),rec=r(rc),f1=r(f1))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--csv',required=True, help='Path to CSV file (e.g., data_dl_augmented.csv)')
    ap.add_argument('--image-root',required=True)
    ap.add_argument('--ckpt-dir',required=True, help='folder with mm_dualtask_fold*.pt')
    ap.add_argument('--outdir',required=True)
    ap.add_argument('--folds',type=int,default=5)
    args=ap.parse_args()

    # Check if file exists and provide helpful error message
    csv_path = Path(args.csv)
    if not csv_path.exists():
        data_dir = Path('data')
        if data_dir.exists():
            print(f"Error: {args.csv} not found!")
            print(f"Available files in data/ directory (recursive search):")
            for f in data_dir.rglob('*.csv'):
                print(f"  - {f}")
            for f in data_dir.rglob('*.xlsx'):
                print(f"  - {f}")
        else:
            print(f"Error: {args.csv} not found and data/ directory doesn't exist!")
        raise FileNotFoundError(f"File not found: {args.csv}")

    # Read CSV file
    df = pd.read_csv(args.csv)
    
    if 'origin_id' not in df.columns: df['origin_id']=df.get('image_id', df.index)
    df_tv=df[df['split'].isin(['train','val'])].reset_index(drop=True)
    df_te=df[df['split']=='test'].reset_index(drop=True)

    gkf=GroupKFold(n_splits=args.folds)
    splits=list(gkf.split(df_tv, groups=df_tv['origin_id']))

    device='cuda' if torch.cuda.is_available() else 'cpu'
    oof_list=[]; test_list=[]; fold_summ=[]
    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    for fold,(tr,va) in enumerate(splits):
        ckpt_path=Path(args.ckpt_dir)/f"mm_dualtask_fold{fold}.pt"
        if not ckpt_path.exists():
            print(f"[WARN] missing {ckpt_path}, skipping fold {fold}")
            continue
        
        # Fix for PyTorch 2.6: set weights_only=False for trusted checkpoints
        ckpt=torch.load(ckpt_path, map_location='cpu', weights_only=False)
        a=ckpt['args']; thr=ckpt['thr']; T=ckpt['T']
        model=Model(backbone=a['backbone'], tab_in=len(TAB_FEATURES), tab_h=a['tab_hidden'], drop=a['dropout']).to(device)
        model.load_state_dict(ckpt['model']); model.eval()

        # rebuild scaler from saved stats - remove feature names to avoid warnings
        scaler=StandardScaler()
        scaler.mean_=np.array(ckpt['scaler_mean']) if ckpt['scaler_mean'] is not None else np.zeros(len(TAB_FEATURES))
        scaler.scale_=np.array(ckpt['scaler_scale']) if ckpt['scaler_scale'] is not None else np.ones(len(TAB_FEATURES))
        scaler.var_=scaler.scale_**2
        scaler.n_features_in_=len(TAB_FEATURES)
        # Don't set feature_names_in_ to avoid warnings when transforming numpy arrays
        # scaler.feature_names_in_=np.array(TAB_FEATURES,dtype=object)

        tf_eval=timm_tf(a['img_size'], False)

        va_df=df_tv.iloc[va].reset_index(drop=True)
        ds_va=TeethMM(va_df, args.image_root, tf_eval, scaler)
        dl_va=DataLoader(ds_va, batch_size=a['batch_size']*2, shuffle=False, num_workers=a['workers'], pin_memory=True)

        te_df=df_te.reset_index(drop=True)
        ds_te=TeethMM(te_df, args.image_root, tf_eval, scaler)
        dl_te=DataLoader(ds_te, batch_size=a['batch_size']*2, shuffle=False, num_workers=a['workers'], pin_memory=True)

        def predict(dl):
            ys=[]; ps=[]; names=[]
            with torch.no_grad():
                for xi, xt, y, n in dl:
                    xi=xi.to(device); xt=xt.to(device)
                    # simple TTA (h & v flips)
                    logits=[]
                    for flip in [None,'h','v']:
                        x=xi
                        if flip=='h': x=torch.flip(x,[3])
                        elif flip=='v': x=torch.flip(x,[2])
                        logit,_=model(x,xt)
                        logits.append(logit)
                    logit=torch.stack(logits,0).mean(0)
                    p=torch.sigmoid(logit/T).cpu().numpy()
                    ys.append(y.numpy()); ps.append(p); names+=list(n)
            return np.concatenate(ys), np.concatenate(ps), names

        yv,pv,nv=predict(dl_va)
        yt,pt,nt=predict(dl_te)

        oof_list.append(pd.DataFrame({'image_name':nv,'y':yv,'prob':pv}))
        test_list.append(pd.DataFrame({'image_name':nt,'y':yt,'prob':pt}))
        fold_summ.append({'fold':fold, 'thr':thr, 'T':T,
                          'val':metrics(yv,pv,thr), 'test':metrics(yt,pt,thr)})

        print(f"[Fold {fold}] VAL {fold_summ[-1]['val']} | TEST {fold_summ[-1]['test']}")

    if not oof_list:
        raise SystemExit("No folds finalized. Check ckpt-dir path.")

    oof=pd.concat(oof_list).reset_index(drop=True)
    test=pd.concat(test_list).reset_index(drop=True)
    oof.to_csv(Path(args.outdir)/"oof_val.csv", index=False)
    test.to_csv(Path(args.outdir)/"pred_test.csv", index=False)

    # mean over folds
    keys=['auc','acc','prec','rec','f1']
    val_mean  = {k: float(np.round(np.mean([f['val'][k]  for f in fold_summ]),4)) for k in keys}
    test_mean = {k: float(np.round(np.mean([f['test'][k] for f in fold_summ]),4)) for k in keys}
    with open(Path(args.outdir)/"summary.json","w") as f:
        json.dump({'val_mean':val_mean,'test_mean':test_mean,'folds':fold_summ}, f, indent=2)
    print("=== VAL (mean) ===", val_mean)
    print("=== TEST (mean) ===", test_mean)

if __name__=="__main__":
    main()
