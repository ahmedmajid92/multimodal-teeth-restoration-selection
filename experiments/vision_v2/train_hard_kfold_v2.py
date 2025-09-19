# experiments/vision_v2/train_hard_kfold_v2.py
import argparse, json, time, math, random
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torchvision.transforms as T
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import timm
from sklearn.metrics import roc_auc_score, accuracy_score

# -----------------------
# Helpers
# -----------------------
def seed_all(seed=42):
    import os
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"]=str(seed)
    torch.backends.cudnn.benchmark = True

def count_params(m):
    tot=sum(p.numel() for p in m.parameters())
    trn=sum(p.numel() for p in m.parameters() if p.requires_grad)
    return tot,trn

def secs(s):
    m,s=divmod(int(s),60); h,m=divmod(m,60)
    return f"{h}h {m:02d}m {s:02d}s" if h else (f"{m}m {s:02d}s" if m else f"{s}s")

def ds_dist(s):
    vc=s.value_counts().to_dict()
    return " | ".join(f"{k}:{v}" for k,v in sorted(vc.items()))

# -----------------------
# Dataset / TFMs
# -----------------------
class ImgDS(Dataset):
    def __init__(self, df, root, tfm):
        self.df=df.reset_index(drop=True)
        self.root=Path(root)
        self.tfm=tfm
    def __len__(self): return len(self.df)
    def _resolve(self, name):
        p=self.root/str(name)
        if p.exists(): return p
        if p.suffix=="":
            for ext in [".jpg",".JPG",".jpeg",".JPEG",".png",".PNG"]:
                q=p.with_suffix(ext)
                if q.exists(): return q
        else:
            for q in [p.with_suffix(p.suffix.lower()), p.with_suffix(p.suffix.upper())]:
                if q.exists(): return q
        return p
    def __getitem__(self,i):
        r=self.df.iloc[i]
        p=self._resolve(r["image_name"])
        img=Image.open(p).convert("RGB")
        x=self.tfm(img)
        y=torch.tensor([float(r["y_majority"])], dtype=torch.float32)
        return x,y

def make_tfms(sz, augs="light"):
    augs = augs.lower()
    if augs == "none":
        train=T.Compose([
            T.Resize((sz,sz)),
            T.ToTensor(),
            T.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
        ])
    elif augs == "light":
        train=T.Compose([
            T.Resize((sz,sz)),
            T.RandomHorizontalFlip(),
            T.RandomRotation(8),
            T.ToTensor(),
            T.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
        ])
    else:  # "full"
        train=T.Compose([
            T.Resize((sz,sz)),
            T.RandomHorizontalFlip(),
            T.RandomRotation(10),
            T.ColorJitter(0.10,0.10,0.05,0.02),
            T.RandomResizedCrop(sz, scale=(0.85,1.0)),
            T.ToTensor(),
            T.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
        ])
    evalt=T.Compose([
        T.Resize((sz,sz)),
        T.ToTensor(),
        T.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
    ])
    return train, evalt

# -----------------------
# Eval
# -----------------------
@torch.no_grad()
def quick_train_sample_metrics(model, dl, device, use_amp=False, max_batches=4):
    """Compute AUC/ACC on a few train batches to confirm learning signal."""
    model.eval()
    probs=[]; ys=[]
    seen=0
    for bi,(x,y) in enumerate(dl, start=1):
        x=x.to(device, non_blocking=True); y=y.to(device, non_blocking=True)
        with torch.amp.autocast('cuda', enabled=use_amp):
            o=model(x)
        p=torch.sigmoid(o).detach().cpu().numpy().ravel()
        probs.append(p); ys.append(y.detach().cpu().numpy().ravel())
        seen += x.size(0)
        if bi>=max_batches: break
    probs=np.concatenate(probs) if probs else np.array([])
    ys=np.concatenate(ys) if ys else np.array([])
    if probs.size==0 or len(np.unique(ys))<2:
        return {"auc": float("nan"), "acc": float("nan")}
    return {"auc": float(roc_auc_score(ys, probs)),
            "acc": float(accuracy_score(ys, (probs>=0.5).astype(np.float32)))}

@torch.no_grad()
def evaluate(model, dl, criterion, device, use_amp=False):
    model.eval()
    total=0.0; probs=[]; ys=[]
    start=time.time()
    for x,y in dl:
        x=x.to(device, non_blocking=True); y=y.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            o=model(x); loss=criterion(o.view_as(y), y)
        total+=loss.item()*x.size(0)
        p=torch.sigmoid(o).detach().cpu().numpy().ravel()
        probs.append(p); ys.append(y.detach().cpu().numpy().ravel())
    dur=time.time()-start
    probs=np.concatenate(probs); ys=np.concatenate(ys)
    auc=roc_auc_score(ys, probs) if len(np.unique(ys))>1 else float("nan")
    acc=accuracy_score(ys, (probs>=0.5).astype(np.float32))
    ips=len(dl.dataset)/dur if dur>0 else 0.0
    return total/len(dl.dataset), {"auc":auc,"acc":acc,"imgs_per_sec":ips}

def grad_norm(model):
    total=0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.data.norm(2).item()**2
    return math.sqrt(total) if total>0 else 0.0

# -----------------------
# Train
# -----------------------
def train_one_epoch(model, dl, criterion, optimizer, device, epoch, epochs,
                    log_interval, accum_steps, scaler, use_amp, lr_sched=None,
                    debug_batch_stats=False):
    model.train()
    total=0.0
    start=time.time()
    n_batches=len(dl); seen=0
    batch_logit_std = None

    for b,(x,y) in enumerate(dl, start=1):
        x=x.to(device, non_blocking=True); y=y.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            o=model(x)
            loss=criterion(o.view_as(y), y)/accum_steps
        if debug_batch_stats and batch_logit_std is None:
            batch_logit_std = float(o.detach().std().cpu().item())
        scaler.scale(loss).backward()
        if b % accum_steps == 0:
            scaler.step(optimizer); scaler.update()
            optimizer.zero_grad(set_to_none=True)
        total += loss.item()*x.size(0)*accum_steps
        seen += x.size(0)

        if (b%log_interval==0) or (b==n_batches):
            elapsed=time.time()-start
            imgs_per_sec = seen/elapsed if elapsed>0 else 0.0
            current_lrs=sorted({pg['lr'] for pg in optimizer.param_groups})
            gnorm = grad_norm(model)
            print(f"[Ep {epoch:03d}/{epochs:03d}] "
                  f"Batch {b:04d}/{n_batches:04d} "
                  f"avg_loss={total/seen:.4f} "
                  f"lr={','.join(f'{lr:.2e}' for lr in current_lrs)} "
                  f"imgs/s={imgs_per_sec:.1f} "
                  f"grad_norm={gnorm:.2f}"
                  + (f"  logit_std~{batch_logit_std:.4f}" if batch_logit_std is not None else ""),
                  flush=True)

    if lr_sched is not None:
        lr_sched.step()
    return total/len(dl.dataset)

# -----------------------
# Main
# -----------------------
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--csv-path", required=True)
    ap.add_argument("--images-root", required=True)
    ap.add_argument("--model-name", default="convnextv2_base")

    # Progressive stages
    ap.add_argument("--stages", default="384,512")
    ap.add_argument("--epochs", default="10,10")
    ap.add_argument("--batch-sizes", default="16,8")
    ap.add_argument("--lrs", default="2e-4,1e-4")

    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=42)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--fold-index", type=int, default=0)
    ap.add_argument("--group-col", default="origin_id")
    ap.add_argument("--save-dir", default="weights/v2_kfold")
    ap.add_argument("--run-name", default="hard_kfold")

    # Speed & robustness
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--accum-steps", type=int, default=1)
    ap.add_argument("--augs", choices=["none","light","full"], default="light")
    ap.add_argument("--log-interval", type=int, default=100)

    # ⚠️ Debug tools
    ap.add_argument("--overfit-n", type=int, default=0, help="train/val on only N samples (tiny sanity check)")
    ap.add_argument("--head-only-epochs", type=int, default=2, help="freeze backbone for first N epochs of stage 1")
    ap.add_argument("--use-sampler", action="store_true", help="use WeightedRandomSampler instead of pos_weight")
    ap.add_argument("--debug-batch-stats", action="store_true", help="log per-epoch batch logit std")

    args=ap.parse_args()
    seed_all(args.seeds)

    # Parse stages
    sizes=[int(s) for s in args.stages.split(",")]
    epochs=[int(s) for s in args.epochs.split(",")]
    bss=[int(s) for s in args.batch_sizes.split(",")]
    lrs=[float(s) for s in args.lrs.split(",")]
    assert len({len(sizes),len(epochs),len(bss),len(lrs)})==1, "Stage lists must align"

    # Load table
    p=Path(args.csv_path)
    print(f"📄 Loading: {p}")
    df=pd.read_excel(p) if p.suffix.lower() in [".xlsx",".xls"] else pd.read_csv(p)
    assert "y_majority" in df.columns
    assert args.group_col in df.columns

    # Build folds
    y=df["y_majority"].astype(int).values
    groups=df[args.group_col].values
    skf=StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seeds)
    splits=list(skf.split(np.zeros(len(df)), y, groups))
    ti,vi=splits[args.fold_index]
    dtr,dva=df.iloc[ti].copy(), df.iloc[vi].copy()

    if args.overfit_n>0:
        # tiny subset sanity check (but keep group integrity best-effort)
        dtr = dtr.sample(n=min(args.overfit_n, len(dtr)), random_state=args.seeds)
        dva = dva.sample(n=min(max(args.overfit_n//4, 1), len(dva)), random_state=args.seeds)
        print(f"🔬 Overfit mode ON: train={len(dtr)} val={len(dva)}")

    print(f"🧩 Folds={args.folds} | Using fold={args.fold_index}")
    print(f"   Train: {len(dtr)} | Val: {len(dva)}")
    print(f"   Train y: {ds_dist(dtr['y_majority'])}")
    print(f"   Val   y: {ds_dist(dva['y_majority'])}")
    print(f"   Unique groups → train {len(set(dtr[args.group_col]))}, val {len(set(dva[args.group_col]))}")

    # Class weighting vs sampler
    pos = int((dtr["y_majority"]==1).sum())
    neg = int((dtr["y_majority"]==0).sum())
    pos_weight_value = neg / max(pos,1)
    print(f"⚖️  Class counts (train): pos={pos} neg={neg}  -> pos_weight={pos_weight_value:.3f}  | sampler={'ON' if args.use_sampler else 'OFF'}")

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=timm.create_model(args.model_name, pretrained=True, num_classes=1).to(device)
    tot,trn=count_params(model)
    print(f"🧠 {args.model_name} | params total={tot:,} trainable={trn:,}")
    print(f"💻 Device: {device} (AMP={'on' if args.amp else 'off'})")

    out_dir=Path(args.save_dir); out_dir.mkdir(parents=True, exist_ok=True)
    ck=out_dir/f"{args.run_name}_f{args.fold_index}_{sizes[-1]}.pt"
    scaler=torch.amp.GradScaler('cuda', enabled=args.amp)

    best=float("inf"); bestm={}
    global_start=time.time()
    global_epoch=0

    # Stage loop
    for stage,(sz,ep,bs,lr) in enumerate(zip(sizes,epochs,bss,lrs), start=1):
        print(f"\n=== Stage {stage}/{len(sizes)} | size={sz} | epochs={ep} | bs={bs} | lr={lr} | augs={args.augs} ===")
        tr_tfm, ev_tfm = make_tfms(sz, augs=args.augs)
        ds_tr=ImgDS(dtr, args.images_root, tr_tfm)
        ds_va=ImgDS(dva, args.images_root, ev_tfm)

        if args.use_sampler:
            # Weighted sampler (slower but robust to skew)
            counts = dtr["y_majority"].value_counts()
            w = dtr["y_majority"].map(lambda c: 1.0 / max(counts[c],1)).astype("float32").values
            sampler = WeightedRandomSampler(w, num_samples=len(w), replacement=True)
            dl_tr=DataLoader(ds_tr, batch_size=bs, sampler=sampler,
                             num_workers=args.num_workers, pin_memory=True,
                             persistent_workers=True, drop_last=False)
        else:
            dl_tr=DataLoader(ds_tr, batch_size=bs, shuffle=True,
                             num_workers=args.num_workers, pin_memory=True,
                             persistent_workers=True, drop_last=False)
        dl_va=DataLoader(ds_va, batch_size=bs, shuffle=False,
                         num_workers=args.num_workers, pin_memory=True,
                         persistent_workers=True, drop_last=False)
        print(f"🧵 train: batches={len(dl_tr)}  | val: batches={len(dl_va)}")

        # Loss / Optim / LR
        if args.use_sampler:
            criterion = nn.BCEWithLogitsLoss().to(device)
        else:
            criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_value], device=device)).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
        lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=ep)

        # Optional head-only warmup on Stage 1
        if stage==1 and args.head_only_epochs>0:
            print(f"🧊 Head-only warmup: freezing backbone for {args.head_only_epochs} epoch(s)")
            for n,p in model.named_parameters():
                if "head" in n or "classifier" in n:
                    p.requires_grad=True
                else:
                    p.requires_grad=False

        for e in range(1, ep+1):
            global_epoch += 1
            ep_start=time.time()

            # Unfreeze after head-only warmup
            if stage==1 and e==args.head_only_epochs+1:
                print("🔥 Unfreezing backbone")
                for p in model.parameters(): p.requires_grad=True

            tr_loss = train_one_epoch(
                model, dl_tr, criterion, optimizer, device,
                epoch=global_epoch, epochs=sum(epochs),
                log_interval=args.log_interval, accum_steps=args.accum_steps,
                scaler=scaler, use_amp=args.amp, lr_sched=None,
                debug_batch_stats=args.debug_batch_stats
            )
            lr_sched.step()
            va_loss, m = evaluate(model, dl_va, criterion, device, use_amp=args.amp)

            # quick train sample metrics (fast learnability check)
            tr_probe = quick_train_sample_metrics(model, dl_tr, device, use_amp=args.amp, max_batches=3)

            elapsed=time.time()-ep_start
            improved = va_loss < best
            if improved:
                best=va_loss; bestm=m
                torch.save({"model":model.state_dict(), "epoch":global_epoch}, ck)

            print(f"✅ [Fold {args.fold_index}] Stage {stage} Ep {e}/{ep} "
                  f"train={tr_loss:.4f}  val={va_loss:.4f}  "
                  f"AUC={m['auc']:.3f} ACC={m['acc']:.3f} "
                  f"| trainProbe AUC={tr_probe['auc']:.3f} ACC={tr_probe['acc']:.3f} "
                  f"{'*' if improved else ''}  epoch_time={secs(elapsed)}",
                  flush=True)

    total=time.time()-global_start
    with open(out_dir/f"{args.run_name}_fold{args.fold_index}_summary.json","w") as f:
        json.dump({"best_val_loss":best, "best_val_metrics":bestm}, f, indent=2)
    print(f"\n🧾 Saved: {ck}")
    print(f"⏱️ Total time: {secs(total)}")

if __name__=="__main__":
    main()
