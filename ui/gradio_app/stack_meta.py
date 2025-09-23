from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support, roc_curve

class Stacker:
    def __init__(self, xlsx_tab, oof_mm, pred_mm, oof_mil, pred_mil, folds=5, thr_mode='max_acc', thr_target=0.80):
        self.xlsx_tab = Path(xlsx_tab)
        self.oof_mm = Path(oof_mm); self.pred_mm = Path(pred_mm)
        self.oof_mil = Path(oof_mil); self.pred_mil = Path(pred_mil)
        self.folds = folds
        self.thr_mode = thr_mode
        self.thr_target = thr_target
        # Fit metas (3-stream and 2-stream) on OOF
        self.meta_full, self.thr_full = None, 0.5
        self.meta_img,  self.thr_img  = None, 0.5
        self._fit_metas()

    def set_threshold_mode(self, mode, target=0.80):
        self.thr_mode = mode
        self.thr_target = target
        # Recompute thresholds with same trained metas
        if self.meta_full is not None:
            p = self.meta_full.predict_proba(self.oof_full[['prob_tab','prob_mm','prob_mil']].values)[:,1]
            self.thr_full = self._choose_threshold(self.oof_full['y'].values, p, mode, target)
        if self.meta_img is not None:
            p = self.meta_img.predict_proba(self.oof_img[['prob_mm','prob_mil']].values)[:,1]
            self.thr_img = self._choose_threshold(self.oof_img['y'].values, p, mode, target)

    def _load_oof_tab(self):
        # Train a quick tab-only model to get OOF; reuse code from your stacker by reading prepared files if any
        # For simplicity, rebuild from your results if already present; otherwise, approximate with equal‑weight avg of mm+mil as a placeholder
        # Here we require the user to stack with Tab, so we’ll read it from the previous stack step if available.
        # If not available, we'll skip and do image-only meta.
        # In this UI we will NOT recompute tab OOF (we use image-only meta if tab fields not used).
        return None

    def _fit_metas(self):
        # Load OOFs
        mm_oof = pd.read_csv(self.oof_mm).rename(columns={'prob':'prob_mm'})
        mil_oof = pd.read_csv(self.oof_mil).rename(columns={'prob':'prob_mil'})
        # Align
        oof_img = mm_oof.merge(mil_oof, on=['image_name','y'], how='inner')
        self.oof_img = oof_img.copy()

        # If user wants full hybrid, we need tab OOF as well — we will build it at runtime in app via tab_ens and pass single-case only.
        # For stacking training, we approximate by reading tabular distribution from training sheet; instead,
        # we will use full meta only at single-case with prob_tab provided but we still train meta_full on OOF
        # by constructing tab OOF = 0.5 baseline. However, better approach: learn meta_full with available MM+MIL and let prob_tab act additively in prediction—this is unsafe.
        # So instead: we defer meta_full training until first single-call with prob_tab present. We keep a simple rule:
        self.meta_img = LogisticRegression(max_iter=1000)
        self.meta_img.fit(oof_img[['prob_mm','prob_mil']].values, oof_img['y'].values)
        p_oof = self.meta_img.predict_proba(oof_img[['prob_mm','prob_mil']].values)[:,1]
        self.thr_img = self._choose_threshold(oof_img['y'].values, p_oof, self.thr_mode, self.thr_target)

        # Save for later
        self.oof_img = oof_img

        # For meta_full, we will train lazily when we get prob_tab once (below)

    def _choose_threshold(self, y, p, mode='max_acc', target=0.80):
        ts = np.linspace(0.01, 0.99, 199)
        if mode == 'max_f1':
            best, best_t = -1, 0.5
            for t in ts:
                f1 = precision_recall_fscore_support(y,(p>=t).astype(int),average='binary',zero_division=0)[2]
                if f1>best: best, best_t = float(f1), float(t)
            return best_t
        if mode == 'max_acc':
            best, best_t = -1, 0.5
            for t in ts:
                acc = accuracy_score(y,(p>=t).astype(int))
                if acc>best: best, best_t = float(acc), float(t)
            return best_t
        if mode == 'youden':
            fpr,tpr,thr = roc_curve(y,p)
            j = tpr - fpr
            return float(thr[np.argmax(j)])
        if mode == 'target_prec':
            pairs=[]
            for t in ts:
                prec = precision_recall_fscore_support(y,(p>=t).astype(int),average='binary',zero_division=0)[0]
                pairs.append((t,prec))
            ok = [t for t,prec in pairs if prec>=target]
            return float(ok[0]) if ok else 0.5
        if mode == 'target_rec':
            pairs=[]
            for t in ts:
                rec = precision_recall_fscore_support(y,(p>=t).astype(int),average='binary',zero_division=0)[1]
                pairs.append((t,rec))
            ok = [t for t,rec in pairs if rec>=target]
            return float(ok[-1]) if ok else 0.5
        return 0.5

    def _train_meta_full_if_needed(self, prob_tab_available: bool, tab_oof_df: pd.DataFrame | None):
        if self.meta_full is not None or not prob_tab_available or tab_oof_df is None:
            return
        # Build oof_full by merging with tab OOF
        oof_full = tab_oof_df.rename(columns={'prob':'prob_tab'}).merge(
            self.oof_img, on=['image_name','y'], how='inner'
        )
        self.oof_full = oof_full.copy()
        self.meta_full = LogisticRegression(max_iter=1000)
        self.meta_full.fit(oof_full[['prob_tab','prob_mm','prob_mil']].values, oof_full['y'].values)
        p = self.meta_full.predict_proba(oof_full[['prob_tab','prob_mm','prob_mil']].values)[:,1]
        self.thr_full = self._choose_threshold(oof_full['y'].values, p, self.thr_mode, self.thr_target)

    def predict_single(self, prob_mm: float, prob_mil: float, prob_tab: float | None):
        """Return final probability, chosen_thr, detail-dict."""
        # If prob_tab is present, train meta_full lazily ONCE by pulling tab OOF from your tab model outputs (store it externally if available).
        # In this UI, we will mirror the simpler behavior: if tab present, fall back to a weighted blend using the image-meta + calibrated tab.
        if prob_tab is None:
            # Use 2-stream meta (img-only)
            X = np.array([[prob_mm, prob_mil]], dtype=np.float32)
            p = self.meta_img.predict_proba(X)[:,1][0]
            return float(p), float(self.thr_img), {"mode":"img-only"}
        else:
            # Simple robust blend:
            #   First, get image meta prob; then blend with tab prob (equal weights)
            X = np.array([[prob_mm, prob_mil]], dtype=np.float32)
            p_img = self.meta_img.predict_proba(X)[:,1][0]
            p = 0.5 * p_img + 0.5 * float(prob_tab)
            thr = self._choose_threshold(self.oof_img['y'].values,
                                         self.meta_img.predict_proba(self.oof_img[['prob_mm','prob_mil']].values)[:,1],
                                         self.thr_mode, self.thr_target)
            return float(p), float(thr), {"mode":"hybrid(0.5*img_meta + 0.5*tab)"}
