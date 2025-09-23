# experiments/fusion_v1/stack_blend.py
"""
Late-fusion stacking of:
  - End-to-end multimodal joint model (mm_dualtask_v1)
  - MIL attention model (optional)
  - Fresh LightGBM on raw tabular features

Why this helps:
- Each base learner captures different biases (global shape, local defects, pure tabular cues).
- A simple meta-learner on OOF predictions often yields a sizable lift and better calibration.

Usage (MM + MIL + Tab):
python experiments/fusion_v1/stack_blend.py \
  --xlsx_tab data/excel/data_processed.xlsx \
  --oof_mm   weights/mm_dualtask_v1/finalized/oof_val.csv \
  --pred_mm  weights/mm_dualtask_v1/finalized/pred_test.csv \
  --oof_mil  weights/mil_v1/oof_val.csv \
  --pred_mil weights/mil_v1/pred_test.csv \
  --outdir   results/stack_v2 \
  --thr-mode youden

Usage (MM + Tab only):
  (omit --oof_mil / --pred_mil)
"""

import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support, roc_curve
from sklearn.linear_model import LogisticRegression

try:
    import lightgbm as lgb
except Exception:
    lgb = None  # will fallback to GradientBoostingClassifier

# ----------------------------- Metrics & helpers -----------------------------

def _metrics(y, p, thr=0.5):
    yhat = (p >= thr).astype(int)
    auc = roc_auc_score(y, p) if len(np.unique(y)) > 1 else float('nan')
    acc = accuracy_score(y, yhat)
    prec, rec, f1, _ = precision_recall_fscore_support(y, yhat, average='binary', zero_division=0)
    r = lambda z: float(np.round(z, 4))
    return dict(auc=r(auc), acc=r(acc), prec=r(prec), rec=r(rec), f1=r(f1))

def choose_threshold(y, p, mode='max_f1', target=0.80):
    ts = np.linspace(0.01, 0.99, 199)
    if mode == 'max_f1':
        best_t, best = 0.5, -1
        for t in ts:
            f1 = precision_recall_fscore_support(y, (p >= t).astype(int),
                                                 average='binary', zero_division=0)[2]
            if f1 > best: best, best_t = float(f1), float(t)
        return best_t
    if mode == 'max_acc':
        best_t, best = 0.5, -1
        for t in ts:
            acc = accuracy_score(y, (p >= t).astype(int))
            if acc > best: best, best_t = float(acc), float(t)
        return best_t
    if mode == 'youden':
        fpr, tpr, thr = roc_curve(y, p)
        j = tpr - fpr
        return float(thr[np.argmax(j)])
    if mode == 'target_prec':
        cands = []
        for t in ts:
            prec = precision_recall_fscore_support(y, (p >= t).astype(int),
                                                   average='binary', zero_division=0)[0]
            cands.append((t, prec))
        ok = [t for (t, prec) in cands if prec >= target]
        return float(ok[0]) if ok else 0.5
    if mode == 'target_rec':
        cands = []
        for t in ts:
            rec = precision_recall_fscore_support(y, (p >= t).astype(int),
                                                  average='binary', zero_division=0)[1]
            cands.append((t, rec))
        ok = [t for (t, rec) in cands if rec >= target]
        return float(ok[-1]) if ok else 0.5
    return 0.5

# ----------------------------- Tabular OOF builder ---------------------------

# Treat low-cardinality features as categorical; drop constant cols; no scaling.
CONT_ALL = ['depth', 'width']  # numeric candidates
CAT_ALL  = ['enamel_cracks','occlusal_load','carious_lesion',
            'opposing_type','adjacent_teeth','age_range','cervical_lesion']

def fit_tab_oof(df_tab: pd.DataFrame, folds=5):
    from sklearn.ensemble import GradientBoostingClassifier
    assert 'split' in df_tab.columns and 'y_majority' in df_tab.columns

    df = df_tab.copy()
    # ensure all columns present
    for c in CONT_ALL + CAT_ALL:
        if c not in df.columns:
            df[c] = np.nan

    # fill & dtypes
    df[CONT_ALL] = df[CONT_ALL].astype(float)
    df[CONT_ALL] = df[CONT_ALL].fillna(df[CONT_ALL].median(numeric_only=True))
    for c in CAT_ALL:
        df[c] = df[c].fillna(-1).astype('int64').astype('category')

    # drop constant features
    nunq = {c: int(df[c].nunique()) for c in CONT_ALL + CAT_ALL}
    const_cols = [c for c, n in nunq.items() if n <= 1]
    if const_cols:
        print("[INFO] Dropping constant features:", const_cols)
    CONT = [c for c in CONT_ALL if c not in const_cols]
    CAT  = [c for c in CAT_ALL  if c not in const_cols]
    FEATS = CONT + CAT
    print("Feature nunique (post-drop):", {c: nunq[c] for c in FEATS})

    df_tv = df[df['split'].isin(['train', 'val'])].reset_index(drop=True)
    df_te = df[df['split'] == 'test'].reset_index(drop=True)
    groups = df_tv.get('origin_id', df_tv.get('image_id', df_tv.index.values))

    gkf = GroupKFold(n_splits=folds)
    idxs = list(gkf.split(df_tv, groups=groups))

    X_te = df_te[FEATS]
    y_te = df_te['y_majority'].values

    oof = np.zeros(len(df_tv))
    test_pred = np.zeros(len(df_te))

    use_lgb = (lgb is not None)
    lgb_params = dict(
        objective='binary',
        learning_rate=0.03,
        n_estimators=700,
        num_leaves=31,
        max_depth=-1,
        subsample=0.85,
        colsample_bytree=0.85,
        min_data_in_leaf=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )

    for fold, (tr, va) in enumerate(idxs):
        tr_df = df_tv.iloc[tr]
        va_df = df_tv.iloc[va]

        X_tr, y_tr = tr_df[FEATS], tr_df['y_majority'].values
        X_va, y_va = va_df[FEATS], va_df['y_majority'].values

        if use_lgb:
            model = lgb.LGBMClassifier(**lgb_params)
            # silence logs via callbacks; fallback if not supported
            try:
                callbacks = [lgb.log_evaluation(period=0)]
                model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_va, y_va)],
                    eval_metric='auc',
                    categorical_feature=CAT,
                    callbacks=callbacks
                )
            except TypeError:
                model.set_params(verbosity=-1)
                model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_va, y_va)],
                    eval_metric='auc',
                    categorical_feature=CAT
                )
        else:
            # sklearn fallback
            model = GradientBoostingClassifier(random_state=42)
            model.fit(pd.get_dummies(X_tr, columns=CAT), y_tr)
            oof[va] = model.predict_proba(pd.get_dummies(X_va, columns=CAT))[:, 1]
            test_pred += model.predict_proba(pd.get_dummies(X_te, columns=CAT))[:, 1] / folds
            continue

        oof[va] = model.predict_proba(X_va)[:, 1]
        test_pred += model.predict_proba(X_te)[:, 1] / folds

    oof_df = df_tv[['image_name', 'y_majority']].rename(columns={'y_majority': 'y'}).copy()
    oof_df['prob'] = oof
    te_df  = df_te[['image_name', 'y_majority']].rename(columns={'y_majority': 'y'}).copy()
    te_df['prob'] = test_pred
    return oof_df, te_df

# ----------------------------------- Main ------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--xlsx_tab', type=str, required=True,
                    help='Path to data_processed.xlsx (or CSV)')
    ap.add_argument('--oof_mm',  type=str, required=True,
                    help='OOF CSV from mm_dualtask (oof_val.csv)')
    ap.add_argument('--pred_mm', type=str, required=True,
                    help='TEST CSV from mm_dualtask (pred_test.csv)')
    ap.add_argument('--oof_mil',  type=str, default='',
                    help='(optional) OOF CSV from MIL (oof_val.csv)')
    ap.add_argument('--pred_mil', type=str, default='',
                    help='(optional) TEST CSV from MIL (pred_test.csv)')
    ap.add_argument('--outdir', type=str, required=True)

    # thresholding controls
    ap.add_argument('--thr-mode', default='youden',
                    choices=['max_f1', 'max_acc', 'youden', 'target_prec', 'target_rec'])
    ap.add_argument('--thr-target', type=float, default=0.80,
                    help='Used for target_prec/target_rec')

    ap.add_argument('--folds', type=int, default=5)
    args = ap.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    # Load tabular data
    if args.xlsx_tab.endswith('.xlsx'):
        df_tab = pd.read_excel(args.xlsx_tab)
    else:
        df_tab = pd.read_csv(args.xlsx_tab)

    if 'origin_id' not in df_tab.columns:
        df_tab['origin_id'] = df_tab.get('image_id', df_tab.index)

    # 1) Build tabular OOF/TEST
    tab_oof, tab_test = fit_tab_oof(df_tab, folds=args.folds)
    tab_oof = tab_oof.rename(columns={'prob': 'prob_tab'})
    tab_test = tab_test.rename(columns={'prob': 'prob_tab'})

    # 2) Load MM OOF/TEST
    mm_oof = pd.read_csv(args.oof_mm).rename(columns={'prob': 'prob_mm'})
    mm_te  = pd.read_csv(args.pred_mm).rename(columns={'prob': 'prob_mm'})

    # 3) Optional MIL
    use_mil = bool(args.oof_mil.strip()) and bool(args.pred_mil.strip())
    if use_mil:
        mil_oof = pd.read_csv(args.oof_mil).rename(columns={'prob': 'prob_mil'})
        mil_te  = pd.read_csv(args.pred_mil).rename(columns={'prob': 'prob_mil'})

    # 4) Merge
    oof = tab_oof.merge(mm_oof, on=['image_name', 'y'], how='inner')
    test = tab_test.merge(mm_te, on=['image_name', 'y'], how='inner')
    if use_mil:
        oof = oof.merge(mil_oof, on=['image_name', 'y'], how='inner')
        test = test.merge(mil_te, on=['image_name', 'y'], how='inner')

    feat_cols = ['prob_tab', 'prob_mm'] + (['prob_mil'] if use_mil else [])
    X_oof = oof[feat_cols].values
    y_oof = oof['y'].values

    # 5) Meta-learner
    meta = LogisticRegression(max_iter=1000)  # try class_weight='balanced' if desired
    meta.fit(X_oof, y_oof)
    p_oof = meta.predict_proba(X_oof)[:, 1]

    # 6) Threshold selection on OOF
    thr = choose_threshold(y_oof, p_oof, mode=args.thr_mode, target=args.thr_target)

    # 7) Apply to TEST
    X_te = test[feat_cols].values
    p_te = meta.predict_proba(X_te)[:, 1]
    m_oof = _metrics(y_oof, p_oof, thr)
    m_te  = _metrics(test['y'].values, p_te, thr)

    # 8) Save
    oof_out = oof[['image_name', 'y']].copy(); oof_out['prob'] = p_oof
    te_out  = test[['image_name', 'y']].copy(); te_out['prob'] = p_te
    oof_out.to_csv(Path(args.outdir) / 'stack_oof.csv', index=False)
    te_out.to_csv(Path(args.outdir) / 'stack_test.csv', index=False)

    with open(Path(args.outdir) / 'summary.json', 'w') as f:
        json.dump({
            'oof': m_oof,
            'test': m_te,
            'thr': float(np.round(thr, 4)),
            'thr_mode': args.thr_mode,
            'thr_target': args.thr_target,
            'features': feat_cols
        }, f, indent=2)

    print("Features used:", feat_cols)
    print(f"Threshold mode: {args.thr_mode} | target: {args.thr_target} | chosen thr: {thr:.3f}")
    print("=== OOF ===", m_oof)
    print("=== TEST ===", m_te)

if __name__ == '__main__':
    main()
