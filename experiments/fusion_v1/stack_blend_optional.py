import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
try:
    import lightgbm as lgb
except Exception:
    lgb = None

TAB_FEATURES = ['depth','width','enamel_cracks','occlusal_load','carious_lesion',
                'opposing_type','adjacent_teeth','age_range','cervical_lesion']

def _metrics(y,p,thr):
    yhat=(p>=thr).astype(int)
    auc=roc_auc_score(y,p) if len(np.unique(y))>1 else float('nan')
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    acc=accuracy_score(y,yhat)
    pr,rc,f1,_=precision_recall_fscore_support(y,yhat,average='binary',zero_division=0)
    r=lambda z: float(np.round(z,4))
    return dict(auc=r(auc),acc=r(acc),prec=r(pr),rec=r(rc),f1=r(f1))

def fit_tab_oof(df_tab, folds=5):
    import lightgbm as lgb
    from sklearn.model_selection import GroupKFold
    import numpy as np
    import pandas as pd

    # Define your columns
    CONT_ALL = ['depth', 'width']  # numeric candidates
    CAT_ALL  = ['enamel_cracks','occlusal_load','carious_lesion',
                'opposing_type','adjacent_teeth','age_range','cervical_lesion']

    # Ensure columns exist; fill missing with safe defaults
    df = df_tab.copy()
    for c in CONT_ALL + CAT_ALL:
        if c not in df.columns:
            df[c] = np.nan

    # Basic cleaning
    df[CONT_ALL] = df[CONT_ALL].astype(float).fillna(df[CONT_ALL].median(numeric_only=True))
    for c in CAT_ALL:
        df[c] = df[c].fillna(-1).astype('int64').astype('category')

    # Drop constant features (nunique <=1) BEFORE training
    nunq = {c:int(df[c].nunique()) for c in CONT_ALL + CAT_ALL}
    const_cols = [c for c,n in nunq.items() if n <= 1]
    if const_cols:
        print("[INFO] Dropping constant features:", const_cols)
    CONT = [c for c in CONT_ALL if c not in const_cols]
    CAT  = [c for c in CAT_ALL  if c not in const_cols]
    FEATS = CONT + CAT
    print("Feature nunique (post-drop):", {c:nunq[c] for c in FEATS})

    # Split sets
    df_tv = df[df['split'].isin(['train','val'])].reset_index(drop=True)
    df_te = df[df['split']=='test'].reset_index(drop=True)
    groups = df_tv.get('origin_id', df_tv.get('image_id', df_tv.index.values))

    gkf = GroupKFold(n_splits=folds)
    idxs = list(gkf.split(df_tv, groups=groups))

    X_te = df_te[FEATS]
    y_te = df_te['y_majority'].values

    oof = np.zeros(len(df_tv))
    test_pred = np.zeros(len(df_te))

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
        verbosity=-1,     # silence internal logs
    )

    for fold, (tr, va) in enumerate(idxs):
        tr_df = df_tv.iloc[tr]
        va_df = df_tv.iloc[va]

        X_tr, y_tr = tr_df[FEATS], tr_df['y_majority'].values
        X_va, y_va = va_df[FEATS], va_df['y_majority'].values

        model = lgb.LGBMClassifier(**lgb_params)

        # Prefer callbacks to silence logs; fall back if not supported
        try:
            callbacks = [lgb.log_evaluation(period=0)]
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_va, y_va)],
                eval_metric='auc',
                categorical_feature=CAT,   # pandas 'category' dtype + names
                callbacks=callbacks
            )
        except TypeError:
            # Older wrapper: no callbacks arg
            model.set_params(verbosity=-1)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_va, y_va)],
                eval_metric='auc',
                categorical_feature=CAT
            )

        oof[va] = model.predict_proba(X_va)[:, 1]
        test_pred += model.predict_proba(X_te)[:, 1] / folds

    oof_df = df_tv[['image_name','y_majority']].rename(columns={'y_majority':'y'}).copy()
    oof_df['prob'] = oof
    te_df  = df_te[['image_name','y_majority']].rename(columns={'y_majority':'y'}).copy()
    te_df['prob'] = test_pred
    return oof_df, te_df

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--xlsx_tab', required=True, help='data_processed.xlsx')
    ap.add_argument('--oof_mm', required=True)
    ap.add_argument('--pred_mm', required=True)
    ap.add_argument('--oof_mil', default='', help='optional')
    ap.add_argument('--pred_mil', default='', help='optional')
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--folds', type=int, default=5)
    args=ap.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    df_tab = pd.read_excel(args.xlsx_tab) if args.xlsx_tab.endswith('.xlsx') else pd.read_csv(args.xlsx_tab)
    if 'origin_id' not in df_tab.columns: df_tab['origin_id']=df_tab.get('image_id', df_tab.index)

    # 1) Build tabular OOF/TEST
    tab_oof, tab_test = fit_tab_oof(df_tab, folds=args.folds); tab_oof=tab_oof.rename(columns={'prob':'prob_tab'}); tab_test=tab_test.rename(columns={'prob':'prob_tab'})
    # 2) Load MM
    mm_oof=pd.read_csv(args.oof_mm).rename(columns={'prob':'prob_mm'})
    mm_te =pd.read_csv(args.pred_mm).rename(columns={'prob':'prob_mm'})

    # 3) Optional MIL
    use_mil = (args.oof_mil.strip()!='' and args.pred_mil.strip()!='')
    if use_mil:
        mil_oof=pd.read_csv(args.oof_mil).rename(columns={'prob':'prob_mil'})
        mil_te =pd.read_csv(args.pred_mil).rename(columns={'prob':'prob_mil'})

    # 4) Merge
    oof = tab_oof.merge(mm_oof, on=['image_name','y'], how='inner')
    test= tab_test.merge(mm_te, on=['image_name','y'], how='inner')
    if use_mil:
        oof = oof.merge(mil_oof, on=['image_name','y'], how='inner')
        test= test.merge(mil_te,  on=['image_name','y'], how='inner')

    feat_cols = ['prob_tab','prob_mm'] + (['prob_mil'] if use_mil else [])
    X_oof=oof[feat_cols].values; y_oof=oof['y'].values
    meta=LogisticRegression(max_iter=1000, class_weight=None)  # you can try 'balanced'
    meta.fit(X_oof,y_oof)
    p_oof=meta.predict_proba(X_oof)[:,1]

    # ---- threshold selection helpers ----
    def choose_threshold(y, p, mode='max_f1', target=0.80):
        from sklearn.metrics import precision_recall_fscore_support, roc_curve, accuracy_score
        ts = np.linspace(0.05, 0.95, 181)
        if mode == 'max_f1':
            best_t, best = 0.5, -1
            for t in ts:
                f1 = precision_recall_fscore_support(y,(p>=t).astype(int),average='binary',zero_division=0)[2]
                if f1>best: best, best_t = float(f1), float(t)
            return best_t
        if mode == 'max_acc':
            best_t, best = 0.5, -1
            for t in ts:
                acc = accuracy_score(y,(p>=t).astype(int))
                if acc>best: best, best_t = float(acc), float(t)
            return best_t
        if mode == 'youden':  # maximize TPR - FPR
            fpr, tpr, thr = roc_curve(y, p)
            j = tpr - fpr
            return float(thr[np.argmax(j)])
        if mode == 'target_prec':
            # smallest t with precision >= target
            cand = []
            for t in ts:
                prec = precision_recall_fscore_support(y,(p>=t).astype(int),average='binary',zero_division=0)[0]
                cand.append((t,prec))
            ok = [t for (t,prec) in cand if prec>=target]
            return float(ok[0]) if ok else 0.5
        if mode == 'target_rec':
            cand = []
            for t in ts:
                rec = precision_recall_fscore_support(y,(p>=t).astype(int),average='binary',zero_division=0)[1]
                cand.append((t,rec))
            ok = [t for (t,rec) in cand if rec>=target]
            return float(ok[-1]) if ok else 0.5
        return 0.5

    # choose threshold (CHANGE HERE to your goal)
    thr_mode = 'max_f1'          # options: 'max_f1','max_acc','youden','target_prec','target_rec'
    thr_target = 0.80             # used for target_* modes
    best_thr = choose_threshold(y_oof, p_oof, mode=thr_mode, target=thr_target)

    X_te=test[feat_cols].values; p_te=meta.predict_proba(X_te)[:,1]
    m_oof=_metrics(y_oof,p_oof,best_thr); m_te=_metrics(test['y'].values,p_te,best_thr)

    oof_out=oof[['image_name','y']].copy(); oof_out['prob']=p_oof
    te_out =test[['image_name','y']].copy(); te_out['prob']=p_te
    oof_out.to_csv(Path(args.outdir)/'stack_oof.csv',index=False)
    te_out.to_csv(Path(args.outdir)/'stack_test.csv',index=False)
    with open(Path(args.outdir)/'summary.json','w') as f:
        json.dump({'oof':m_oof,'test':m_te,'thr':best_thr,'thr_mode':thr_mode,'thr_target':thr_target,'features':feat_cols}, f, indent=2)
    print("Features:", feat_cols)
    print("Threshold mode:", thr_mode, "| target:", thr_target, "| chosen thr:", round(best_thr,3))
    print("OOF:", m_oof)
    print("TEST:", m_te)

if __name__=='__main__':
    main()
