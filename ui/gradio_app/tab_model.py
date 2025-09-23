from pathlib import Path
import numpy as np
import pandas as pd
import contextlib, os, sys
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
try:
    import lightgbm as lgb
    # Suppress LightGBM warnings globally
    lgb.set_verbosity(-1)
except Exception:
    lgb = None
from sklearn.ensemble import GradientBoostingClassifier

# Additional warning suppression for LightGBM
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='lightgbm')
warnings.filterwarnings('ignore', message='.*No further splits with positive gain.*')

TAB_FEATURES = [
    'depth','width','enamel_cracks','occlusal_load','carious_lesion',
    'opposing_type','adjacent_teeth','age_range','cervical_lesion'
]
CONT_FEATURES = ['depth','width']
CAT_FEATURES  = ['enamel_cracks','occlusal_load','carious_lesion',
                 'opposing_type','adjacent_teeth','age_range','cervical_lesion']

class TabEnsemble:
    def __init__(self, sheet_path, folds=5):
        self.path = Path(sheet_path)
        self.folds = folds
        self.models = []
        self.FEATS = TAB_FEATURES.copy()
        self.CAT = CAT_FEATURES.copy()
        self.CONT = CONT_FEATURES.copy()
        self._train_kfold()

    def _train_kfold(self):
        if self.path.suffix.lower() == ".xlsx":
            df = pd.read_excel(self.path)
        else:
            df = pd.read_csv(self.path)

        # Ensure cols exist
        for c in self.FEATS:
            if c not in df.columns:
                df[c] = np.nan

        # Dtypes & fills
        df[self.CONT] = df[self.CONT].astype(float).fillna(df[self.CONT].median(numeric_only=True))
        for c in self.CAT:
            df[c] = df[c].fillna(-1).astype('int64').astype('category')

        # Drop constant
        nunq = {c:int(df[c].nunique()) for c in self.FEATS}
        const_cols = [c for c,n in nunq.items() if n<=1]
        self.FEATS = [c for c in self.FEATS if c not in const_cols]
        self.CONT  = [c for c in self.CONT  if c not in const_cols]
        self.CAT   = [c for c in self.CAT   if c not in const_cols]

        df_tv = df[df['split'].isin(['train','val'])].reset_index(drop=True)
        groups = df_tv.get('origin_id', df_tv.get('image_id', df_tv.index.values))

        gkf = GroupKFold(n_splits=self.folds)
        idxs = list(gkf.split(df_tv, groups=groups))

        for fold, (tr,va) in enumerate(idxs):
            tr_df = df_tv.iloc[tr]; va_df = df_tv.iloc[va]
            Xtr, ytr = tr_df[self.FEATS], tr_df['y_majority'].values
            Xva, yva = va_df[self.FEATS], va_df['y_majority'].values

            if lgb is not None:
                params = dict(
                    objective='binary', learning_rate=0.03, n_estimators=700,
                    num_leaves=31, subsample=0.85, colsample_bytree=0.85,
                    min_data_in_leaf=5, class_weight='balanced', random_state=42, n_jobs=-1,
                    verbosity=-1  # Suppress all LightGBM output
                )
                model = lgb.LGBMClassifier(**params)
                try:
                    # Suppress callback logging as well
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        model.fit(
                            Xtr, ytr,
                            eval_set=[(Xva, yva)],
                            eval_metric='auc',
                            categorical_feature=self.CAT,
                            callbacks=[lgb.log_evaluation(period=0)]  # No logging
                        )
                except TypeError:
                    model.set_params(verbosity=-1)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        model.fit(
                            Xtr, ytr,
                            eval_set=[(Xva, yva)],
                            eval_metric='auc',
                            categorical_feature=self.CAT
                        )
            else:
                # Fallback
                model = GradientBoostingClassifier(random_state=42)
                Xtr = pd.get_dummies(Xtr, columns=self.CAT)
                Xva = pd.get_dummies(Xva, columns=self.CAT)
                model.fit(Xtr, ytr)
            self.models.append(model)

    def predict_one(self, tab_dict: dict) -> float:
        # Validate full fields
        x = {}
        for c in TAB_FEATURES:
            if c not in tab_dict:
                raise ValueError(f"Missing feature: {c}")
            x[c] = tab_dict[c]
        # Build row
        row = {c: np.nan for c in TAB_FEATURES}
        row.update(x)
        df = pd.DataFrame([row])
        # same preprocessing
        df[self.CONT] = df[self.CONT].astype(float)
        for c in self.CAT:
            df[c] = df[c].astype('int64').astype('category')

        probs = []
        for m in self.models:
            if lgb is not None:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    probs.append(m.predict_proba(df[self.FEATS])[:,1][0])
            else:
                probs.append(m.predict_proba(pd.get_dummies(df[self.FEATS], columns=self.CAT))[:,1][0])
        return float(np.mean(probs))
