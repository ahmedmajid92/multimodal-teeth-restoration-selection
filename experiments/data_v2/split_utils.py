# experiments/data_v2/split_utils.py
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, Dict, Optional

from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit


@dataclass
class SplitConfig:
    train_frac: float = 0.70
    val_frac: float   = 0.15
    test_frac: float  = 0.15
    seed: int         = 42
    group_col: str    = "origin_id"
    label_col: str    = "y_majority"   # binary 0/1 column
    # acceptable deviation from requested ratio (absolute, per split)
    tol: float        = 0.02


def _check_fracs(cfg: SplitConfig):
    s = cfg.train_frac + cfg.val_frac + cfg.test_frac
    if abs(s - 1.0) > 1e-6:
        raise ValueError(f"Fractions must sum to 1.0 (got {s})")


def stratified_group_split(
    df: pd.DataFrame,
    cfg: SplitConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns (train_df, val_df, test_df) with:
      * Group exclusivity by cfg.group_col (no group appears in multiple splits)
      * Approximate label stratification on cfg.label_col (0/1)
    Strategy:
      1) Grouped split to carve out TEST by cfg.test_frac
      2) Grouped split on the remaining to carve out VAL by cfg.val_frac / (1 - test_frac)
    """
    _check_fracs(cfg)
    df = df.copy()

    if cfg.label_col not in df.columns:
        raise KeyError(f"label_col '{cfg.label_col}' not found in dataframe")
    if cfg.group_col not in df.columns:
        # fallback: every row is its own group -> not ideal but safe
        df[cfg.group_col] = np.arange(len(df))
        print(f"⚠️ group_col '{cfg.group_col}' not found; using each row as its own group (no leakage control).")

    y = df[cfg.label_col].astype(int).values
    g = df[cfg.group_col].astype(str).values

    # Step 1: pick TEST groups via grouped split
    gss_test = GroupShuffleSplit(n_splits=1, test_size=cfg.test_frac, random_state=cfg.seed)
    idx_trval, idx_test = next(gss_test.split(df, groups=g))
    df_trval = df.iloc[idx_trval].reset_index(drop=True)
    df_test  = df.iloc[idx_test].reset_index(drop=True)

    # Step 2: on TR+VAL pool, pick VAL groups (relative fraction)
    remain_frac = 1.0 - cfg.test_frac
    rel_val_frac = cfg.val_frac / remain_frac if remain_frac > 0 else 0.0
    gss_val = GroupShuffleSplit(n_splits=1, test_size=rel_val_frac, random_state=cfg.seed)
    g_tr = df_trval[cfg.group_col].values
    idx_train, idx_val = next(gss_val.split(df_trval, groups=g_tr))
    df_train = df_trval.iloc[idx_train].reset_index(drop=True)
    df_val   = df_trval.iloc[idx_val].reset_index(drop=True)

    return df_train, df_val, df_test


def stratified_row_split(
    df: pd.DataFrame,
    cfg: SplitConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Row-level (non-grouped) stratified split on cfg.label_col.
    Only used if group_col is absent and you still want proportions.
    """
    _check_fracs(cfg)
    df = df.copy()
    y = df[cfg.label_col].astype(int).values

    # Step 1: hold-out TEST
    sss_test = StratifiedShuffleSplit(n_splits=1, test_size=cfg.test_frac, random_state=cfg.seed)
    idx_trval, idx_test = next(sss_test.split(np.zeros(len(df)), y))
    trval = df.iloc[idx_trval].reset_index(drop=True)
    test  = df.iloc[idx_test].reset_index(drop=True)

    # Step 2: hold-out VAL from the remaining
    remain_frac = 1.0 - cfg.test_frac
    rel_val_frac = cfg.val_frac / remain_frac if remain_frac > 0 else 0.0
    y_trval = trval[cfg.label_col].astype(int).values
    sss_val = StratifiedShuffleSplit(n_splits=1, test_size=rel_val_frac, random_state=cfg.seed)
    idx_train, idx_val = next(sss_val.split(np.zeros(len(trval)), y_trval))
    train = trval.iloc[idx_train].reset_index(drop=True)
    val   = trval.iloc[idx_val].reset_index(drop=True)
    return train, val, test


def propagate_split_to_augmented(
    df_aug: pd.DataFrame,
    df_base: pd.DataFrame,
    group_col: str = "origin_id",
    image_col_aug: str = "image_name",
    image_col_base: str = "image_name",
    parent_col_aug: Optional[str] = None,
) -> pd.DataFrame:
    """
    Propagate train/val/test splits from base data to augmented data.
    Ensures no group appears in multiple splits (zero leakage).
    """
    df_result = df_aug.copy()
    
    # Case 1: Direct group-based merge (preferred when group_col exists in both)
    if group_col and group_col in df_base.columns and group_col in df_aug.columns:
        print(f"✅ Using direct group-based merge on '{group_col}'")
        
        # Create lookup table from base data
        merge_cols = [group_col, "split"]
        if "y_majority" in df_base.columns:
            merge_cols.append("y_majority")
        
        base_lookup = df_base[merge_cols].drop_duplicates()
        
        # Merge on group column
        df_result = df_result.merge(base_lookup, on=group_col, how="left", suffixes=('', '_base'))
        
        # Handle conflicts: if augmented data already has split/label, keep base version
        if "split_base" in df_result.columns:
            df_result["split"] = df_result["split_base"].fillna(df_result.get("split", "train"))
            df_result = df_result.drop(columns=["split_base"])
        
        # Fill missing splits with 'train' as fallback
        df_result["split"] = df_result["split"].fillna("train")
        return df_result
    
    # Case 2: Image name-based merge (fallback when no group column match)
    elif image_col_base in df_base.columns and image_col_aug in df_aug.columns:
        print(f"⚠️  Falling back to image name-based merge")
        
        def _derive_parent(img_name: str) -> str:
            """Extract parent image name from augmented image name"""
            if not isinstance(img_name, str):
                return str(img_name)
            # Remove common augmentation suffixes
            for suffix in ["_aug", "_flip", "_rot", "_bright", "_contrast", "_noise"]:
                if suffix in img_name:
                    return img_name.split(suffix)[0] + ".jpg"
            return img_name
        
        # Create mapping from base data
        base_map = df_base[[image_col_base, "split"]].copy()
        if "y_majority" in df_base.columns:
            base_map["y_majority"] = df_base["y_majority"]
            
        base_map["__parent_stem"] = base_map[image_col_base].astype(str).map(_derive_parent)
        
        # Create mapping for augmented data
        if parent_col_aug and parent_col_aug in df_aug.columns:
            # Use explicit parent column if available
            aug_map = df_aug[[image_col_aug, parent_col_aug]].copy()
            aug_map["__aug_parent_stem"] = aug_map[parent_col_aug].astype(str).map(_derive_parent)
        else:
            # Derive parent from image name
            aug_map = df_aug[[image_col_aug]].copy()
            aug_map["__aug_parent_stem"] = aug_map[image_col_aug].astype(str).map(_derive_parent)
        
        # Merge base splits to augmented data
        merge_df = aug_map.merge(
            base_map.rename(columns={"__parent_stem": "__aug_parent_stem"}),
            on="__aug_parent_stem", how="left"
        )
        
        # Update result with inherited splits
        df_result = df_result.merge(
            merge_df[[image_col_aug, "split"] + (["y_majority"] if "y_majority" in merge_df.columns else [])],
            on=image_col_aug, how="left", suffixes=('', '_inherited')
        )
        
        # Handle conflicts
        if "split_inherited" in df_result.columns:
            df_result["split"] = df_result["split_inherited"].fillna(df_result.get("split", "train"))
            df_result = df_result.drop(columns=["split_inherited"])
        
        # Fill missing splits
        df_result["split"] = df_result["split"].fillna("train")
        return df_result
    
    # Case 3: No valid merge columns
    else:
        print(f"⚠️  Cannot find valid columns for split propagation, assigning all to 'train'")
        df_result["split"] = "train"
        return df_result


def audit_report(df: pd.DataFrame, label_col: str = "y_majority") -> Dict[str, Dict[str, float]]:
    """
    Returns a nested dict with per-split counts, class balances and positive rates.
    """
    out = {}
    for s in ["train", "val", "test"]:
        d = df[df["split"].astype(str).str.lower() == s]
        n = len(d)
        if n == 0:
            out[s] = {"rows": 0}
            continue
        vc = d[label_col].astype(int).value_counts().sort_index()
        neg, pos = int(vc.get(0, 0)), int(vc.get(1, 0))
        out[s] = {
            "rows": n,
            "neg": neg,
            "pos": pos,
            "pos_rate": round(pos / n, 4),
        }
    return out
