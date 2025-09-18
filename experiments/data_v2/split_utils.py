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
    tol: float        = 0.02           # not hard enforced; used in reporting
    max_trials: int   = 400            # search tries for better balance


def _check_fracs(cfg: SplitConfig):
    s = cfg.train_frac + cfg.val_frac + cfg.test_frac
    if abs(s - 1.0) > 1e-6:
        raise ValueError(f"Fractions must sum to 1.0 (got {s})")


def _pos_rate(df: pd.DataFrame, label_col: str) -> float:
    if len(df) == 0: return 0.0
    return float(df[label_col].astype(int).mean())


def _score_balance(df_tr: pd.DataFrame, df_va: pd.DataFrame, df_te: pd.DataFrame,
                   cfg: SplitConfig) -> float:
    """
    Lower is better. Combines (a) split size error vs target and (b) pos_rate deviation.
    """
    n = len(df_tr) + len(df_va) + len(df_te)
    tgt_sizes = np.array([cfg.train_frac, cfg.val_frac, cfg.test_frac]) * n
    got_sizes = np.array([len(df_tr), len(df_va), len(df_te)])

    size_err = np.abs(got_sizes - tgt_sizes) / (n + 1e-6)

    p_tr, p_va, p_te = _pos_rate(df_tr, cfg.label_col), _pos_rate(df_va, cfg.label_col), _pos_rate(df_te, cfg.label_col)
    p_all = _pos_rate(pd.concat([df_tr, df_va, df_te], ignore_index=True), cfg.label_col)
    pr_err = np.array([abs(p_tr - p_all), abs(p_va - p_all), abs(p_te - p_all)])

    # emphasize val/test balance a bit more
    weights = np.array([0.6, 1.0, 1.0])
    score = (size_err * weights).sum() + 0.75 * (pr_err * weights).sum()
    return float(score)


def _grouped_split_search(df: pd.DataFrame, cfg: SplitConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Search over many grouped splits to pick the one with best (row-level) size & pos_rate balance.
    Always enforces group exclusivity by cfg.group_col.
    """
    gcol = cfg.group_col
    ycol = cfg.label_col

    if gcol not in df.columns:
        # Fallback to row-level stratified split (no leakage control)
        return stratified_row_split(df, cfg)

    best = None
    best_score = 1e9

    # We first pick TEST groups, then VAL groups from remaining, per trial
    for t in range(cfg.max_trials):
        rnd = cfg.seed + t
        # TEST
        gss_test = GroupShuffleSplit(n_splits=1, test_size=cfg.test_frac, random_state=rnd)
        idx_trval, idx_test = next(gss_test.split(df, groups=df[gcol].astype(str).values))
        df_trval = df.iloc[idx_trval].reset_index(drop=True)
        df_test  = df.iloc[idx_test].reset_index(drop=True)

        # VAL
        remain_frac = 1.0 - cfg.test_frac
        rel_val_frac = cfg.val_frac / remain_frac if remain_frac > 0 else 0.0
        gss_val = GroupShuffleSplit(n_splits=1, test_size=rel_val_frac, random_state=rnd + 11)
        idx_train, idx_val = next(gss_val.split(df_trval, groups=df_trval[gcol].astype(str).values))
        df_train = df_trval.iloc[idx_train].reset_index(drop=True)
        df_val   = df_trval.iloc[idx_val].reset_index(drop=True)

        sc = _score_balance(df_train, df_val, df_test, cfg)
        if sc < best_score:
            best_score = sc
            best = (df_train, df_val, df_test)

    return best


def stratified_group_split(
    df: pd.DataFrame,
    cfg: SplitConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Public API: grouped, leakage-safe, balanced split (search-based).
    """
    _check_fracs(cfg)
    df = df.copy()
    if cfg.label_col not in df.columns:
        raise KeyError(f"label_col '{cfg.label_col}' not found in dataframe")

    # Force integer labels
    df[cfg.label_col] = df[cfg.label_col].astype(int)

    tr, va, te = _grouped_split_search(df, cfg)
    return tr, va, te


def stratified_row_split(
    df: pd.DataFrame,
    cfg: SplitConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Row-level (non-grouped) stratified split on cfg.label_col.
    """
    _check_fracs(cfg)
    df = df.copy()
    df[cfg.label_col] = df[cfg.label_col].astype(int)
    y = df[cfg.label_col].values

    sss_test = StratifiedShuffleSplit(n_splits=1, test_size=cfg.test_frac, random_state=cfg.seed)
    idx_trval, idx_test = next(sss_test.split(np.zeros(len(df)), y))
    trval = df.iloc[idx_trval].reset_index(drop=True)
    test  = df.iloc[idx_test].reset_index(drop=True)

    remain_frac = 1.0 - cfg.test_frac
    rel_val_frac = cfg.val_frac / remain_frac if remain_frac > 0 else 0.0
    y_trval = trval[cfg.label_col].values
    sss_val = StratifiedShuffleSplit(n_splits=1, test_size=rel_val_frac, random_state=cfg.seed + 1)
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
    Aug rows inherit split from base rows. Priority:
      1) Join on group_col
      2) If parent_col_aug exists, join on parent image
      3) Heuristic: derive parent stem from image_name and join on that
    """
    out = df_aug.copy()
    if "split" in out.columns:
        out = out.drop(columns=["split"])

    # Case 1: match on group
    if (group_col in out.columns) and (group_col in df_base.columns):
        m = df_base[[group_col, "split"]].drop_duplicates()
        out = out.merge(m, on=group_col, how="left")
        if out["split"].notna().all():
            return out

    # Case 2: match on explicit parent
    if parent_col_aug and (parent_col_aug in out.columns):
        parent_map = df_base[[image_col_base, "split"]].rename(
            columns={image_col_base: parent_col_aug}
        )
        out = out.merge(parent_map, on=parent_col_aug, how="left")
        if out["split"].notna().all():
            return out

    # Case 3: heuristic stem
    def _derive_parent(name: str) -> str:
        stem = str(name)
        for key in ["__aug", "_aug", "__AUG", "_AUG"]:
            if key in stem:
                stem = stem.split(key)[0]
        return stem

    base_map = df_base.copy()
    base_map["__parent_stem"] = base_map[image_col_base].astype(str).map(_derive_parent)
    out["__parent_stem"] = out[image_col_aug].astype(str).map(_derive_parent)

    j = out.merge(
        base_map[["__parent_stem", "split"]].drop_duplicates(),
        on="__parent_stem", how="left"
    )
    if "split_x" in j.columns and "split_y" in j.columns:
        j["split"] = j["split_x"].fillna(j["split_y"])
        j = j.drop(columns=["split_x", "split_y"])

    # Final fallback: any remaining NaNs -> train
    if j["split"].isna().any():
        j["split"] = j["split"].fillna("train")

    return j.drop(columns="__parent_stem", errors="ignore")


def audit_report(df: pd.DataFrame, label_col: str = "y_majority") -> Dict[str, Dict[str, float]]:
    """
    Per-split counts & positive rates.
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
