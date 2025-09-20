# scripts/make_group_splits.py
# Requires scikit-learn >= 1.1 (for StratifiedGroupKFold)

import argparse, os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit

def main(args):
    df = pd.read_csv(args.csv) if args.csv.endswith(".csv") else pd.read_excel(args.csv)

    # Columns we expect
    assert "origin_id" in df.columns, "origin_id column missing"
    # Use your binary hard label
    label_col = args.label
    assert label_col in df.columns, f"{label_col} column missing"
    # Image column
    img_col = args.image_col if args.image_col in df.columns else "image_name"
    assert img_col in df.columns, f"{img_col} column missing"

    # Optional: drop any prior split column
    if "split" in df.columns:
        df = df.drop(columns=["split"])

    groups = df["origin_id"].values
    y = df[label_col].astype(int).values

    # First create a clean **test** split by groups
    if args.test_prop > 0:
        gss = GroupShuffleSplit(n_splits=1, test_size=args.test_prop, random_state=args.seed)
        train_val_groups_idx, test_groups_idx = next(gss.split(np.zeros_like(groups), y, groups))
        keep_train_val_groups = set(groups[train_val_groups_idx])
        keep_test_groups = set(groups[test_groups_idx])

        df["split"] = np.where(df["origin_id"].isin(keep_test_groups), "test", "pool")
        df_pool = df[df["split"] == "pool"].copy()
    else:
        df["split"] = "pool"
        df_pool = df.copy()

    # Now build K folds (val folds) on remaining pool with group-stratification
    sgkf = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    folds = np.full(len(df_pool), -1, dtype=int)

    for fold_id, (_, val_idx) in enumerate(sgkf.split(df_pool, df_pool[label_col].values, df_pool["origin_id"].values)):
        folds[val_idx] = fold_id

    assert (folds >= 0).all(), "some pool rows not assigned to a fold"

    df_pool["fold"] = folds
    df = df.merge(df_pool[[img_col, "fold"]], on=img_col, how="left")

    # Mark train/val for fold 0 by default for convenience
    df.loc[df["split"] == "pool", "split"] = "train"
    df.loc[(df["split"] == "train") & (df["fold"] == 0), "split"] = "val"

    os.makedirs(args.outdir, exist_ok=True)
    all_path = os.path.join(args.outdir, "folds_group.csv")
    df.to_csv(all_path, index=False)
    print(f"Saved {all_path}")

    # Also export per-fold CSVs (train/val) using the pool set
    for k in range(args.folds):
        tr = df[(df["fold"] != k) & (df["split"] != "test")].copy()
        va = df[(df["fold"] == k) & (df["split"] != "test")].copy()
        tr.to_csv(os.path.join(args.outdir, f"train_fold{k}.csv"), index=False)
        va.to_csv(os.path.join(args.outdir, f"val_fold{k}.csv"), index=False)

    if (df["split"] == "test").any():
        df[df["split"] == "test"].to_csv(os.path.join(args.outdir, "test.csv"), index=False)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, required=True,
                   help="Path to data_dl_augmented.csv (or .xlsx). Must have origin_id, image_name, y_majority.")
    p.add_argument("--label", type=str, default="y_majority")
    p.add_argument("--image-col", type=str, default="image_name")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--test-prop", type=float, default=0.2, help="Groupwise test proportion. 0 to skip test holdout.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--outdir", type=str, default="data/splits_grouped")
    args = p.parse_args()
    main(args)
