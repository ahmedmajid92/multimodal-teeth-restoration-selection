# experiments/data_v2/make_balanced_splits.py
import os, sys, argparse, json
from pathlib import Path
import pandas as pd

# repo path fix
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.data_v2.split_utils import (
    SplitConfig,
    stratified_group_split,
    stratified_row_split,
    propagate_split_to_augmented,
    audit_report,
)

def _ensure_dir(p: str):
    Path(p).parent.mkdir(parents=True, exist_ok=True)

def _save_dual(df: pd.DataFrame, xlsx_path: str, csv_path: str):
    _ensure_dir(xlsx_path); _ensure_dir(csv_path)
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as xw:
        df.to_excel(xw, index=False, sheet_name="data")
    df.to_csv(csv_path, index=False)
    print(f"✅ Saved:\n  • {xlsx_path}\n  • {csv_path}")

def _print_audit(title: str, df: pd.DataFrame, label_col: str):
    rep = audit_report(df, label_col=label_col)
    print(f"\n📋 {title} split audit:")
    print(json.dumps(rep, indent=2))

def _has(df: pd.DataFrame, col: str) -> bool:
    return col in df.columns

def _coerce_label(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    if _has(df, label_col):
        df[label_col] = df[label_col].astype(int)
    else:
        raise KeyError(f"Missing label column '{label_col}'")
    return df

def main():
    ap = argparse.ArgumentParser()
    # Inputs
    ap.add_argument("--raw-xlsx",           default="data/excel/data.xlsx")
    ap.add_argument("--processed-xlsx",     default="data/excel/data_processed.xlsx")
    ap.add_argument("--dl-xlsx",            default="data/excel/data_dl.xlsx")
    ap.add_argument("--dl-aug-xlsx",        default="data/excel/data_dl_augmented.xlsx")

    # Outputs
    ap.add_argument("--processed-xlsx-out", default="data/excel/data_processed.xlsx")
    ap.add_argument("--processed-csv-out",  default="data/excel/data_processed.csv")
    ap.add_argument("--dl-xlsx-out",        default="data/excel/data_dl.xlsx")
    ap.add_argument("--dl-csv-out",         default="data/excel/data_dl.csv")
    ap.add_argument("--dl-aug-xlsx-out",    default="data/excel/data_dl_augmented.xlsx")
    ap.add_argument("--dl-aug-csv-out",     default="data/excel/data_dl_augmented.csv")

    # Split config
    ap.add_argument("--train-frac", type=float, default=0.70)
    ap.add_argument("--val-frac",   type=float, default=0.15)
    ap.add_argument("--test-frac",  type=float, default=0.15)
    ap.add_argument("--seed",       type=int,   default=42)
    ap.add_argument("--group-col",  default="origin_id")
    ap.add_argument("--label-col",  default="y_majority")
    ap.add_argument("--max-trials", type=int,   default=400, help="Search tries for best grouped balance")

    args = ap.parse_args()

    cfg = SplitConfig(
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
        group_col=args.group_col,
        label_col=args.label_col,
        max_trials=args.max_trials,
    )

    print("="*88)
    print("🧮 Balanced Train/Val/Test splits (grouped & stratified, search-based)")
    print(f"  Fractions: train={cfg.train_frac} val={cfg.val_frac} test={cfg.test_frac}  |  seed={cfg.seed}  |  trials={cfg.max_trials}")
    print(f"  group_col='{cfg.group_col}'  label_col='{cfg.label_col}'")
    print("="*88)

    # ------------------ 1) PROCESSED (ML) ------------------
    df_proc = pd.read_excel(args.processed_xlsx, engine="openpyxl")
    df_proc = _coerce_label(df_proc, cfg.label_col)

    if _has(df_proc, cfg.group_col):
        tr, va, te = stratified_group_split(df_proc, cfg)
    else:
        tr, va, te = stratified_row_split(df_proc, cfg)

    df_proc_out = pd.concat([tr.assign(split="train"), va.assign(split="val"), te.assign(split="test")], ignore_index=True)
    _print_audit("Processed (ML)", df_proc_out, cfg.label_col)
    _save_dual(df_proc_out, args.processed_xlsx_out, args.processed_csv_out)

    # ------------------ 2) DL BASE ------------------
    df_dl = pd.read_excel(args.dl_xlsx, engine="openpyxl")
    df_dl = _coerce_label(df_dl, cfg.label_col)

    if _has(df_dl, cfg.group_col):
        tr, va, te = stratified_group_split(df_dl, cfg)
    else:
        tr, va, te = stratified_row_split(df_dl, cfg)

    df_dl_out = pd.concat([tr.assign(split="train"), va.assign(split="val"), te.assign(split="test")], ignore_index=True)
    _print_audit("DL base", df_dl_out, cfg.label_col)
    _save_dual(df_dl_out, args.dl_xlsx_out, args.dl_csv_out)

    # ------------------ 3) DL AUGMENTED ------------------
    df_aug = pd.read_excel(args.dl_aug_xlsx, engine="openpyxl")
    if _has(df_aug, cfg.label_col):
        df_aug[cfg.label_col] = df_aug[cfg.label_col].astype(int)

    # Debug: Check what columns are available
    print(f"📊 DL base columns: {list(df_dl_out.columns)}")
    print(f"📊 DL augmented columns: {list(df_aug.columns)}")

    # Special case: Base has image_name but no origin_id, augmented has both
    if (not _has(df_dl_out, cfg.group_col) and 
        _has(df_aug, cfg.group_col) and 
        "image_name" in df_dl_out.columns):
        
        print(f"⚠️  Base data missing '{cfg.group_col}', creating mapping via augmented data")
        
        # Create mapping from origin_id to original image name using aug_idx=0
        origin_mapping = (
            df_aug[df_aug["aug_idx"] == 0][["origin_id", "image_name"]]
            .drop_duplicates()
            .rename(columns={"image_name": "original_image_name"})
        )
        
        print(f"📊 Created {len(origin_mapping)} origin_id -> image_name mappings")
        
        # Add origin_id to base data
        df_base_with_origin = df_dl_out.merge(
            origin_mapping.rename(columns={"original_image_name": "image_name"}),
            on="image_name",
            how="left"
        )
        
        # Check merge success
        missing_origins = df_base_with_origin["origin_id"].isna().sum()
        if missing_origins > 0:
            print(f"⚠️  {missing_origins} base images couldn't be mapped to origin_id")
        
        # Use origin_id from the enhanced base for propagation
        base_cols = ["origin_id", "split", "y_majority"]
        propagation_base = df_base_with_origin[base_cols].dropna().drop_duplicates()
        
        print(f"📊 Using {len(propagation_base)} origin-based mappings for propagation")
        
        # Now propagate using origin_id directly
        df_aug_out = df_aug.copy()
        df_aug_out = df_aug_out.merge(
            propagation_base.rename(columns={"split": "split_base", "y_majority": "y_majority_base"}),
            on="origin_id",
            how="left"
        )
        
        # Update splits - prioritize base splits over existing ones
        df_aug_out["split"] = df_aug_out["split_base"].fillna(df_aug_out["split"]).fillna("train")
        df_aug_out = df_aug_out.drop(columns=["split_base"], errors="ignore")
        
        # Update labels if needed
        if "y_majority_base" in df_aug_out.columns and cfg.label_col not in df_aug_out.columns:
            df_aug_out[cfg.label_col] = df_aug_out["y_majority_base"]
        df_aug_out = df_aug_out.drop(columns=["y_majority_base"], errors="ignore")
        
    else:
        # Original logic for when columns match properly
        base_cols = ["image_name"]
        if _has(df_dl_out, cfg.group_col):
            base_cols.append(cfg.group_col)
        base_cols.append("split")
        if _has(df_dl_out, cfg.label_col):
            base_cols.append(cfg.label_col)

        print(f"📊 Using base columns for propagation: {base_cols}")

        df_aug_out = propagate_split_to_augmented(
            df_aug=df_aug,
            df_base=df_dl_out[base_cols].copy(),
            group_col=cfg.group_col if _has(df_dl_out, cfg.group_col) else None,
            image_col_aug="image_name",
            image_col_base="image_name",
            parent_col_aug="parent_image" if "parent_image" in df_aug.columns else None,
        )

    # Sanity: ensure no group spans multiple splits (only if we have the group column)
    if cfg.group_col in df_aug_out.columns:
        cross = df_aug_out.groupby(cfg.group_col)["split"].nunique()
        bad_count = (cross > 1).sum()
        
        if bad_count > 0:
            bad_groups = cross[cross > 1].index.tolist()[:5]  # Show fewer for readability
            print(f"❌ ERROR: {bad_count} groups spanning multiple splits (showing first 5): {bad_groups}")
            
            # Debug info for the first problematic group
            problem_group = bad_groups[0]
            debug_data = df_aug_out[df_aug_out[cfg.group_col] == problem_group][
                ["image_name", cfg.group_col, "aug_idx", "split"]
            ].head(5)
            print("🔍 Debug info for first problematic group:")
            print(debug_data)
            raise RuntimeError("Split propagation failed: groups span multiple splits")
        else:
            print(f"✅ All {df_aug_out[cfg.group_col].nunique()} groups properly contained within single splits")
    else:
        print("⚠️  No group column available for cross-split validation")

    _print_audit("DL augmented", df_aug_out, cfg.label_col if cfg.label_col in df_aug_out.columns else cfg.label_col)
    _save_dual(df_aug_out, args.dl_aug_xlsx_out, args.dl_aug_csv_out)

    print("\n✅ All splits generated successfully. Retrain ML & DL against these CSVs.")
    print("   DL trainers will auto-detect 'val' split if present.\n")

if __name__ == "__main__":
    main()
