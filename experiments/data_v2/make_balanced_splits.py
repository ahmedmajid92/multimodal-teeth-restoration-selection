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

def _has_col(df: pd.DataFrame, col: str) -> bool:
    return col in df.columns

def main():
    ap = argparse.ArgumentParser()
    # Inputs (existing files)
    ap.add_argument("--raw-xlsx",           default="data/excel/data.xlsx")
    ap.add_argument("--processed-xlsx",     default="data/excel/data_processed.xlsx")
    ap.add_argument("--dl-xlsx",            default="data/excel/data_dl.xlsx")
    ap.add_argument("--dl-aug-xlsx",        default="data/excel/data_dl_augmented.xlsx")

    # Outputs (will overwrite with split columns)
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
    ap.add_argument("--label-col",  default="y_majority")  # binary 0/1 everywhere
    args = ap.parse_args()

    cfg = SplitConfig(
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
        group_col=args.group_col,
        label_col=args.label_col,
    )

    print("="*88)
    print("🧮 Making balanced Train/Val/Test splits (grouped & stratified)")
    print(f"  Fractions: train={cfg.train_frac} val={cfg.val_frac} test={cfg.test_frac}  |  seed={cfg.seed}")
    print(f"  group_col='{cfg.group_col}'  label_col='{cfg.label_col}'")
    print("="*88)

    # ------------------------------------------------------------------
    # 1) Tabular (processed) — used by your ML models
    # ------------------------------------------------------------------
    df_proc = pd.read_excel(args.processed_xlsx, engine="openpyxl")  # Fixed: processed_xlsx not processed-xlsx
    # Ensure label column types
    if _has_col(df_proc, cfg.label_col):
        df_proc[cfg.label_col] = df_proc[cfg.label_col].astype(int)
    else:
        raise KeyError(f"Processed data missing '{cfg.label_col}'")

    # Choose grouped vs row split
    if _has_col(df_proc, cfg.group_col):
        tr, va, te = stratified_group_split(df_proc, cfg)
    else:
        tr, va, te = stratified_row_split(df_proc, cfg)
    df_proc_out = pd.concat([tr.assign(split="train"), va.assign(split="val"), te.assign(split="test")], ignore_index=True)

    _print_audit("Processed (ML)", df_proc_out, cfg.label_col)
    _save_dual(df_proc_out, args.processed_xlsx_out, args.processed_csv_out)

    # ------------------------------------------------------------------
    # 2) DL base (no augmentation) — used to define DL splits
    # ------------------------------------------------------------------
    df_dl = pd.read_excel(args.dl_xlsx, engine="openpyxl")  # Fixed: dl_xlsx not dl-xlsx
    if _has_col(df_dl, cfg.label_col):
        df_dl[cfg.label_col] = df_dl[cfg.label_col].astype(int)
    else:
        raise KeyError(f"DL data missing '{cfg.label_col}' (0/1)")

    if _has_col(df_dl, cfg.group_col):
        tr, va, te = stratified_group_split(df_dl, cfg)
    else:
        tr, va, te = stratified_row_split(df_dl, cfg)

    df_dl_out = pd.concat([tr.assign(split="train"), va.assign(split="val"), te.assign(split="test")], ignore_index=True)
    _print_audit("DL base", df_dl_out, cfg.label_col)
    _save_dual(df_dl_out, args.dl_xlsx_out, args.dl_csv_out)

    # ------------------------------------------------------------------
    # 3) DL augmented — must INHERIT split from DL base (zero leakage)
    # ------------------------------------------------------------------
    df_aug = pd.read_excel(args.dl_aug_xlsx, engine="openpyxl")
    # Allow missing label in augmented file; if present, cast to int
    if _has_col(df_aug, cfg.label_col):
        df_aug[cfg.label_col] = df_aug[cfg.label_col].astype(int)

    # Debug: Check what columns are available
    print(f"📊 DL base columns: {list(df_dl_out.columns)}")
    print(f"📊 DL augmented columns: {list(df_aug.columns)}")
    
    # Special handling: if base doesn't have origin_id but augmented does,
    # we need to map origin_id back to image_name for the base split lookup
    if (cfg.group_col not in df_dl_out.columns and 
        cfg.group_col in df_aug.columns and 
        "image_name" in df_dl_out.columns):
        
        print(f"⚠️  Base data missing '{cfg.group_col}', using image_name mapping")
        
        # Create a mapping from origin_id to base image_name using the augmented data
        # We'll use the first occurrence (aug_idx=0) to map back to original
        origin_to_base = (
            df_aug[df_aug["aug_idx"] == 0][["origin_id", "image_name"]]
            .drop_duplicates()
            .rename(columns={"image_name": "base_image_name"})
        )
        
        # Add origin_id to base data via this mapping
        df_dl_out_with_origin = df_dl_out.merge(
            origin_to_base.rename(columns={"base_image_name": "image_name"}),
            on="image_name",
            how="left"
        )
        
        # Use origin_id as the group column for propagation
        base_cols = ["origin_id", "split"]
        if _has_col(df_dl_out, cfg.label_col):
            base_cols.append(cfg.label_col)
        
        print(f"📊 Using mapped base columns for propagation: {base_cols}")
        
        # Now propagate using origin_id
        df_aug_out = propagate_split_to_augmented(
            df_aug=df_aug,
            df_base=df_dl_out_with_origin[base_cols].drop_duplicates(),
            group_col=cfg.group_col,
            image_col_aug="image_name",
            image_col_base="image_name",
            parent_col_aug=None,  # We're using origin_id directly
        )
    else:
        # Original logic for when columns match
        base_cols = ["image_name"]
        if _has_col(df_dl_out, cfg.group_col):
            base_cols.append(cfg.group_col)
        base_cols.append("split")
        if _has_col(df_dl_out, cfg.label_col):
            base_cols.append(cfg.label_col)
        
        print(f"📊 Using base columns for propagation: {base_cols}")

        df_aug_out = propagate_split_to_augmented(
            df_aug=df_aug,
            df_base=df_dl_out[base_cols].copy(),
            group_col=cfg.group_col if _has_col(df_dl_out, cfg.group_col) else None,
            image_col_aug="image_name",
            image_col_base="image_name",
            parent_col_aug="parent_image" if "parent_image" in df_aug.columns else None,
        )

    # If augmented file does not have labels, fill from base on group
    if (cfg.label_col not in df_aug_out.columns and 
        cfg.group_col in df_aug_out.columns and 
        cfg.group_col in df_dl_out_with_origin.columns if 'df_dl_out_with_origin' in locals() else False):
        
        source_df = df_dl_out_with_origin if 'df_dl_out_with_origin' in locals() else df_dl_out
        if cfg.group_col in source_df.columns:
            df_aug_out = df_aug_out.merge(
                source_df[[cfg.group_col, cfg.label_col]].drop_duplicates(),
                on=cfg.group_col, how="left"
            )

    # Sanity: check no group crosses splits (only if group_col exists)
    if cfg.group_col in df_aug_out.columns:
        cross = (
            df_aug_out.groupby(cfg.group_col)["split"]
            .nunique()
            .reset_index(name="n_splits")
        )
        bad = cross[cross["n_splits"] > 1]
        if len(bad) > 0:
            print("❌ ERROR: some groups appear in multiple splits after propagation. Showing first 10:")
            print(bad.head(10))
            print("\n🔍 Debugging info for first problematic group:")
            problem_group = bad.iloc[0]["origin_id"]
            debug_data = df_aug_out[df_aug_out[cfg.group_col] == problem_group][["image_name", "origin_id", "aug_idx", "split"]].head(10)
            print(debug_data)
            raise SystemExit(1)
    else:
        print("⚠️  WARNING: No group column found, cannot verify group exclusivity across splits")

    # Save & audit
    audit_label = cfg.label_col if cfg.label_col in df_aug_out.columns else "y_majority"
    _print_audit("DL augmented", df_aug_out, audit_label)
    _save_dual(df_aug_out, args.dl_aug_xlsx_out, args.dl_aug_csv_out)

    print("\n✅ All splits generated successfully. You can now retrain ML & DL models.")
    print("   • DL trainers will automatically use 'val' if present in CSV.")
    print("   • Augmented images inherit the base split → no leakage.\n")

if __name__ == "__main__":
    main()
