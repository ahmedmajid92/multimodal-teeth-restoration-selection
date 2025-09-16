# run_fusion.py
# Unified CLI for training and using the hybrid fusion model.
# Place this at the repository root.
import argparse
import json
from pathlib import Path
import sys

def _default_paths():
    return dict(
        data_csv="data/excel/data_processed.csv",
        image_root="data/processed/images",
        weight_dir="weight",
        ml_dir="models/outputs",
        fusion_dir="weight/fusion",
    )

def _ensure_exists(p: Path, kind="file"):
    if kind == "file" and not p.is_file():
        sys.exit(f"[ERROR] Missing {kind}: {p}")
    if kind == "dir" and not p.exists():
        sys.exit(f"[ERROR] Missing {kind}: {p}")

def cmd_train(args):
    from src.fusion.fuse_train import fit_fusion
    kw = dict(
        data_csv=args.data_csv,
        image_root=args.image_root,
        weight_dir=args.weight_dir,
        ml_dir=args.ml_dir,
        out_dir=args.fusion_dir,
        calibrator_kind=args.calibrator,
        threshold_metric=args.metric,
        xgb_model_path=args.xgb_model,
        lgbm_model_path=args.lgbm_model,
        skip_tabular=args.skip_tabular,
    )
    print("[INFO] Training hybrid fusion with:", json.dumps(kw, indent=2))
    fit_fusion(**kw)
    print("[OK] Fusion artifacts saved to:", args.fusion_dir)

def cmd_info(args):
    fs = Path(args.fusion_dir) / "fusion_summary.json"
    st = Path(args.fusion_dir) / "streams.json"
    _ensure_exists(fs, "file")
    _ensure_exists(st, "file")

    with open(fs, "r") as f:
        summary = json.load(f)
    with open(st, "r") as f:
        streams = json.load(f)

    print("\n=== Fusion Summary ===")
    print(json.dumps(summary, indent=2))
    print("\n=== Streams Used ===")
    print(json.dumps(streams, indent=2))

def _load_row(args):
    import pandas as pd
    df = pd.read_csv(args.data_csv)
    if args.image_name:
        sub = df[df["image_name"] == args.image_name]
        if sub.empty:
            sys.exit(f"[ERROR] image_name '{args.image_name}' not found in {args.data_csv}")
        row = sub.iloc[0]
    else:
        try:
            row = df[df["split"] == args.split].iloc[args.row_idx]
        except IndexError:
            sys.exit(f"[ERROR] row_idx {args.row_idx} out of range for split '{args.split}'")
    return row

def cmd_infer_one(args):
    from src.fusion.fuse_infer import infer_case
    row = _load_row(args)
    res = infer_case(
        row,
        image_root=args.image_root,
        weight_dir=args.weight_dir,
        ml_dir=args.ml_dir,
        fusion_dir=args.fusion_dir,
        override_threshold=args.threshold
    )
    out = json.dumps(res, indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(out, encoding="utf-8")
        print("[OK] Saved:", args.out)
    else:
        print(out)

def cmd_infer_batch(args):
    import pandas as pd
    from src.fusion.fuse_infer import infer_case

    df = pd.read_csv(args.data_csv)
    if args.csv_in:
        # Use a custom CSV for inference (must contain columns needed by infer_case)
        test = pd.read_csv(args.csv_in)
    else:
        test = df if args.split == "all" else df[df["split"] == args.split].copy()

    if test.empty:
        sys.exit(f"[ERROR] No rows found for split '{args.split}' (or csv_in empty).")

    outs = []
    for _, r in test.iterrows():
        o = infer_case(
            r,
            image_root=args.image_root,
            weight_dir=args.weight_dir,
            ml_dir=args.ml_dir,
            fusion_dir=args.fusion_dir,
            override_threshold=args.threshold
        )
        outs.append(o)

    test["p_indirect"] = [o["p_indirect"] for o in outs]
    test["decision"]   = [o["decision"] for o in outs]

    out_path = Path(args.out or f"{args.ml_dir}/hybrid_{args.split}_predictions.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    test.to_csv(out_path, index=False)
    print("[OK] Saved:", out_path)

def build_parser():
    D = _default_paths()
    p = argparse.ArgumentParser(description="Hybrid Fusion CLI (train, info, infer-one, infer-batch)")
    sub = p.add_subparsers(dest="cmd", required=True)

    # train
    t = sub.add_parser("train", help="Fit fusion (calibrate + blend/stack + threshold tuning)")
    t.add_argument("--data-csv", default=D["data_csv"])
    t.add_argument("--image-root", default=D["image_root"])
    t.add_argument("--weight-dir", default=D["weight_dir"])
    t.add_argument("--ml-dir", default=D["ml_dir"])
    t.add_argument("--fusion-dir", default=D["fusion_dir"])
    t.add_argument("--calibrator", choices=["isotonic","platt"], default="isotonic")
    t.add_argument("--metric", choices=["f1","youden","pr_auc"], default="f1")
    t.add_argument("--xgb-model", default=None, help="Explicit path to XGBoost model (.json/.ubj/.bin/.pkl)")
    t.add_argument("--lgbm-model", default=None, help="Explicit path to LightGBM model (.txt/.pkl)")
    t.add_argument("--skip-tabular", action="store_true", help="Ignore tabular models and use vision only")
    t.set_defaults(func=cmd_train)

    # info
    i = sub.add_parser("info", help="Show fusion summary and streams used")
    i.add_argument("--fusion-dir", default=D["fusion_dir"])
    i.set_defaults(func=cmd_info)

    # infer-one
    o = sub.add_parser("infer-one", help="Infer a single case (Direct/Indirect)")
    o.add_argument("--data-csv", default=D["data_csv"])
    o.add_argument("--image-root", default=D["image_root"])
    o.add_argument("--weight-dir", default=D["weight_dir"])
    o.add_argument("--ml-dir", default=D["ml_dir"])
    o.add_argument("--fusion-dir", default=D["fusion_dir"])
    o.add_argument("--split", default="test", choices=["train","val","test","all"])
    o.add_argument("--row-idx", type=int, default=0, help="Index within the chosen split (ignored if --image-name set)")
    o.add_argument("--image-name", default=None, help="Exact filename in data CSV to select a case")
    o.add_argument("--threshold", type=float, default=None, help="Override final decision threshold")
    o.add_argument("--out", default=None, help="Path to save JSON output")
    o.set_defaults(func=cmd_infer_one)

    # infer-batch
    b = sub.add_parser("infer-batch", help="Batch inference over a split or custom CSV")
    b.add_argument("--data-csv", default=D["data_csv"])
    b.add_argument("--image-root", default=D["image_root"])
    b.add_argument("--weight-dir", default=D["weight_dir"])
    b.add_argument("--ml-dir", default=D["ml_dir"])
    b.add_argument("--fusion-dir", default=D["fusion_dir"])
    b.add_argument("--split", default="test", choices=["train","val","test","all"], help="Ignored if --csv-in is provided")
    b.add_argument("--csv-in", default=None, help="Optional custom CSV to run inference on")
    b.add_argument("--threshold", type=float, default=None, help="Override final decision threshold")
    b.add_argument("--out", default=None, help="Output CSV path (default: models/outputs/hybrid_<split>_predictions.csv)")
    b.set_defaults(func=cmd_infer_batch)

    return p

def main():
    p = build_parser()
    args = p.parse_args()

    # Minimal path sanity
    if args.cmd in {"train","infer-one","infer-batch"}:
        # data_csv may not be needed for infer-batch if csv_in is provided,
        # but checking it doesn't hurt unless csv_in is set.
        if getattr(args, "csv_in", None) is None:
            _ensure_exists(Path(args.data_csv), "file")
        _ensure_exists(Path(args.weight_dir), "dir")
        _ensure_exists(Path(args.ml_dir), "dir")
        Path(args.fusion_dir).parent.mkdir(parents=True, exist_ok=True)
    if args.cmd in {"infer-one","infer-batch"}:
        _ensure_exists(Path(args.image_root), "dir")

    args.func(args)

if __name__ == "__main__":
    main()
