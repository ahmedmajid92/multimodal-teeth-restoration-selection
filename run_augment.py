#!/usr/bin/env python3
import argparse
from pathlib import Path
from src.preprocessing.augment import augment_dataset

def main():
    ap = argparse.ArgumentParser(description="Augment molar image dataset")
    ap.add_argument("--input_dir", required=True, type=Path,
                    help="Preprocessed images dir. Either flat with CSV labels or {direct,indirect} subfolders.")
    ap.add_argument("--output_dir", required=True, type=Path,
                    help="Where to write augmented dataset; will create {direct,indirect} subfolders.")
    ap.add_argument("--target_total", type=int, default=5000,
                    help="Desired total images across both classes (balanced). Default 5000.")
    ap.add_argument("--per_class_target", type=int, default=None,
                    help="Override target per class (e.g., 2500). If set, ignores --target_total.")
    ap.add_argument("--labels_csv", type=Path, default=None,
                    help="Optional CSV with labels. Must have 'image_name' and 'label' OR 'Direct' & 'Indirect' columns.")
    ap.add_argument("--label_column", type=str, default="label",
                    help="Column name with string labels ('direct' or 'indirect') if not using votes.")
    ap.add_argument("--direct_column", type=str, default=None,
                    help="If provided with --indirect_column, majority vote decides label.")
    ap.add_argument("--indirect_column", type=str, default=None,
                    help="If provided with --direct_column, majority vote decides label.")
    ap.add_argument("--strength", type=str, choices=["light", "medium", "strong"], default="medium",
                    help="Augmentation strength.")
    ap.add_argument("--size", type=int, default=512,
                    help="Output image size (square).")
    ap.add_argument("--no_copy_originals", action="store_true",
                    help="If set, do not copy originals into output set (only augmented).")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    args = ap.parse_args()

    augment_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        target_total=None if args.per_class_target is not None else args.target_total,
        per_class_target=args.per_class_target,
        labels_csv=args.labels_csv,
        label_column=args.label_column,
        direct_column=args.direct_column,
        indirect_column=args.indirect_column,
        strength=args.strength,
        size=args.size,
        copy_originals=not args.no_copy_originals,
        seed=args.seed,
    )
    print(f"✔ Augmentation complete. Output in {args.output_dir}")

if __name__ == "__main__":
    main()
