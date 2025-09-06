#!/usr/bin/env python3
import argparse
from pathlib import Path
from src.preprocessing.augment_records import augment_with_records

def main():
    ap = argparse.ArgumentParser(description="Augment images + append spreadsheet records")
    ap.add_argument("--input_dir", required=True, type=Path,
                    help="Folder with source images (e.g., .\\data\\processed\\images)")
    ap.add_argument("--output_dir", required=True, type=Path,
                    help="Folder to save augmented images (e.g., .\\data\\augmented)")
    ap.add_argument("--excel_in", required=True, type=Path,
                    help="Path to the existing Excel file (e.g., .\\data\\excel\\data_dl.xlsx)")
    ap.add_argument("--excel_out", required=True, type=Path,
                    help="Path to write the NEW Excel (e.g., .\\data\\excel\\data_dl_augmented.xlsx)")
    ap.add_argument("--csv_out", required=True, type=Path,
                    help="Path to write the NEW CSV (e.g., .\\data\\excel\\data_dl_augmented.csv)")
    ap.add_argument("--multiplier", type=int, default=10,
                    help="How many new images per source (default 10)")
    ap.add_argument("--size", type=int, default=512,
                    help="Output image size (square). Default 512")
    ap.add_argument("--strength", type=str, choices=["light", "medium", "strong"], default="medium",
                    help="Augmentation intensity (default 'medium')")
    ap.add_argument("--no_blur", action="store_true", help="Disable motion blur")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    args = ap.parse_args()

    augment_with_records(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        excel_in=args.excel_in,
        excel_out=args.excel_out,
        csv_out=args.csv_out,
        multiplier=args.multiplier,
        size=args.size,
        strength=args.strength,
        seed=args.seed,
        use_blur=not args.no_blur,
    )

if __name__ == "__main__":
    main()
