# run_augment_records.py
"""
Convenience runner that stays compatible with the old API.
It simply forwards CLI args to src.preprocessing.augment_records.augment_records().
"""

import argparse
from src.preprocessing.augment_records import augment_records

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-table", default="data/excel/data_dl.xlsx")
    ap.add_argument("--images-src", default="data/processed/images")
    ap.add_argument("--images-dst", default="data/augmented")
    ap.add_argument("--num-aug-per-image", type=int, default=10)
    ap.add_argument("--start-id", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--make-val", action="store_true")
    ap.add_argument("--val-frac", type=float, default=0.12)
    ap.add_argument("--out-csv", default="data/excel/data_dl_augmented.csv")
    ap.add_argument("--out-xlsx", default="data/excel/data_dl_augmented.xlsx")
    ap.add_argument("--aug-preset", choices=["legacy", "simple", "none"], default="legacy")
    ap.add_argument("--img-size-for-aug", type=int, default=512)
    args = ap.parse_args()

    augment_records(
        input_table=args.input_table,
        images_src=args.images_src,
        images_dst=args.images_dst,
        num_aug_per_image=args.num_aug_per_image,
        start_id=args.start_id,
        seed=args.seed,
        make_val=args.make_val,
        val_frac=args.val_frac,
        out_csv=args.out_csv,
        out_xlsx=args.out_xlsx,
        aug_preset=args.aug_preset,
        img_size_for_aug=args.img_size_for_aug,
    )

if __name__ == "__main__":
    main()
