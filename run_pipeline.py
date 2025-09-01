#!/usr/bin/env python3
import argparse
from pathlib import Path

from src import config
from src.preprocessing.pipeline import ImagePreprocessor

def main():
    parser = argparse.ArgumentParser(description="Run preprocessing pipeline")
    parser.add_argument(
        "--input_dir", required=True, type=Path,
        help="Folder with raw JPG/PNG images"
    )
    parser.add_argument(
        "--output_dir", required=True, type=Path,
        help="Folder for processed images"
    )
    parser.add_argument(
        "--model_path", required=True, type=Path,
        help="Path to Mask-RCNN weights (ignored if --no_crop)"
    )
    parser.add_argument(
        "--no_crop", action="store_true",
        help="Skip segmentation cropping; do CLAHE + (optional) deskew + resize only"
    )
    parser.add_argument(
        "--no_rotate", action="store_true",
        help="Skip deskew/rotation (keep original orientation)"
    )
    args = parser.parse_args()

    # Override global directories so the pipeline writes where you expect
    config.RAW_IMG_DIR = args.input_dir
    config.PROC_IMG_DIR = args.output_dir

    # Instantiate the preprocessor with the two toggles
    pre = ImagePreprocessor(
        seg_model_path=args.model_path,
        crop=not args.no_crop,
        rotate=not args.no_rotate
    )
    pre.process_dir(args.input_dir)

    print(f"✔ Finished preprocessing. Results in {args.output_dir}")

if __name__ == "__main__":
    main()
