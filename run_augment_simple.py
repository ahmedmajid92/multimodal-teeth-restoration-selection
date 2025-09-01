#!/usr/bin/env python3
import argparse
from pathlib import Path
from src.preprocessing.augment_simple import augment_folder_fixed_multiplicity

def main():
    ap = argparse.ArgumentParser(description="Simple label-free augmentation")
    ap.add_argument("--input_dir", required=True, type=Path,
                    help="Folder with preprocessed images (e.g., data/processed/images)")
    ap.add_argument("--output_dir", required=True, type=Path,
                    help="Folder to write augmented images (e.g., data/augmented)")
    ap.add_argument("--multiplier", type=int, default=10,
                    help="How many new images to generate per source (default 10)")
    ap.add_argument("--size", type=int, default=512,
                    help="Output side length (square). Default 512")
    ap.add_argument("--strength", type=str, choices=["light", "medium", "strong"], default="medium",
                    help="Augmentation intensity. Default 'medium'")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--no_blur", action="store_true", help="Disable motion blur augmentation")
    args = ap.parse_args()

    augment_folder_fixed_multiplicity(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        multiplier=args.multiplier,
        size=args.size,
        strength=args.strength,
        seed=args.seed,
        use_blur=not args.no_blur, 
    )
    print(f"✔ Augmentation complete. Output in {args.output_dir}")

if __name__ == "__main__":
    main()
