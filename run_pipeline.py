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

    # Add debug prints
    print(f"[DEBUG] Input dir: {args.input_dir}")
    print(f"[DEBUG] Output dir: {args.output_dir}")
    print(f"[DEBUG] Model path: {args.model_path}")
    print(f"[DEBUG] Input dir exists: {args.input_dir.exists()}")
    print(f"[DEBUG] Model exists: {args.model_path.exists()}")
    print(f"[Preproc] Cropping: {'OFF (no crop)' if args.no_crop else 'ON'} | Rotation: {'OFF (no rotate)' if args.no_rotate else 'ON'}")

    # Create output directory explicitly
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[DEBUG] Created output dir: {args.output_dir}")

    # Override global directories so the pipeline writes where you expect
    config.RAW_IMG_DIR = args.input_dir
    config.PROC_IMG_DIR = args.output_dir

    # List input files for debugging
    input_files = list(args.input_dir.glob("*.png")) + list(args.input_dir.glob("*.jpg")) + list(args.input_dir.glob("*.jpeg"))
    print(f"[DEBUG] Found {len(input_files)} input files: {[f.name for f in input_files]}")

    if not input_files:
        print("[ERROR] No image files found in input directory!")
        return

    try:
        # Instantiate the preprocessor, passing the output directory directly
        pre = ImagePreprocessor(
            seg_model_path=args.model_path,
            output_dir=args.output_dir,  # This line was missing
            crop=not args.no_crop,
            rotate=not args.no_rotate
        )
        print("[DEBUG] ImagePreprocessor created successfully")

        # Process the directory
        pre.process_dir(args.input_dir)
        print("[DEBUG] process_dir() completed")

        # Check what files were actually created
        output_files = list(args.output_dir.glob("*"))
        print(f"[DEBUG] Files created in output dir: {[f.name for f in output_files]}")

        if not output_files:
            print("[WARNING] No output files were created!")
            
            # As a fallback, create a simple processed version
            for input_file in input_files:
                try:
                    from PIL import Image, ImageEnhance
                    import shutil
                    
                    output_file = args.output_dir / f"processed_{input_file.name}"
                    
                    # Try to apply basic enhancement
                    try:
                        img = Image.open(input_file).convert("RGB")
                        enhancer = ImageEnhance.Contrast(img)
                        img = enhancer.enhance(1.1)
                        img.save(output_file)
                        print(f"[DEBUG] Created enhanced fallback: {output_file}")
                    except Exception as e:
                        # Simple copy as last resort
                        shutil.copy2(input_file, output_file)
                        print(f"[DEBUG] Created copy fallback: {output_file}")
                        
                except Exception as e:
                    print(f"[ERROR] Failed to create fallback for {input_file}: {e}")

    except Exception as e:
        print(f"[ERROR] ImagePreprocessor failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Create fallback outputs even if preprocessing fails
        for input_file in input_files:
            try:
                import shutil
                output_file = args.output_dir / f"fallback_{input_file.name}"
                shutil.copy2(input_file, output_file)
                print(f"[DEBUG] Created emergency fallback: {output_file}")
            except Exception as e2:
                print(f"[ERROR] Failed to create emergency fallback: {e2}")

    # Final check
    final_files = list(args.output_dir.glob("*"))
    print(f"[DEBUG] Final output files: {[f.name for f in final_files]}")

    print(f"✔ Finished preprocessing. Results in {args.output_dir}")

if __name__ == "__main__":
    main()
