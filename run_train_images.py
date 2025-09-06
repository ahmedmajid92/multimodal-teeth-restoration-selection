# run_train_images.py
import argparse, subprocess, sys, os

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["hard", "soft"], required=True)
    p.add_argument("--csv-path", default="data/excel/data_dl_augmented.csv")
    p.add_argument("--images-root", default="data/augmented")
    p.add_argument("--model-name", default=None)
    p.add_argument("--img-size", type=int, default=384)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=24)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--val-frac", type=float, default=0.12)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    args, extra = p.parse_known_args()

    base = [sys.executable]
    if args.task == "hard":
        script = "models/vision/train_hard.py"
        dflt = "tf_efficientnet_b3_ns"
    else:
        script = "models/vision/train_soft.py"
        dflt = "convnext_tiny"

    model_name = args.model_name or dflt

    cmd = base + [script,
        "--csv-path", args.csv_path,
        "--images-root", args.images_root,
        "--model-name", model_name,
        "--img-size", str(args.img_size),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--lr", str(args.lr),
        "--weight-decay", str(args.weight_decay),
        "--val-frac", str(args.val_frac),
        "--num-workers", str(args.num_workers),
        "--seed", str(args.seed)
    ] + extra

    # Set PYTHONPATH to include project root
    env = os.environ.copy()
    env['PYTHONPATH'] = os.getcwd()

    print("Running:", " ".join(cmd))
    raise SystemExit(subprocess.call(cmd, env=env))

if __name__ == "__main__":
    main()
