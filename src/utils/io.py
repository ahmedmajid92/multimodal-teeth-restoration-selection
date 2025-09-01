"""
Tiny helpers for IO and logging.
"""
import json, shutil
from datetime import datetime
from pathlib import Path

def ensure_dir(d: Path) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    return d

def save_json(obj, path: Path):
    with open(path, "w") as fp:
        json.dump(obj, fp, indent=2)

def timestamp() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")

def copy_with_new_name(src: Path, dst_dir: Path, new_name: str):
    ensure_dir(dst_dir)
    ext = src.suffix.lower()
    dst = dst_dir / f"{new_name}{ext}"
    shutil.copy2(src, dst)
    return dst
