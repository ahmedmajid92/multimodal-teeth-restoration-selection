# src/fusion/fuse_infer.py
from pathlib import Path
import json
import numpy as np
import pandas as pd

from .prepare_streams import _resolve_image, _build_features, _ALL_FEATURES
from ..vision.predict_vision import predict_image
from ..tabular.predict_tabular import predict_xgb, predict_lgbm

def _load_recipe(fusion_dir: Path):
    """Load fusion recipe, supporting both 'streams.json' and 'fusion_summary.json'."""
    fusion_dir = Path(fusion_dir)
    # First try streams.json (if you want to keep compatibility with older code)
    streams_p = fusion_dir / "streams.json"
    if streams_p.exists():
        data = json.loads(streams_p.read_text(encoding="utf-8"))
        # Normalize into a uniform structure expected below
        # Accept either a modern schema (with 'streams_used', 'blend', 'stack') or a minimal legacy one.
        if "streams_used" in data and "blend" in data and "stack" in data:
            return {
                "choice": data.get("choice", "blend"),
                "streams_used": data["streams_used"],
                "threshold": data.get("threshold", data["blend"].get("threshold")),
                "blend": {
                    "weights": data["blend"]["weights"],
                    "threshold": data["blend"].get("threshold", data.get("threshold", 0.5)),
                },
                "stack": {
                    "weights": data["stack"]["weights"],
                    "threshold": data["stack"].get("threshold", data.get("threshold", 0.5)),
                },
            }
        # If someone stored {"streams": {"v_hard": w, ...}, "choice": ..., "threshold": ...}
        if "streams" in data:
            names = list(data["streams"].keys())
            weights = [float(data["streams"][k]) for k in names]
            choice = data.get("choice", "blend")
            thr = float(data.get("threshold", 0.5))
            return {
                "choice": choice,
                "streams_used": names,
                "threshold": thr,
                "blend": {"weights": weights, "threshold": thr},
                "stack": {"weights": weights, "threshold": thr},
            }
        # Fall through to summary if format is unknown
    # Fallback to fusion_summary.json (what our trainer writes)
    meta_p = fusion_dir / "fusion_summary.json"
    if not meta_p.exists():
        raise FileNotFoundError(f"Fusion artifacts not found in {fusion_dir}")
    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    return {
        "choice": meta["choice"],
        "streams_used": meta["streams_used"],
        "threshold": meta["threshold"],
        "blend": {"weights": meta["blend"]["weights"], "threshold": meta["blend"]["threshold"]},
        "stack": {"weights": meta["stack"]["weights"], "threshold": meta["stack"]["threshold"]},
    }

def infer_case(
    row: pd.Series,
    image_root: Path,
    weight_dir: Path,
    fusion_dir: Path,
    xgb_model_path: Path | None = None,
    lgbm_model_path: Path | None = None,
    override_threshold: float | None = None
):
    rec = _load_recipe(Path(fusion_dir))
    used = rec["streams_used"]
    choice = rec["choice"]

    # If caller passed an override, use that; otherwise let the chosen recipe decide
    global_t = rec["threshold"]
    t_override = override_threshold

    # ---------- Vision ----------
    streams = {}
    img_path = _resolve_image(Path(image_root), str(row["image_name"]))

    vh = Path(weight_dir) / "vision_hard_best.pt"
    if vh.exists():
        streams["v_hard"] = float(predict_image(str(vh), str(img_path), is_regressor=False))

    vs = Path(weight_dir) / "vision_soft_best.pt"
    if vs.exists():
        streams["v_soft"] = float(predict_image(str(vs), str(img_path), is_regressor=True))

    # ---------- Tabular ----------
    X = _build_features(pd.DataFrame([row]))[_ALL_FEATURES]
    if xgb_model_path:
        streams["xgb"] = float(predict_xgb(xgb_model_path, X)[0])
    if lgbm_model_path:
        streams["lgbm"] = float(predict_lgbm(lgbm_model_path, X)[0])

    # ---------- Compose ----------
    present, probs = [], []
    for k in used:
        if k in streams:
            present.append(k)
            probs.append(streams[k])

    if not probs:
        raise RuntimeError("No available streams for this case. Check weights/models.")

    probs = np.array(probs, dtype=float)

    if choice == "blend":
        all_w = np.array(rec["blend"]["weights"], dtype=float)
        mask = [i for i, k in enumerate(used) if k in present]
        w = all_w[mask]
        p = float((probs * w).sum())
        thr_method = rec["blend"]["threshold"]
    else:  # "stack"
        all_w = np.array(rec["stack"]["weights"], dtype=float)
        mask = [i for i, k in enumerate(used) if k in present]
        w = all_w[mask]
        p = float((probs * w).sum())
        thr_method = rec["stack"]["threshold"]

    # Choose threshold priority: explicit override > global tuned > method-specific
    t = (override_threshold if override_threshold is not None
         else (rec.get("threshold", None) if rec.get("threshold", None) is not None else thr_method))

    y = 1 if p >= t else 0

    return {
        "image_name": row["image_name"],
        "p_indirect": p,
        "threshold": t,
        "label": "Indirect" if y == 1 else "Direct",
        "streams_used": present,
        "streams_raw": {k: float(v) for k, v in streams.items()},   # NEW
        "weights_used": {k: float(wi) for k, wi in zip(present, w)},# NEW
        "components": {k: float(wi * sv) for k, wi, sv in zip(present, w, probs)},  # NEW
        "choice": choice
    }

