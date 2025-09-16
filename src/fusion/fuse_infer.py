# src/fusion/fuse_infer.py
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .calibration import Calibrator
from .metrics import evaluate

LABELS = {0: "Direct", 1: "Indirect"}

def infer_case(
    case_row: pd.Series,
    image_root="images",
    weight_dir="weight",
    ml_dir="models/outputs",
    fusion_dir="weight/fusion",
    override_threshold=None
):
    # load fusion choice
    with open(Path(fusion_dir)/"fusion_summary.json") as f:
        info = json.load(f)
    with open(Path(fusion_dir)/"streams.json") as f:
        streams_info = json.load(f)
    streams = streams_info["streams"]
    calib_kind = streams_info["calibrator_kind"]

    # build base predictions for this single case
    df = pd.DataFrame([case_row])
    from .prepare_streams import collect_base_preds
    preds = collect_base_preds(
        df_val=df, df_test=df,
        image_root=Path(image_root), weight_dir=Path(weight_dir), ml_dir=Path(ml_dir)
    )
    p = []
    for k in streams:
        v = preds["val"][k]
        if v is None:
            continue
        # re-fit a tiny calibrator on-the-fly using the training VAL bundle if you prefer; here identity
        c = Calibrator(kind=calib_kind)  # identity without fitting
        c.model = None
        p.append(v.reshape(-1))
    P = np.vstack(p).T  # shape [1, n_streams]

    # fuse
    if info["choice"] == "blend":
        w = np.array(info["blend"]["weights"]).reshape(-1)
        p_final = float(np.clip((P * w).sum(axis=1)[0], 0, 1))
        t = info["blend"]["threshold"]
    else:
        # meta stacker weights approximated from normalized coef; use them for interpretability
        w = np.array(info["stack"]["weights"]).reshape(-1)
        p_final = float(np.clip((P * w).sum(axis=1)[0], 0, 1))
        t = info["stack"]["threshold"]

    if override_threshold is not None:
        t = float(override_threshold)

    label = 1 if p_final >= t else 0
    return {
        "p_indirect": p_final,
        "threshold": t,
        "decision": LABELS[label],
        "stream_weights": dict(zip(streams, w.round(3).tolist()))
    }
