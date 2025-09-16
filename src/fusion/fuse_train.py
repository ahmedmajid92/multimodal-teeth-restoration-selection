# src/fusion/fuse_train.py
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from .metrics import evaluate, tune_threshold
from .calibration import Calibrator
from .meta_learner import MetaStacker
from .weight_search import search_weights

def fit_fusion(
    data_csv="data_processed.csv",
    image_root="images",
    weight_dir="weights",          # default -> 'weights'
    ml_dir="models/outputs",
    out_dir="weights/fusion",
    calibrator_kind="isotonic",
    threshold_metric="f1",
    xgb_model_path=None,
    lgbm_model_path=None,
    skip_tabular=False,
    val_ratio=0.2,                 # NEW: auto-val share from train if missing
    random_state=42                # NEW: reproducible split
):
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_csv)
    # normalize split labels
    if "split" not in df.columns:
        raise RuntimeError("Column 'split' not found in data CSV.")
    df["split"] = df["split"].astype(str).str.lower()

    has_val = (df["split"] == "val").any()
    has_test = (df["split"] == "test").any()

    if not has_test:
        raise RuntimeError("TEST split not found. Ensure some rows have split='test'.")

    if not has_val:
        # Build a validation split from the TRAIN subset (stratified by y_majority)
        train_df = df[df["split"] == "train"].copy()
        if train_df.empty:
            raise RuntimeError("No 'val' and no 'train' rows — cannot create validation.")
        if "y_majority" not in train_df.columns:
            raise RuntimeError("y_majority column required to create stratified validation split.")

        sss = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=random_state)
        idx_train, idx_val = next(sss.split(train_df, train_df["y_majority"].astype(int)))
        df_val = train_df.iloc[idx_val].copy()
        df_test = df[df["split"] == "test"].copy()
        print(f"[INFO] Auto-created VAL from TRAIN: val_ratio={val_ratio}  "
              f"VAL={len(df_val)}  TEST={len(df_test)}")
    else:
        df_val = df[df["split"] == "val"].copy()
        df_test = df[df["split"] == "test"].copy()

    y_val = df_val["y_majority"].values.astype(int)
    y_test = df_test["y_majority"].values.astype(int)

    from .prepare_streams import collect_base_preds
    preds = collect_base_preds(
        df_val, df_test, Path(image_root), Path(weight_dir), Path(ml_dir),
        Path(xgb_model_path) if xgb_model_path else None,
        Path(lgbm_model_path) if lgbm_model_path else None,
        skip_tabular=skip_tabular
    )

    streams = ["v_hard","v_soft","xgb","lgbm"]
    val_list, test_list, used = [], [], []
    for k in streams:
        if preds["val"].get(k) is not None:
            val_list.append(preds["val"][k].reshape(-1))
            test_list.append(preds["test"][k].reshape(-1))
            used.append(k)

    if len(val_list) == 0:
        raise RuntimeError(
            "No base predictions available. "
            "Check image_root, model paths, or run with --skip-tabular to use vision only."
        )

    P_val = np.vstack(val_list).T
    P_test = np.vstack(test_list).T

    # Calibrate each stream on VAL
    for i in range(P_val.shape[1]):
        Ci = Calibrator(kind=calibrator_kind).fit(P_val[:, i], y_val)
        P_val[:, i]  = Ci.transform(P_val[:, i])
        P_test[:, i] = Ci.transform(P_test[:, i])

    # Meta-stacker
    stacker = MetaStacker(C=1.0).fit(P_val, y_val)
    p_val_stacked = stacker.predict_proba(P_val)
    t_stacked, _ = tune_threshold(y_val, p_val_stacked, threshold_metric)
    val_stacked = evaluate(y_val, p_val_stacked, t_stacked)

    # Non-negative blend
    blend = search_weights(P_val, y_val, metric="f1", step=0.1, threshold_mode="tune")
    p_val_blend = (P_val * np.array(blend["weights"]).reshape(1,-1)).sum(axis=1)
    val_blend = evaluate(y_val, p_val_blend, blend["threshold"])

    use_blend = val_blend["f1"] >= val_stacked["f1"]
    choice = "blend" if use_blend else "stack"

    if use_blend:
        p_test = (P_test * np.array(blend["weights"]).reshape(1,-1)).sum(axis=1)
        t = blend["threshold"]
    else:
        p_test = stacker.predict_proba(P_test)
        t = t_stacked

    test_metrics = evaluate(y_test, p_test, t)
    meta = {
        "choice": choice,
        "streams_used": used,
        "calibrator": calibrator_kind,
        "stack": {"weights": stacker.weights_, "threshold": t_stacked, "val_metrics": val_stacked},
        "blend": {"weights": blend["weights"], "threshold": blend["threshold"], "val_metrics": val_blend},
        "test_metrics": test_metrics,
        "threshold": t,
    }

    with open(Path(out_dir)/"fusion_summary.json", "w") as f:
        json.dump(meta, f, indent=2)
    np.save(Path(out_dir)/"P_val.npy", P_val)
    np.save(Path(out_dir)/"P_test.npy", P_test)

    print("== Fusion selection ==", choice)
    print("VAL (stack):", val_stacked)
    print("VAL (blend):", val_blend)
    print("TEST:", test_metrics)
