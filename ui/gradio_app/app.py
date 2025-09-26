import os, sys, json, shutil
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
import utils
import gradio as gr

# --- Make subprocess + prints UTF-8 friendly on Windows ---
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

# Local imports
from tab_model import TabEnsemble
from utils import (
    ensure_dir, load_feature_choices_from_sheet, check_image_resolution,
    run_preprocessing_pipeline, build_tab_vector, BLUE_CSS
)

# --- Paths anchored to repo root (…/multimodal-tooth-restoration-ai) ---
# <repo>/ui/gradio_app/app.py -> parents[2] is repo root
REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULTS = {
    "xlsx_tab": REPO_ROOT / "data" / "excel" / "data_processed.xlsx",
    "mm_ckpt_dir": REPO_ROOT / "weights" / "mm_dualtask_v1",
    "mil_ckpt_dir": REPO_ROOT / "weights" / "mil_v1",

    # OOF/TEST CSVs (exactly as in your working command)
    "oof_mm":  REPO_ROOT / "weights" / "mm_dualtask_v1" / "finalized" / "oof_val.csv",
    "pred_mm": REPO_ROOT / "weights" / "mm_dualtask_v1" / "finalized" / "pred_test.csv",
    "oof_mil": REPO_ROOT / "weights" / "mil_v1" / "oof_val.csv",
    "pred_mil": REPO_ROOT / "weights" / "mil_v1" / "pred_test.csv",

    "thr_mode": "max_acc",
    "thr_target": 0.80,

    "segmenter_model": REPO_ROOT / "models" / "segmenter" / "mask_rcnn_molar.pt",
    "pipeline_script": REPO_ROOT / "run_pipeline.py",
    "tmp_raw_dir": REPO_ROOT / "ui" / "tmp" / "raw_images",
    "tmp_proc_dir": REPO_ROOT / "ui" / "tmp" / "processed_images",

    "stack_summary": REPO_ROOT / "results" / "stack_v2" / "summary.json",
    "tab_ckpt_dir": REPO_ROOT / "weights" / "tab_v1",
}

# Tabular feature definitions (kept here since tab_model no longer exports them)
CONT_FEATURES = ["depth", "width"]
CAT_FEATURES = [
    "enamel_cracks",
    "occlusal_load",
    "carious_lesion",
    "opposing_type",
    "adjacent_teeth",
    "age_range",
    "cervical_lesion",
]
TAB_FEATURES = CONT_FEATURES + CAT_FEATURES

# --- Load feature choices dynamically from your sheet for the form ---
try:
    FEATURE_CHOICES = load_feature_choices_from_sheet(DEFAULTS["xlsx_tab"])
except Exception as e:
    print(f"Warning: Could not load feature choices: {e}")
    FEATURE_CHOICES = {feat: [] for feat in CAT_FEATURES}

# --- Initialize ensembles and stacker at app startup ---
print("🔧 Initializing models...")
device = "cuda" if torch.cuda.is_available() else "cpu"

mm_ens = None
mil_ens = None
tab_ens = None
stacker = None
dbg_init = []

try:
    from infer_mm import MMEnsemble
    mm_ens = MMEnsemble(ckpt_dir=DEFAULTS["mm_ckpt_dir"], device=device)
    dbg_init.append(f"MM folds={mm_ens.num_folds}")
except Exception as e:
    dbg_init.append(f"MM load failed: {e}")

try:
    from infer_mil import MILEnsemble
    mil_ens = MILEnsemble(ckpt_dir=DEFAULTS["mil_ckpt_dir"], device=device)
    dbg_init.append(f"MIL folds={mil_ens.num_folds}")
except Exception as e:
    dbg_init.append(f"MIL load failed: {e}")

try:
    tab_ens = TabEnsemble.from_folder(DEFAULTS["tab_ckpt_dir"])
    if tab_ens is not None:
        dbg_init.append(f"TAB folds={tab_ens.num_folds}")
    else:
        dbg_init.append("TAB not found")
except Exception as e:
    dbg_init.append(f"TAB init failed: {e}")


from stack_meta import Stacker
try:
    stacker = Stacker(
        xlsx_tab=DEFAULTS["xlsx_tab"],
        oof_mm=DEFAULTS["oof_mm"], pred_mm=DEFAULTS["pred_mm"],
        oof_mil=DEFAULTS["oof_mil"], pred_mil=DEFAULTS["pred_mil"],
        folds=5, thr_mode=DEFAULTS["thr_mode"], thr_target=DEFAULTS["thr_target"]
    )
    dbg_init.append(f"Stacker thr_mode={stacker.thr_mode}")
except Exception as e:
    dbg_init.append(f"Stacker init failed: {e}")

print("✅ Init:", " | ".join(dbg_init))

def load_overall_metrics():
    """
    Read results/stack_v2/summary.json and render a small report.
    If the file is missing, fall back to the user's provided metrics.
    Returns (markdown_text, table_df)
    """
    # Default fallback: user's posted results
    fallback = {
        "thr_mode": "max_acc",
        "thr_target": 0.8,
        "thr": 0.470,
        "oof": {"auc": 0.8935, "acc": 0.8456, "prec": 0.8644, "rec": 0.9053, "f1": 0.8844},
        "test": {"auc": 0.8695, "acc": 0.8223, "prec": 0.8192, "rec": 0.9062, "f1": 0.8605},
    }

    path = DEFAULTS["stack_summary"]
    data = None
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            thr = data.get("thr") or data.get("threshold") or 0.470
            thr_mode = data.get("thr_mode", "max_acc")
            thr_target = data.get("thr_target", 0.80)
            oof = data.get("oof") or data.get("val_mean") or {}
            test = data.get("test") or data.get("test_mean") or {}
            payload = {
                "thr_mode": thr_mode, "thr_target": thr_target, "thr": thr,
                "oof": oof, "test": test
            }
        except Exception:
            payload = fallback
    else:
        payload = fallback

    md = (
        f"**Threshold mode:** `{payload['thr_mode']}`  |  "
        f"**target:** `{payload['thr_target']}`  |  "
        f"**chosen thr:** `{payload['thr']:.3f}`\n\n"
        f"**OOF**  —  AUC: **{payload['oof'].get('auc','-')}**, "
        f"ACC: **{payload['oof'].get('acc','-')}**, "
        f"Prec: **{payload['oof'].get('prec','-')}**, "
        f"Rec: **{payload['oof'].get('rec','-')}**, "
        f"F1: **{payload['oof'].get('f1','-')}**\n\n"
        f"**TEST** —  AUC: **{payload['test'].get('auc','-')}**, "
        f"ACC: **{payload['test'].get('acc','-')}**, "
        f"Prec: **{payload['test'].get('prec','-')}**, "
        f"Rec: **{payload['test'].get('rec','-')}**, "
        f"F1: **{payload['test'].get('f1','-')}**"
    )

    def row(d): return [d.get('auc','-'), d.get('acc','-'), d.get('prec','-'), d.get('rec','-'), d.get('f1','-')]
    df = pd.DataFrame(
        [row(payload['oof']), row(payload['test'])],
        columns=["AUC","ACC","PREC","REC","F1"],
        index=["OOF","TEST"]
    ).reset_index(names=["Split"])
    return md, df

from pathlib import Path
import os

def _find_first_image_local(directory: Path, extensions=(".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".gif"), recursive=True):
    directory = Path(directory)
    if not directory.exists():
        return None
    if recursive:
        for root, _, files in os.walk(directory):
            for f in sorted(files):
                if f.lower().endswith(extensions):
                    return str(Path(root) / f)
    else:
        for f in sorted(os.listdir(directory)):
            if f.lower().endswith(extensions):
                return str(directory / f)
    return None

def _is_empty_field(v) -> bool:
    """Return True if the clinical field should be considered 'not provided'."""
    if v is None:
        return True
    # Numbers: treat 0 or NaN as empty
    try:
        import math
        if isinstance(v, (int, float)):
            return v == 0 or (isinstance(v, float) and math.isnan(v))
    except Exception:
        pass
    # Strings: treat blank & common placeholders as empty
    if isinstance(v, str):
        s = v.strip().lower()
        return s == "" or s in {"select", "select...", "none", "n/a", "na", "-"}
    # Fallback: consider falsy values empty
    return not bool(v)


def predict_one(
    image: Image.Image,
    no_crop: bool, no_rotate: bool,
    seg_model_path: str,
    # tab fields (all optional collectively, but all-or-none)
    depth, width,
    enamel_cracks, occlusal_load, carious_lesion,
    opposing_type, adjacent_teeth, age_range, cervical_lesion,
    threshold_mode
):
    """Main callback for Gradio button."""
    
    # 0) Sanity: make sure we can call helpers
    try:
        _ = utils  # noqa
    except NameError:
        raise gr.Error("Internal import error: utils is not imported at top of app.py")

    # 1) Validate image resolution
    ok, msg = check_image_resolution(image, min_size=512)
    if not ok:
        raise gr.Error(msg)

    # 2) Create per-session dirs and persist the uploaded image (PNG is fine)
    base_raw = DEFAULTS["tmp_raw_dir"]
    base_proc = DEFAULTS["tmp_proc_dir"]
    utils.ensure_dir(base_raw)
    utils.ensure_dir(base_proc)
    raw_dir, proc_dir, sid = utils.make_session_dirs(base_raw, base_proc)

    raw_path = raw_dir / "input.png"
    try:
        # always save as RGB PNG
        (image.convert("RGB")).save(raw_path)
        print(f"[DEBUG] Saved uploaded image to: {raw_path}")
        print(f"[DEBUG] Image file exists: {raw_path.exists()}")
        print(f"[DEBUG] Image size: {image.size}")
    except Exception as e:
        raise gr.Error(f"Failed to save uploaded image: {e}")

    # 3) Run preprocessing pipeline (if image models are loaded)
    have_mm = (mm_ens is not None and getattr(mm_ens, 'num_folds', 0) > 0)
    have_mil = (mil_ens is not None and getattr(mil_ens, 'num_folds', 0) > 0)

    # 4) Determine whether tabular features were fully provided
    tab_inputs = {
        "depth": depth, "width": width,
        "enamel_cracks": enamel_cracks, "occlusal_load": occlusal_load, "carious_lesion": carious_lesion,
        "opposing_type": opposing_type, "adjacent_teeth": adjacent_teeth, "age_range": age_range, "cervical_lesion": cervical_lesion
    }

    # Treat 0/NaN/placeholder strings as NOT provided
    provided_mask = {k: not _is_empty_field(v) for k, v in tab_inputs.items()}
    any_filled = any(provided_mask.values())
    all_filled = all(provided_mask.values())

    use_tabular = False
    if any_filled and not all_filled:
        missing = [k for k, ok in provided_mask.items() if not ok]
        # Make the message more actionable, but still short
        raise gr.Error(
            "If you provide clinical fields, please provide **all** of them. "
            f"Missing: {', '.join(missing)}"
        )
    if all_filled:
        use_tabular = True

    # 5) Run preprocessing if we have image models or need preprocessing
    if have_mm or have_mil:
        print(f"[DEBUG] Starting preprocessing pipeline...")
        print(f"[DEBUG] Input dir: {raw_dir}")
        print(f"[DEBUG] Output dir: {proc_dir}")
        print(f"[DEBUG] Segmenter path: {Path(seg_model_path) if seg_model_path else DEFAULTS['segmenter_model']}")
        
        produced_dir, rep_img, pipe_log = utils.run_preprocessing_pipeline(
            pipeline_script=DEFAULTS["pipeline_script"],
            input_dir=raw_dir,
            output_dir=proc_dir,
            segmenter_path=Path(seg_model_path) if seg_model_path else DEFAULTS["segmenter_model"],
            no_crop=no_crop,
            no_rotate=no_rotate,
        )

        if produced_dir is None:
            # hard failure launching pipeline
            raise gr.Error(
                "Preprocessing failed to start.\n\n"
                f"Details:\n{pipe_log}"
            )

        # Decide what to show in the UI
        representative_img_path = rep_img if rep_img is not None else raw_path

        # Decide whether to run image-based streams
        has_proc_imgs = utils.has_any_images(produced_dir) if produced_dir else False

        prob_mm, mm_dbg = (None, "MM not loaded")
        prob_mil, mil_dbg = (None, "MIL not loaded")
        prob_tab = None

        if use_tabular and tab_ens is not None:
            prob_tab = tab_ens.predict_one(tab_inputs)

        if has_proc_imgs and have_mm:
            prob_mm, mm_dbg = mm_ens.predict(representative_img_path, tab_inputs if use_tabular else None)
        elif have_mm:
            mm_dbg = f"MM skipped: no preprocessed images found in {produced_dir.name if produced_dir else 'None'}"

        if has_proc_imgs and have_mil:
            prob_mil, mil_dbg = mil_ens.predict(produced_dir)
        elif have_mil:
            mil_dbg = f"MIL skipped: no preprocessed images found in {produced_dir.name if produced_dir else 'None'}"

        available = [p for p in [prob_mm, prob_mil, prob_tab] if p is not None]
        if not available:
            # Nothing ran — show the pipeline log to help debug
            raise gr.Error(
                "No model streams ran. Likely no preprocessed images were produced and no tabular inputs were provided.\n\n"
                f"Preprocessing log:\n{pipe_log}"
            )

        # Stacking / fallback average (unchanged)
        if stacker is not None:
            stacker.set_threshold_mode(threshold_mode, DEFAULTS["thr_target"])
            prob_final, chosen_thr, details = stacker.predict_single(
                prob_mm=prob_mm, prob_mil=prob_mil, prob_tab=prob_tab
            )
        else:
            prob_final = sum(available) / len(available)
            chosen_thr = 0.5
            details = {"mode": "fallback_average"}

        label = "Indirect" if prob_final >= chosen_thr else "Direct"
        prob_pct = f"{prob_final:.3f} (thr={chosen_thr:.3f})"

        rows = []
        if prob_tab is not None: rows.append(["Tabular (LightGBM)", f"{prob_tab:.3f}"])
        if prob_mm  is not None: rows.append(["MM (Image+Tab)",   f"{prob_mm:.3f}"])
        if prob_mil is not None: rows.append(["MIL (Image)",      f"{prob_mil:.3f}"])
        contrib_df = pd.DataFrame(rows, columns=["Stream", "Probability"])

        stacker_mode = stacker.thr_mode if stacker else "fallback"
        msg = (
            f"**Predicted:** {label}\n\n"
            f"**Calibrated probability (Indirect):** {prob_pct}\n\n"
            f"**Threshold mode:** {stacker_mode}\n"
            f"**Active streams:** {len(available)}\n"
        )

        # Show the best-available image (processed if exists, otherwise raw)
        out_image_path = str(representative_img_path)
        dbg = f"MM: {mm_dbg}\nMIL: {mil_dbg}\n\n{pipe_log}"
        return msg, contrib_df, out_image_path, dbg

    else:
        # No image-based streams loaded; handle tabular-only case
        if not use_tabular or tab_ens is None:
            raise gr.Error("No model streams available. Check your weights paths and/or provide tabular inputs.")
        
        prob_tab = tab_ens.predict_one(tab_inputs)
        prob_final = prob_tab
        chosen_thr = 0.5
        
        label = "Indirect" if prob_final >= chosen_thr else "Direct"
        prob_pct = f"{prob_final:.3f} (thr={chosen_thr:.3f})"
        
        rows = [["Tabular (LightGBM)", f"{prob_tab:.3f}"]]
        contrib_df = pd.DataFrame(rows, columns=["Stream", "Probability"])
        
        msg = (
            f"**Predicted:** {label}\n\n"
            f"**Calibrated probability (Indirect):** {prob_pct}\n\n"
            f"**Threshold mode:** tabular-only\n"
            f"**Active streams:** 1\n"
        )
        
        out_image_path = str(raw_path)
        dbg = "Tabular-only prediction (no image models loaded)"
        return msg, contrib_df, out_image_path, dbg


# ----------------------- Build Gradio UI -----------------------

theme = gr.themes.Soft(primary_hue="blue", secondary_hue="blue")
with gr.Blocks(theme=theme, css=BLUE_CSS, title="Dental Restoration Classifier") as demo:
    gr.Markdown(
        """
        # 🦷 Dental Restoration Classifier — Direct vs Indirect
        **Multimodal hybrid system** for dentistry clinics.  
        Upload a **tooth image** (≥ 512×512). Optionally provide clinical fields to enable **full hybrid** prediction.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            img_in = gr.Image(type="pil", label="Tooth image (≥ 512×512)")

            gr.Markdown("### Preprocessing options")
            no_crop = gr.Checkbox(label="--no_crop", value=True)
            no_rotate = gr.Checkbox(label="--no_rotate", value=True)
            seg_model_path = gr.Textbox(
                label="Segmentation model path",
                value=str(DEFAULTS["segmenter_model"]),
                placeholder="Path to mask_rcnn_molar.pt"
            )

            gr.Markdown("### Clinical / Tabular Fields (optional — but **all-or-none**)")
            # Continuous
            depth = gr.Number(label="depth (mm)")
            width = gr.Number(label="width (mm)")
            # Categoricals: choices read from your sheet
            enamel_cracks = gr.Dropdown(choices=FEATURE_CHOICES.get('enamel_cracks', []), label="enamel_cracks", value=None)
            occlusal_load = gr.Dropdown(choices=FEATURE_CHOICES.get('occlusal_load', []), label="occlusal_load", value=None)
            carious_lesion = gr.Dropdown(choices=FEATURE_CHOICES.get('carious_lesion', []), label="carious_lesion", value=None)
            opposing_type = gr.Dropdown(choices=FEATURE_CHOICES.get('opposing_type', []), label="opposing_type", value=None)
            adjacent_teeth = gr.Dropdown(choices=FEATURE_CHOICES.get('adjacent_teeth', []), label="adjacent_teeth", value=None)
            age_range = gr.Dropdown(choices=FEATURE_CHOICES.get('age_range', []), label="age_range", value=None)
            cervical_lesion = gr.Dropdown(choices=FEATURE_CHOICES.get('cervical_lesion', []), label="cervical_lesion", value=None)

            threshold_mode = gr.Dropdown(
                choices=["max_acc","max_f1","youden","target_prec","target_rec"],
                value=DEFAULTS["thr_mode"], label="Threshold mode"
            )

            run_btn = gr.Button("Preprocess & Predict", variant="primary")

        with gr.Column(scale=1):
            gr.Markdown("### Prediction")
            out_msg = gr.Markdown()
            out_table = gr.Dataframe(label="Per-Stream Probabilities", interactive=False)
            out_image = gr.Image(type="filepath", label="Preprocessed image")
            out_dbg = gr.Textbox(label="Debug info", lines=6)

    gr.Markdown("---")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## 📊 Model Performance")
            perf_btn = gr.Button("Show Current Performance", variant="secondary")
        with gr.Column(scale=2):
            perf_md = gr.Markdown()
            perf_df = gr.Dataframe(label="Summary", interactive=False)

    perf_btn.click(fn=load_overall_metrics, inputs=None, outputs=[perf_md, perf_df])

    run_btn.click(
        fn=predict_one,
        inputs=[img_in, no_crop, no_rotate, seg_model_path,
                depth, width, enamel_cracks, occlusal_load, carious_lesion,
                opposing_type, adjacent_teeth, age_range, cervical_lesion,
                threshold_mode],
        outputs=[out_msg, out_table, out_image, out_dbg]
    )

if __name__ == "__main__":
    # Minimal queue() call for Gradio v4; concurrency_count was removed.
    demo.queue().launch(
        server_name="127.0.0.1",   # bind to loopback
        server_port=7860,
        share=False,
        show_error=True,
        inbrowser=True
    )
