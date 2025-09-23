# Gradio UI for Dental Restoration Classifier

This UI wraps your final **hybrid** system (MM + MIL + Tab) to classify **Direct vs Indirect** restorations for single cases.

## Features
- ✅ Image upload (required) with **resolution check** (≥ 512×512)
- ✅ Optional **clinical fields** (all-or-none) to enable full hybrid
- ✅ Runs your **segmentation pipeline** before inference
- ✅ Uses trained fold checkpoints:
  - `weights/mm_dualtask_v1/mm_dualtask_fold*.pt`
  - `weights/mil_v1/mil_v1_fold*.pt`
- ✅ Trains a LightGBM **tabular K-fold** ensemble and a **2-stream image meta** (MM+MIL) on your OOF at startup
- ✅ Blue, clinic-friendly theme

## Layout
