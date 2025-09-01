#!/usr/bin/env python3
"""
Preprocess the tooth-restoration tabular dataset for the Dentist Mawada project.

- Input  (auto-detected):  <project_root>/data/excel/data.xlsx  (or data.xlxs / data.xls)
- Output (same folder):    data_processed.xlsx, data_processed.csv

Pipeline (single run):
  • Standardize categorical fields IN-PLACE to numeric encodings.
  • Compute: p_indirect, y_majority, weight.
  • Add train/test split with exactly 80 test samples (random, reproducible).

Usage:
  python src/preprocessing/preprocess_dataset.py
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

# ---- Split configuration ----
TEST_COUNT = 80
SEED = 42  # change if you want a different random split


# ---------- Path discovery (relative to this script) ----------
def find_input_file() -> Path:
    script_path = Path(__file__).resolve()
    project_root = script_path.parents[2]  # .../multimodal-tooth-restoration-ai
    data_dir = project_root / "data" / "excel"

    candidates = [
        data_dir / "data.xlsx",
        data_dir / "data.xlxs",  # common typo
        data_dir / "data.xls",
    ]
    candidates += sorted(data_dir.glob("data.*xls*"))  # fallback

    for p in candidates:
        if p.exists():
            return p

    raise FileNotFoundError(
        f"Could not find input file in {data_dir}. Expected 'data.xlsx' (or 'data.xlxs'/'data.xls')."
    )


# ---------- Helpers for robust string mapping ----------
def _norm(s: Optional[Union[str, float, int]]) -> str:
    """Normalize strings for matching."""
    if pd.isna(s):
        return ""
    s = str(s).strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = s.replace("≤", "<=").replace("≥", ">=").replace("–", "-").replace("—", "-")
    s = s.replace("mm", " mm")
    s = re.sub(r"\s+", " ", s)
    return s


def map_depth(val):
    s = _norm(val)
    if not s:
        return pd.NA
    if re.search(r"(>|\bgreater)\s*=?\s*4\s*mm", s):
        return 1
    if re.search(r"(<=|<|≤|\ble?\b)\s*=?\s*4\s*mm", s):
        return 0
    m = re.search(r"(\d+(?:\.\d+)?)\s*mm", s)
    if m:
        try:
            return 1 if float(m.group(1)) > 4.0 else 0
        except Exception:
            return pd.NA
    return pd.NA


def map_width(val):
    s = _norm(val)
    if not s:
        return pd.NA
    if "all" in s and ("1 mm" in s or ">= 1 mm" in s or ">=1 mm" in s):
        return 1
    if "some" in s and ("< 1 mm" in s or "<1 mm" in s or "<1mm" in s):
        return 0
    if re.search(r"(>=|>)\s*1\s*mm", s):
        return 1
    if re.search(r"(<|<=)\s*1\s*mm", s):
        return 0
    return pd.NA


def map_yes_no(val):
    s = _norm(val)
    if not s:
        return pd.NA
    if s in {"yes", "y", "present", "presence", "true", "1"}:
        return 1
    if s in {"no", "n", "absent", "absence", "false", "0"}:
        return 0
    return pd.NA


def map_carious_lesion(val):
    s = _norm(val)
    if not s:
        return pd.NA
    if "low" in s:
        return -1
    if "moderate" in s or "medium" in s:
        return 0
    if "high" in s:
        return 1
    return pd.NA


def map_opposing_type(val):
    s = _norm(val)
    if not s:
        return pd.NA
    if "natural" in s:
        return 0
    if "missing" in s or "none" in s:
        return 1
    if "fpd" in s or "fixed partial denture" in s:
        return 2
    if "implant" in s:
        return 3
    return pd.NA


def map_adjacent_teeth(val):
    s = _norm(val)
    if not s:
        return pd.NA
    if "presence from one side" in s or "one side" in s:
        return 0
    if "presence" in s or "present" in s:
        return 1
    return pd.NA


def map_age_range(val):
    s = _norm(val).replace("&", "")
    if not s:
        return pd.NA
    if "< 20" in s or "<20" in s:
        return 0
    if "20-60" in s or ">= 20" in s or "≥ 20" in s or "20 - 60" in s:
        return 1
    m = re.search(r"(\d+)\s*-\s*(\d+)", s)
    if m:
        lo, hi = int(m.group(1)), int(m.group(2))
        return 1 if lo >= 20 and hi >= 60 else 0
    return pd.NA


# ---------- Core processing ----------
def compute_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert Direct/Indirect to numeric counts (missing -> 0), then compute:
      p_indirect = Indirect / (Direct + Indirect)   (NaN -> 0.0)
      y_majority = 1 if p_indirect >= 0.5 else 0
      weight     = |2*p_indirect - 1|               (NaN -> 0.0)
    """
    # Make numeric and treat blanks as 0 counts
    df["Direct"] = pd.to_numeric(df.get("Direct"), errors="coerce").fillna(0)
    df["Indirect"] = pd.to_numeric(df.get("Indirect"), errors="coerce").fillna(0)

    total = df["Direct"] + df["Indirect"]

    # Avoid division-by-zero NaNs; any remaining NaN becomes 0.0
    p_indirect = (df["Indirect"] / total).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    p_indirect = p_indirect.clip(0.0, 1.0)  # safety

    df["p_indirect"] = p_indirect.astype(float)
    df["y_majority"] = (df["p_indirect"] >= 0.5).astype("Int64")
    df["weight"] = (df["p_indirect"] * 2 - 1).abs().fillna(0.0)

    return df


def process_inplace(df: pd.DataFrame) -> pd.DataFrame:
    # Columns updated in place
    mappers = {
        "depth": map_depth,
        "width": map_width,
        "enamel_cracks": map_yes_no,
        "occlusal_load": map_yes_no,
        "carious_lesion": map_carious_lesion,
        "opposing_type": map_opposing_type,
        "adjacent_teeth": map_adjacent_teeth,
        "age_range": map_age_range,
        "cervical_lesion": map_yes_no,
    }

    for col, func in mappers.items():
        if col not in df.columns:
            raise KeyError(f"Missing required column: '{col}'")
        df[col] = df[col].apply(func).astype("Int64")

    df = compute_targets(df)
    return df


def add_split(df: pd.DataFrame, test_count: int = TEST_COUNT, seed: int = SEED) -> pd.DataFrame:
    n = len(df)
    k = min(test_count, n)
    rng = np.random.default_rng(seed)
    test_idx = rng.choice(n, size=k, replace=False)
    split = np.array(["train"] * n, dtype=object)
    split[test_idx] = "test"
    df["split"] = split
    return df


def main() -> int:
    in_path = find_input_file()
    out_xlsx = in_path.with_name("data_processed.xlsx")
    out_csv = in_path.with_name("data_processed.csv")

    # Read
    if in_path.suffix.lower() in {".xlsx", ".xls", ".xlxs"}:
        df = pd.read_excel(in_path)
    else:
        df = pd.read_csv(in_path)

    # Process and split
    df = process_inplace(df)
    df = add_split(df, TEST_COUNT, SEED)

    # Save
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="processed")
    df.to_csv(out_csv, index=False)

    print(f"Input : {in_path}")
    print(f"Output: {out_xlsx}")
    print(f"Output: {out_csv}")
    print(df["split"].value_counts().to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
