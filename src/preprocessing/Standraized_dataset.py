#!/usr/bin/env python3  # Shebang line to specify Python 3 interpreter for Unix-like systems
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
from __future__ import annotations  # Enable postponed evaluation of annotations for type hints (Python 3.10+ feature backported)

import re  # Regular expression module for pattern matching in strings
from pathlib import Path  # Object-oriented filesystem path handling
from typing import Optional, Union  # Type hints for optional parameters and union types

import numpy as np  # Numerical computing library for array operations and mathematical functions
import pandas as pd  # Data manipulation and analysis library for structured data

# ---- Split configuration ----
TEST_COUNT = 80  # Number of samples to randomly assign to test set (remaining samples go to training set)
SEED = 42  # Random seed for reproducible train/test splits across different runs


# ---------- Path discovery (relative to this script) ----------
def find_input_file() -> Path:  # Function to automatically locate the input data file, returns Path object
    script_path = Path(__file__).resolve()  # Get absolute path of current script file
    project_root = script_path.parents[2]  # Navigate up 2 directories to reach project root (.../multimodal-tooth-restoration-ai)
    data_dir = project_root / "data" / "excel"  # Construct path to data directory using Path object

    candidates = [  # List of potential input file names to search for
        data_dir / "data.xlsx",  # Standard Excel format
        data_dir / "data.xlxs",  # Common typo in Excel extension
        data_dir / "data.xls",   # Legacy Excel format
    ]
    candidates += sorted(data_dir.glob("data.*xls*"))  # Add any other Excel files starting with "data" as fallback options

    for p in candidates:  # Iterate through candidate file paths
        if p.exists():  # Check if current candidate file exists on filesystem
            return p  # Return first existing file found

    raise FileNotFoundError(  # Raise exception if no valid input file is found
        f"Could not find input file in {data_dir}. Expected 'data.xlsx' (or 'data.xlxs'/'data.xls')."
    )


# ---------- Helpers for robust string mapping ----------
def _norm(s: Optional[Union[str, float, int]]) -> str:  # Normalize input values to standardized string format
    """Normalize strings for matching."""
    if pd.isna(s):  # Check if input is NaN/None/missing using pandas
        return ""  # Return empty string for missing values
    s = str(s).strip().lower()  # Convert to string, remove leading/trailing whitespace, convert to lowercase
    s = re.sub(r"\s+", " ", s)  # Replace multiple consecutive whitespace characters with single space
    s = s.replace("≤", "<=").replace("≥", ">=").replace("–", "-").replace("—", "-")  # Normalize special characters to ASCII equivalents
    s = s.replace("mm", " mm")  # Add space before "mm" unit for consistent parsing
    s = re.sub(r"\s+", " ", s)  # Clean up any double spaces created by previous operations
    return s  # Return normalized string


def map_depth(val):  # Convert cavity depth descriptions to binary encoding (0: ≤4mm, 1: >4mm)
    s = _norm(val)  # Normalize input string using helper function
    if not s:  # Check if string is empty after normalization
        return pd.NA  # Return pandas NA for missing/empty values
    if re.search(r"(>|\bgreater)\s*=?\s*4\s*mm", s):  # Search for patterns indicating depth greater than 4mm
        return 1  # Return 1 for deep cavities (>4mm)
    if re.search(r"(<=|<|≤|\ble?\b)\s*=?\s*4\s*mm", s):  # Search for patterns indicating depth less than or equal to 4mm
        return 0  # Return 0 for shallow cavities (≤4mm)
    m = re.search(r"(\d+(?:\.\d+)?)\s*mm", s)  # Extract numeric value followed by mm unit
    if m:  # If numeric match is found
        try:
            return 1 if float(m.group(1)) > 4.0 else 0  # Compare extracted number to 4.0 threshold
        except Exception:  # Handle any conversion errors
            return pd.NA  # Return NA if conversion fails
    return pd.NA  # Return NA if no recognizable pattern is found


def map_width(val):  # Convert remaining tooth width descriptions to binary encoding (0: <1mm, 1: ≥1mm)
    s = _norm(val)  # Normalize input string
    if not s:  # Check for empty string
        return pd.NA  # Return NA for missing values
    if "all" in s and ("1 mm" in s or ">= 1 mm" in s or ">=1 mm" in s):  # Check for "all remaining walls ≥1mm"
        return 1  # Return 1 for adequate remaining tooth structure
    if "some" in s and ("< 1 mm" in s or "<1 mm" in s or "<1mm" in s):  # Check for "some walls <1mm"
        return 0  # Return 0 for inadequate remaining tooth structure
    if re.search(r"(>=|>)\s*1\s*mm", s):  # Search for patterns indicating width ≥1mm
        return 1  # Return 1 for adequate width
    if re.search(r"(<|<=)\s*1\s*mm", s):  # Search for patterns indicating width <1mm
        return 0  # Return 0 for inadequate width
    return pd.NA  # Return NA if no pattern matches


def map_yes_no(val):  # Convert yes/no categorical values to binary encoding (0: no/absent, 1: yes/present)
    s = _norm(val)  # Normalize input string
    if not s:  # Check for empty string
        return pd.NA  # Return NA for missing values
    if s in {"yes", "y", "present", "presence", "true", "1"}:  # Check for positive indicators
        return 1  # Return 1 for yes/present/true values
    if s in {"no", "n", "absent", "absence", "false", "0"}:  # Check for negative indicators
        return 0  # Return 0 for no/absent/false values
    return pd.NA  # Return NA if value doesn't match expected patterns


def map_carious_lesion(val):  # Convert caries risk levels to ordinal encoding (-1: low, 0: moderate, 1: high)
    s = _norm(val)  # Normalize input string
    if not s:  # Check for empty string
        return pd.NA  # Return NA for missing values
    if "low" in s:  # Check for low risk indicators
        return -1  # Return -1 for low caries risk
    if "moderate" in s or "medium" in s:  # Check for moderate risk indicators
        return 0  # Return 0 for moderate caries risk
    if "high" in s:  # Check for high risk indicators
        return 1  # Return 1 for high caries risk
    return pd.NA  # Return NA if risk level not recognized


def map_opposing_type(val):  # Convert opposing tooth type to categorical encoding (0: natural, 1: missing, 2: FPD, 3: implant)
    s = _norm(val)  # Normalize input string
    if not s:  # Check for empty string
        return pd.NA  # Return NA for missing values
    if "natural" in s:  # Check for natural opposing tooth
        return 0  # Return 0 for natural tooth opposing
    if "missing" in s or "none" in s:  # Check for missing opposing tooth
        return 1  # Return 1 for missing opposing tooth
    if "fpd" in s or "fixed partial denture" in s:  # Check for fixed partial denture opposing
        return 2  # Return 2 for FPD opposing
    if "implant" in s:  # Check for implant opposing
        return 3  # Return 3 for implant opposing
    return pd.NA  # Return NA if opposing type not recognized


def map_adjacent_teeth(val):  # Convert adjacent teeth presence to binary encoding (0: one side, 1: both sides)
    s = _norm(val)  # Normalize input string
    if not s:  # Check for empty string
        return pd.NA  # Return NA for missing values
    if "presence from one side" in s or "one side" in s:  # Check for adjacent teeth on one side only
        return 0  # Return 0 for partial adjacent support
    if "presence" in s or "present" in s:  # Check for general presence (implies both sides)
        return 1  # Return 1 for full adjacent support
    return pd.NA  # Return NA if adjacent status not recognized


def map_age_range(val):  # Convert age ranges to binary encoding (0: <20 years, 1: ≥20 years)
    s = _norm(val).replace("&", "")  # Normalize string and remove ampersand characters
    if not s:  # Check for empty string
        return pd.NA  # Return NA for missing values
    if "< 20" in s or "<20" in s:  # Check for age less than 20 years
        return 0  # Return 0 for young patients
    if "20-60" in s or ">= 20" in s or "≥ 20" in s or "20 - 60" in s:  # Check for age 20 years or older
        return 1  # Return 1 for adult patients
    m = re.search(r"(\d+)\s*-\s*(\d+)", s)  # Extract age range numbers (e.g., "25-45")
    if m:  # If age range pattern is found
        lo, hi = int(m.group(1)), int(m.group(2))  # Extract lower and upper bounds as integers
        return 1 if lo >= 20 and hi >= 60 else 0  # Return 1 if range includes adults, 0 otherwise
    return pd.NA  # Return NA if age pattern not recognized


# ---------- Core processing ----------
def compute_targets(df: pd.DataFrame) -> pd.DataFrame:  # Calculate target variables from Direct/Indirect expert consensus counts
    """
    Convert Direct/Indirect to numeric counts (missing -> 0), then compute:
      p_indirect = Indirect / (Direct + Indirect)   (NaN -> 0.0)
      y_majority = 1 if p_indirect >= 0.5 else 0
      weight     = |2*p_indirect - 1|               (NaN -> 0.0)
    """
    # Make numeric and treat blanks as 0 counts
    df["Direct"] = pd.to_numeric(df.get("Direct"), errors="coerce").fillna(0)  # Convert Direct column to numeric, replace missing with 0
    df["Indirect"] = pd.to_numeric(df.get("Indirect"), errors="coerce").fillna(0)  # Convert Indirect column to numeric, replace missing with 0

    total = df["Direct"] + df["Indirect"]  # Calculate total expert responses for each case

    # Avoid division-by-zero NaNs; any remaining NaN becomes 0.0
    p_indirect = (df["Indirect"] / total).replace([np.inf, -np.inf], np.nan).fillna(0.0)  # Calculate proportion favoring indirect restoration, handle division by zero
    p_indirect = p_indirect.clip(0.0, 1.0)  # Ensure probabilities are within valid range [0,1]

    df["p_indirect"] = p_indirect.astype(float)  # Store proportion as continuous target variable
    df["y_majority"] = (df["p_indirect"] >= 0.5).astype("Int64")  # Create binary target: 1 if majority favors indirect, 0 otherwise
    df["weight"] = (df["p_indirect"] * 2 - 1).abs().fillna(0.0)  # Calculate consensus weight: higher for strong agreement (near 0 or 1), lower for uncertainty (near 0.5)

    return df  # Return dataframe with added target columns


def process_inplace(df: pd.DataFrame) -> pd.DataFrame:  # Apply all categorical mappings and compute target variables
    # Columns updated in place
    mappers = {  # Dictionary mapping column names to their respective transformation functions
        "depth": map_depth,  # Cavity depth mapping function
        "width": map_width,  # Remaining tooth width mapping function
        "enamel_cracks": map_yes_no,  # Enamel cracks presence mapping function
        "occlusal_load": map_yes_no,  # Occlusal load presence mapping function
        "carious_lesion": map_carious_lesion,  # Caries risk level mapping function
        "opposing_type": map_opposing_type,  # Opposing tooth type mapping function
        "adjacent_teeth": map_adjacent_teeth,  # Adjacent teeth presence mapping function
        "age_range": map_age_range,  # Age range mapping function
        "cervical_lesion": map_yes_no,  # Cervical lesion presence mapping function
    }

    for col, func in mappers.items():  # Iterate through each column and its mapping function
        if col not in df.columns:  # Check if required column exists in dataframe
            raise KeyError(f"Missing required column: '{col}'")  # Raise error if column is missing
        df[col] = df[col].apply(func).astype("Int64")  # Apply mapping function and convert to nullable integer type

    df = compute_targets(df)  # Calculate target variables from expert consensus data
    return df  # Return processed dataframe


def add_split(df: pd.DataFrame, test_count: int = TEST_COUNT, seed: int = SEED) -> pd.DataFrame:  # Add train/test split column with specified number of test samples
    n = len(df)  # Get total number of rows in dataframe
    k = min(test_count, n)  # Ensure test count doesn't exceed total samples
    rng = np.random.default_rng(seed)  # Create random number generator with fixed seed for reproducibility
    test_idx = rng.choice(n, size=k, replace=False)  # Randomly select k indices for test set without replacement
    split = np.array(["train"] * n, dtype=object)  # Create array with all samples initially labeled as "train"
    split[test_idx] = "test"  # Change selected indices to "test" label
    df["split"] = split  # Add split column to dataframe
    return df  # Return dataframe with split information


def main() -> int:  # Main function that orchestrates the entire preprocessing pipeline
    in_path = find_input_file()  # Automatically locate input data file
    out_xlsx = in_path.with_name("data_processed.xlsx")  # Create output path for Excel file with same directory as input
    out_csv = in_path.with_name("data_processed.csv")  # Create output path for CSV file with same directory as input

    # Read
    if in_path.suffix.lower() in {".xlsx", ".xls", ".xlxs"}:  # Check if input file is Excel format
        df = pd.read_excel(in_path)  # Read Excel file into pandas dataframe
    else:  # If not Excel format
        df = pd.read_csv(in_path)  # Read CSV file into pandas dataframe

    # Process and split
    df = process_inplace(df)  # Apply all categorical mappings and compute targets
    df = add_split(df, TEST_COUNT, SEED)  # Add train/test split with specified configuration

    # Save
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:  # Create Excel writer context manager
        df.to_excel(writer, index=False, sheet_name="processed")  # Write dataframe to Excel file without row indices
    df.to_csv(out_csv, index=False)  # Write dataframe to CSV file without row indices

    print(f"Input : {in_path}")  # Print input file path for confirmation
    print(f"Output: {out_xlsx}")  # Print Excel output file path
    print(f"Output: {out_csv}")  # Print CSV output file path
    print(df["split"].value_counts().to_dict())  # Print distribution of train/test samples
    return 0  # Return success code


if __name__ == "__main__":  # Check if script is being run directly (not imported)
    raise SystemExit(main())  # Execute main function and exit with its return code
