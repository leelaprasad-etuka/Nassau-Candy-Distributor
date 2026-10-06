"""Stage 1: Data ingestion, text cleaning, financial integrity, and IQR outlier rejection.

Implements Stage 1 of the Nassau Candy Distributor optimization pipeline.
Validates accounting identity, handles relative lead time indexing, and filters extreme outliers.
"""

from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from src.config import (
    CLEANED_DATA_PATH,
    IQR_MULTIPLIER,
    PRODUCT_NAME_CLEAN_MAP,
    RAW_DATA_PATH,
)


def load_raw_dataset(filepath: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load raw Nassau Candy Distributor dataset.

    Args:
        filepath: Path to the raw CSV file.

    Returns:
        Raw DataFrame.
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Raw data file not found at: {filepath}")
    return pd.read_csv(filepath)


def clean_text_and_validate_financials(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Clean text whitespace, standardize product names, and validate financials.

    Validates that Gross Profit == Sales - Cost with zero discrepancies.

    Args:
        df: Input DataFrame.

    Returns:
        Tuple of (Cleaned DataFrame, financial validation summary).
    """
    df_clean = df.copy()

    # Clean object/string columns
    string_cols = df_clean.select_dtypes(include=["object", "string"]).columns
    for col in string_cols:
        df_clean[col] = df_clean[col].astype(str).str.strip()

    # Standardize product names
    for raw_name, clean_name in PRODUCT_NAME_CLEAN_MAP.items():
        df_clean["Product Name"] = df_clean["Product Name"].replace(raw_name, clean_name)

    # Validate financial identity: Gross Profit = Sales - Cost
    profit_discrepancy = np.abs(
        df_clean["Gross Profit"] - (df_clean["Sales"] - df_clean["Cost"])
    )
    max_discrepancy = float(profit_discrepancy.max())
    discrepancy_count = int((profit_discrepancy > 0.01).sum())

    validation_summary = {
        "max_financial_discrepancy": max_discrepancy,
        "discrepant_rows_count": discrepancy_count,
        "financial_integrity_passed": float(discrepancy_count == 0),
    }

    return df_clean, validation_summary


def derive_lead_time_index(df: pd.DataFrame) -> pd.DataFrame:
    """Parse dd-mm-yyyy dates and compute lead_time_days index.

    Data limitation note:
    Orders date from 2024 to 2025 while ship dates fall from 2026 to 2030,
    generating large lead times (904 to 1,642 days). These are retained
    unmodified as a relative lead time index.

    Args:
        df: Cleaned DataFrame.

    Returns:
        DataFrame with derived lead_time_days.
    """
    df_parsed = df.copy()
    df_parsed["Order Date"] = pd.to_datetime(df_parsed["Order Date"], format="%d-%m-%Y")
    df_parsed["Ship Date"] = pd.to_datetime(df_parsed["Ship Date"], format="%d-%m-%Y")
    df_parsed["lead_time_days"] = (
        df_parsed["Ship Date"] - df_parsed["Order Date"]
    ).dt.days

    return df_parsed


def remove_extreme_outliers_iqr(
    df: pd.DataFrame,
    cols: List[str] = ["lead_time_days", "Sales", "Units"],
    multiplier: float = IQR_MULTIPLIER,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Remove extreme outliers using the Interquartile Range (IQR) rule.

    Calculates Q1 and Q3, removes observations outside [Q1 - multiplier*IQR, Q3 + multiplier*IQR],
    and logs outlier counts per evaluated column.

    Args:
        df: Input DataFrame.
        cols: Columns to evaluate.
        multiplier: IQR multiplier (default 1.5).

    Returns:
        Tuple of (Filtered DataFrame, outlier audit dict).
    """
    df_filtered = df.copy()
    initial_rows = len(df_filtered)
    audit: Dict[str, int] = {"initial_rows": initial_rows}

    combined_mask = pd.Series(True, index=df_filtered.index)

    for col in cols:
        q1 = df_filtered[col].quantile(0.25)
        q3 = df_filtered[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - multiplier * iqr
        upper_bound = q3 + multiplier * iqr

        col_mask = (df_filtered[col] >= lower_bound) & (df_filtered[col] <= upper_bound)
        flagged = int((~col_mask).sum())
        audit[f"{col}_outliers_flagged"] = flagged
        combined_mask = combined_mask & col_mask

    df_filtered = df_filtered[combined_mask].reset_index(drop=True)
    audit["total_rows_removed"] = initial_rows - len(df_filtered)
    audit["final_rows_retained"] = len(df_filtered)

    return df_filtered, audit


def run_stage_1(
    raw_path: Path = RAW_DATA_PATH, save_output: bool = True
) -> Tuple[pd.DataFrame, Dict]:
    """Execute end-to-end Stage 1 data preparation.

    Args:
        raw_path: Path to raw CSV.
        save_output: Whether to persist cleaned DataFrame.

    Returns:
        Tuple of (Cleaned DataFrame, Stage 1 verification summary).
    """
    df_raw = load_raw_dataset(raw_path)
    df_clean, val_summary = clean_text_and_validate_financials(df_raw)
    df_parsed = derive_lead_time_index(df_clean)
    df_ready, outlier_audit = remove_extreme_outliers_iqr(df_parsed)

    if save_output:
        CLEANED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        df_ready.to_csv(CLEANED_DATA_PATH, index=False)

    summary = {
        "raw_rows": len(df_raw),
        "cleaned_rows": len(df_ready),
        "validation": val_summary,
        "outlier_audit": outlier_audit,
    }

    return df_ready, summary


if __name__ == "__main__":
    df_out, s1_summary = run_stage_1()
    print("STAGE 1 VERIFICATION SUMMARY:")
    print(f"Raw rows: {s1_summary['raw_rows']}")
    print(f"Cleaned rows: {s1_summary['cleaned_rows']}")
    print(f"Financial validation: {s1_summary['validation']}")
    print(f"Outlier audit: {s1_summary['outlier_audit']}")
