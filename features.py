"""Stage 2: Feature engineering and spatial bridge construction.

Implements Stage 2 of the Nassau Candy Distributor optimization pipeline.
Calculates great-circle distance_km, assigns historical manufacturing plants,
derives profit margins, calendar attributes, route keys, and prepares training matrices.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.config import (
    CLEANED_DATA_PATH,
    FACTORIES,
    PROCESSED_DATA_PATH,
    PRODUCT_FACTORY_MAP,
)
from src.geo import get_centroid, haversine_distance


def engineer_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Engineer spatial distance, logistical, financial, and temporal features.

    Args:
        df: Cleaned DataFrame from Stage 1.

    Returns:
        Tuple of (Processed DataFrame, audit metadata dict).
    """
    df_feat = df.copy()

    # 1. Historical Plant Assignment
    df_feat["assigned_factory"] = df_feat["Product Name"].map(PRODUCT_FACTORY_MAP)
    unmapped = df_feat[df_feat["assigned_factory"].isna()]["Product Name"].unique().tolist()
    if unmapped:
        raise ValueError(f"Unmapped product SKU encountered: {unmapped}")

    # 2. Customer Destination Centroid Mapping
    coords_series = df_feat["State/Province"].apply(get_centroid)
    unmatched_states = df_feat[coords_series.isna()]["State/Province"].unique().tolist()

    df_feat["dest_latitude"] = coords_series.apply(
        lambda c: c[0] if c is not None else np.nan
    )
    df_feat["dest_longitude"] = coords_series.apply(
        lambda c: c[1] if c is not None else np.nan
    )

    # 3. Plant Coordinates
    df_feat["factory_latitude"] = df_feat["assigned_factory"].apply(
        lambda f: FACTORIES[f][0] if f in FACTORIES else np.nan
    )
    df_feat["factory_longitude"] = df_feat["assigned_factory"].apply(
        lambda f: FACTORIES[f][1] if f in FACTORIES else np.nan
    )

    # 4. Spatial Great-Circle Haversine Distance (km)
    df_feat["distance_km"] = haversine_distance(
        df_feat["factory_latitude"].values,
        df_feat["factory_longitude"].values,
        df_feat["dest_latitude"].values,
        df_feat["dest_longitude"].values,
    )

    # 5. Financial Margins
    df_feat["margin_percent"] = (
        df_feat["Gross Profit"] / df_feat["Sales"].replace(0, np.nan)
    ) * 100.0
    df_feat["profit_per_unit"] = (
        df_feat["Gross Profit"] / df_feat["Units"].replace(0, np.nan)
    )

    # 6. Temporal Features (from Order Date only, no Ship Date leakage)
    if not pd.api.types.is_datetime64_any_dtype(df_feat["Order Date"]):
        df_feat["Order Date"] = pd.to_datetime(df_feat["Order Date"], format="%d-%m-%Y")

    df_feat["order_month"] = df_feat["Order Date"].dt.month
    df_feat["order_day_of_week"] = df_feat["Order Date"].dt.dayofweek

    # 7. Discrete Route Key
    df_feat["route_key"] = (
        df_feat["assigned_factory"] + " -> " + df_feat["State/Province"]
    )

    audit = {
        "total_records": len(df_feat),
        "unmatched_states": unmatched_states,
        "mean_distance_km": float(df_feat["distance_km"].mean()),
        "min_distance_km": float(df_feat["distance_km"].min()),
        "max_distance_km": float(df_feat["distance_km"].max()),
        "unique_routes": int(df_feat["route_key"].nunique()),
    }

    return df_feat, audit


def run_stage_2(
    input_path: Path = CLEANED_DATA_PATH, save_output: bool = True
) -> Tuple[pd.DataFrame, Dict]:
    """Execute end-to-end Stage 2 feature engineering.

    Args:
        input_path: Path to cleaned CSV.
        save_output: Whether to persist processed DataFrame.

    Returns:
        Tuple of (Processed DataFrame, audit summary).
    """
    if not input_path.exists():
        from src.data_prep import run_stage_1

        df_clean, _ = run_stage_1()
    else:
        df_clean = pd.read_csv(input_path)
        if "Order Date" in df_clean.columns:
            df_clean["Order Date"] = pd.to_datetime(df_clean["Order Date"])

    df_processed, audit = engineer_features(df_clean)

    if save_output:
        PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        df_processed.to_csv(PROCESSED_DATA_PATH, index=False)

    return df_processed, audit


if __name__ == "__main__":
    df_out, s2_audit = run_stage_2()
    print("STAGE 2 VERIFICATION SUMMARY:")
    print(f"Total processed rows: {s2_audit['total_records']}")
    print(f"Unmatched states: {s2_audit['unmatched_states']}")
    print(f"Mean distance: {s2_audit['mean_distance_km']:.2f} km")
    print(f"Distance range: [{s2_audit['min_distance_km']:.2f}, {s2_audit['max_distance_km']:.2f}] km")
    print(f"Unique routes: {s2_audit['unique_routes']}")
