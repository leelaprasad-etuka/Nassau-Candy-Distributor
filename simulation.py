"""Stage 5: Scenario simulation engine and uncertainty estimation.

Implements Stage 5 of the Nassau Candy Distributor optimization pipeline.
Evaluates 17,700 permutations of product, destination, and ship mode across all 5
candidate facilities. Recomputes distance, calculates like-for-like model predictions,
freight changes, adjusted profits, and tree variance uncertainty.
"""

from pathlib import Path
from typing import Dict, List, Tuple
import joblib
import numpy as np
import pandas as pd

from src.config import (
    BEST_MODEL_PATH,
    DEFAULT_FREIGHT_RATE_PER_UNIT_KM,
    ENCODERS_PATH,
    FACTORIES,
    PRECOMPUTED_SCENARIO_PATH,
    PROCESSED_DATA_PATH,
    PRODUCT_FACTORY_MAP,
    SCALER_PATH,
)
from src.geo import get_centroid, haversine_distance


def build_simulation_grid(df: pd.DataFrame) -> pd.DataFrame:
    """Build unique permutations of product, destination state, and ship mode.

    Args:
        df: Processed DataFrame.

    Returns:
        DataFrame containing grid tuples with baseline economic metrics.
    """
    prod_profile = (
        df.groupby(["Product Name", "Division"])
        .agg(
            base_units=("Units", "mean"),
            base_sales=("Sales", "mean"),
            base_cost=("Cost", "mean"),
            base_profit=("Gross Profit", "mean"),
            product_sample_size=("Units", "count"),
        )
        .reset_index()
    )

    state_region_map = (
        df.groupby("State/Province")["Region"].agg(lambda x: x.mode()[0]).to_dict()
    )

    unique_states = sorted(list(state_region_map.keys()))
    unique_ship_modes = sorted(df["Ship Mode"].unique().tolist())
    median_month = int(df["order_month"].median())

    grid_records = []
    for _, prod_row in prod_profile.iterrows():
        product_name = prod_row["Product Name"]
        division = prod_row["Division"]
        current_factory = PRODUCT_FACTORY_MAP.get(product_name)

        for state in unique_states:
            region = state_region_map[state]
            dest_coords = get_centroid(state)
            if dest_coords is None:
                continue

            for ship_mode in unique_ship_modes:
                grid_records.append({
                    "Product Name": product_name,
                    "Division": division,
                    "current_factory": current_factory,
                    "State/Province": state,
                    "Region": region,
                    "Ship Mode": ship_mode,
                    "dest_latitude": dest_coords[0],
                    "dest_longitude": dest_coords[1],
                    "base_units": float(prod_row["base_units"]),
                    "base_sales": float(prod_row["base_sales"]),
                    "base_cost": float(prod_row["base_cost"]),
                    "base_profit": float(prod_row["base_profit"]),
                    "product_sample_size": int(prod_row["product_sample_size"]),
                    "order_month": median_month,
                })

    return pd.DataFrame(grid_records)


def simulate_all_factory_allocations(
    grid_df: pd.DataFrame,
    freight_rate_per_unit_km: float = DEFAULT_FREIGHT_RATE_PER_UNIT_KM,
) -> pd.DataFrame:
    """Simulate assigning each grid row to each of the 5 factories.

    Performs like-for-like predictions and calculates lead time reductions,
    freight changes, adjusted profit changes, and tree variance dispersion.

    Args:
        grid_df: Base grid DataFrame.
        freight_rate_per_unit_km: Freight rate ($/unit/km).

    Returns:
        Complete simulated scenarios DataFrame.
    """
    model = joblib.load(BEST_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    encoder = joblib.load(ENCODERS_PATH)

    factory_names = list(FACTORIES.keys())
    categorical_cols = ["Ship Mode", "Region", "Division"]

    sim_rows = []
    for f_name in factory_names:
        f_lat, f_lon = FACTORIES[f_name]
        f_df = grid_df.copy()
        f_df["candidate_factory"] = f_name
        f_df["factory_latitude"] = f_lat
        f_df["factory_longitude"] = f_lon
        f_df["candidate_distance_km"] = haversine_distance(
            f_lat, f_lon, f_df["dest_latitude"].values, f_df["dest_longitude"].values
        )
        sim_rows.append(f_df)

    sim_df = pd.concat(sim_rows, ignore_index=True)

    cur_dists = []
    for _, row in sim_df.iterrows():
        c_lat, c_lon = FACTORIES[row["current_factory"]]
        d = haversine_distance(c_lat, c_lon, row["dest_latitude"], row["dest_longitude"])
        cur_dists.append(d)
    sim_df["current_distance_km"] = cur_dists

    # Features for candidate predictions
    X_cand_cat = encoder.transform(sim_df[categorical_cols])
    X_cand_num = np.column_stack([
        sim_df["candidate_distance_km"].values,
        sim_df["base_units"].values,
        sim_df["base_sales"].values,
        sim_df["order_month"].values,
    ])
    X_cand = scaler.transform(np.hstack([X_cand_num, X_cand_cat]))

    # Features for current factory predictions (like-for-like comparison)
    X_cur_cat = X_cand_cat
    X_cur_num = np.column_stack([
        sim_df["current_distance_km"].values,
        sim_df["base_units"].values,
        sim_df["base_sales"].values,
        sim_df["order_month"].values,
    ])
    X_cur = scaler.transform(np.hstack([X_cur_num, X_cur_cat]))

    sim_df["predicted_candidate_lead_time"] = model.predict(X_cand)
    sim_df["predicted_current_lead_time"] = model.predict(X_cur)

    # Uncertainty from ensemble tree spread
    if hasattr(model, "estimators_"):
        tree_preds = np.column_stack([
            estimator.predict(X_cand) for estimator in model.estimators_
        ])
        sim_df["prediction_uncertainty_std"] = np.std(tree_preds, axis=1)
    else:
        sim_df["prediction_uncertainty_std"] = 15.0

    # Lead time changes
    sim_df["lead_time_diff_days"] = (
        sim_df["predicted_current_lead_time"] - sim_df["predicted_candidate_lead_time"]
    )
    sim_df["lead_time_reduction_pct"] = (
        sim_df["lead_time_diff_days"] / sim_df["predicted_current_lead_time"]
    ) * 100.0

    # Freight and profit impact
    sim_df["distance_change_km"] = (
        sim_df["candidate_distance_km"] - sim_df["current_distance_km"]
    )
    sim_df["current_freight_cost"] = (
        sim_df["current_distance_km"] * sim_df["base_units"] * freight_rate_per_unit_km
    )
    sim_df["candidate_freight_cost"] = (
        sim_df["candidate_distance_km"] * sim_df["base_units"] * freight_rate_per_unit_km
    )
    sim_df["estimated_freight_change"] = (
        sim_df["candidate_freight_cost"] - sim_df["current_freight_cost"]
    )
    sim_df["adjusted_profit_change"] = -sim_df["estimated_freight_change"]
    sim_df["profit_impact_pct"] = (
        sim_df["adjusted_profit_change"] / sim_df["base_profit"].replace(0, np.nan)
    ) * 100.0

    sim_df["is_current_factory"] = (
        sim_df["candidate_factory"] == sim_df["current_factory"]
    )

    return sim_df


def run_stage_5(
    processed_path: Path = PROCESSED_DATA_PATH, save_output: bool = True
) -> Tuple[pd.DataFrame, Dict]:
    """Execute end-to-end Stage 5 scenario simulation.

    Args:
        processed_path: Path to processed CSV.
        save_output: Whether to write precomputed scenarios CSV.

    Returns:
        Tuple of (Simulated scenarios DataFrame, summary dict).
    """
    df = pd.read_csv(processed_path)
    grid_df = build_simulation_grid(df)
    sim_df = simulate_all_factory_allocations(grid_df)

    if save_output:
        PRECOMPUTED_SCENARIO_PATH.parent.mkdir(parents=True, exist_ok=True)
        sim_df.to_csv(PRECOMPUTED_SCENARIO_PATH, index=False)

    summary = {
        "total_simulations": len(sim_df),
        "products_simulated": int(sim_df["Product Name"].nunique()),
        "destination_states": int(sim_df["State/Province"].nunique()),
        "factories_evaluated": int(sim_df["candidate_factory"].nunique()),
        "mean_lead_time_reduction_pct": float(sim_df["lead_time_reduction_pct"].mean()),
        "mean_profit_impact_pct": float(sim_df["profit_impact_pct"].mean()),
        "mean_uncertainty_days": float(sim_df["prediction_uncertainty_std"].mean()),
    }

    return sim_df, summary


if __name__ == "__main__":
    simulations, s5_summary = run_stage_5()
    print("STAGE 5 VERIFICATION SUMMARY:")
    print(f"Total scenario simulations generated: {s5_summary['total_simulations']}")
    print(f"Products: {s5_summary['products_simulated']}")
    print(f"States: {s5_summary['destination_states']}")
    print(f"Factories: {s5_summary['factories_evaluated']}")
    print(f"Mean lead time reduction pct across all options: {s5_summary['mean_lead_time_reduction_pct']:.2f}%")
    print(f"Mean profit impact pct across all options: {s5_summary['mean_profit_impact_pct']:.2f}%")
    print(f"Mean uncertainty std dev: {s5_summary['mean_uncertainty_days']:.2f} days")
