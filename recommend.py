"""Stage 6: Multi-criteria optimization ranking and KPI recommendation engine.

Implements Stage 6 of the Nassau Candy Distributor optimization pipeline.
Ranks candidate factory allocations per SKU using a composite score balancing
speed vs profit with risk penalties for uncertainty and low sample size.
Computes four operational KPIs: Lead Time Reduction %, Profit Impact Stability,
Scenario Confidence Score, and Recommendation Coverage.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.config import (
    DEFAULT_FREIGHT_RATE_PER_UNIT_KM,
    PRECOMPUTED_SCENARIO_PATH,
    PROFIT_LOSS_TOLERANCE_PCT,
)


def compute_scenario_confidence_score(
    sample_size: Optional[int] = None,
    uncertainty_std: float = 15.0,
    max_sample_size: int = 2200,
    max_uncertainty: float = 50.0,
) -> float:
    """Calculate Scenario Confidence Score on a 0-100 scale.

    Formula incorporates:
      - Empirical sample size adequacy (logarithmic scaling)
      - Ensemble tree standard deviation (uncertainty dispersion)

    Args:
        sample_size: Historical SKU or route sample size.
        uncertainty_std: Prediction standard deviation across ensemble trees.
        max_sample_size: Reference catalog max sample size.
        max_uncertainty: Reference max tree standard deviation.

    Returns:
        Confidence score between 0.0 and 100.0.
    """
    safe_sample = max(1, int(sample_size)) if sample_size is not None else 100
    sample_factor = np.clip(np.log1p(safe_sample) / np.log1p(max_sample_size), 0.05, 1.0)
    uncertainty_factor = np.clip(1.0 - (uncertainty_std / max_uncertainty), 0.0, 1.0)

    confidence = (0.60 * sample_factor + 0.40 * uncertainty_factor) * 100.0
    return float(np.clip(confidence, 5.0, 99.5))


def rank_factory_recommendations(
    scenario_df: pd.DataFrame,
    speed_weight: float = 0.5,
    top_n: int = 3,
    profit_tolerance_pct: float = PROFIT_LOSS_TOLERANCE_PCT,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    """Rank factory allocations per product using multi-criteria composite scoring.

    Composite Score Formula:
      Score = w * norm_speed + (1 - w) * norm_profit - risk_penalty

    Where:
      - w: speed_weight in [0.0, 1.0].
      - norm_speed: min-max normalized lead_time_reduction_pct.
      - norm_profit: min-max normalized profit_impact_pct.
      - risk_penalty: weighted penalty for high uncertainty and sparse sample size.

    Args:
        scenario_df: Precomputed simulation DataFrame.
        speed_weight: Weight given to lead time reduction (0.0 to 1.0).
        top_n: Number of recommendations per SKU.
        profit_tolerance_pct: Maximum tolerable profit decrease percentage.

    Returns:
        Tuple of (Product-level ranked DataFrame, Top Recommendations DataFrame, KPIs dict).
    """
    df = scenario_df.copy()

    agg_df = (
        df.groupby([
            "Product Name",
            "Division",
            "current_factory",
            "candidate_factory",
        ])
        .agg(
            mean_lead_time_reduction_pct=("lead_time_reduction_pct", "mean"),
            mean_profit_impact_pct=("profit_impact_pct", "mean"),
            mean_freight_change=("estimated_freight_change", "mean"),
            mean_candidate_lead_time=("predicted_candidate_lead_time", "mean"),
            mean_current_lead_time=("predicted_current_lead_time", "mean"),
            mean_candidate_distance_km=("candidate_distance_km", "mean"),
            mean_current_distance_km=("current_distance_km", "mean"),
            mean_uncertainty_std=("prediction_uncertainty_std", "mean"),
            product_sample_size=("product_sample_size", "first"),
            std_profit_impact=("profit_impact_pct", "std"),
        )
        .reset_index()
    )
    agg_df["std_profit_impact"] = agg_df["std_profit_impact"].fillna(0.0)
    agg_df["is_current"] = agg_df["candidate_factory"] == agg_df["current_factory"]

    # Min-max normalization
    lt_min = agg_df["mean_lead_time_reduction_pct"].min()
    lt_max = agg_df["mean_lead_time_reduction_pct"].max()
    lt_denom = (lt_max - lt_min) if (lt_max - lt_min) > 1e-6 else 1.0
    agg_df["norm_lead_time"] = (agg_df["mean_lead_time_reduction_pct"] - lt_min) / lt_denom

    pr_min = agg_df["mean_profit_impact_pct"].min()
    pr_max = agg_df["mean_profit_impact_pct"].max()
    pr_denom = (pr_max - pr_min) if (pr_max - pr_min) > 1e-6 else 1.0
    agg_df["norm_profit"] = (agg_df["mean_profit_impact_pct"] - pr_min) / pr_denom

    unc_max = agg_df["mean_uncertainty_std"].max()
    unc_denom = unc_max if unc_max > 1e-6 else 1.0
    agg_df["norm_uncertainty"] = agg_df["mean_uncertainty_std"] / unc_denom

    max_samples = agg_df["product_sample_size"].max()
    agg_df["sample_penalty"] = 1.0 - (
        np.log1p(agg_df["product_sample_size"]) / np.log1p(max_samples)
    )
    agg_df["risk_penalty"] = (
        0.10 * agg_df["norm_uncertainty"] + 0.15 * agg_df["sample_penalty"]
    )

    # Composite Score
    agg_df["composite_score"] = (
        speed_weight * agg_df["norm_lead_time"]
        + (1.0 - speed_weight) * agg_df["norm_profit"]
        - agg_df["risk_penalty"]
    )

    # Scenario Confidence Score (0-100)
    conf_scores = [
        compute_scenario_confidence_score(
            sample_size=row["product_sample_size"],
            uncertainty_std=row["mean_uncertainty_std"],
            max_sample_size=max_samples,
        )
        for _, row in agg_df.iterrows()
    ]
    agg_df["confidence_score"] = conf_scores

    agg_df["rank"] = (
        agg_df.groupby("Product Name")["composite_score"]
        .rank(ascending=False, method="dense")
        .astype(int)
    )

    agg_df = agg_df.sort_values(
        by=["Product Name", "rank"], ascending=[True, True]
    ).reset_index(drop=True)

    # Filter recommendations for alternative factories only
    reallocations = agg_df[~agg_df["is_current"]].copy()
    reallocations["realloc_rank"] = (
        reallocations.groupby("Product Name")["composite_score"]
        .rank(ascending=False, method="dense")
        .astype(int)
    )
    top_per_product = reallocations[reallocations["realloc_rank"] <= top_n].copy()

    # Four Operational KPIs
    # 1. Lead Time Reduction percent (mean across top-1 picks)
    best_picks = reallocations[reallocations["realloc_rank"] == 1]
    kpi_lead_time_reduction_pct = float(
        best_picks["mean_lead_time_reduction_pct"].mean()
    )

    # 2. Profit Impact Stability (100 - CV of profit across recommended scenarios)
    profit_std = float(best_picks["mean_profit_impact_pct"].std())
    profit_mean = float(best_picks["mean_profit_impact_pct"].mean())
    denom = abs(profit_mean) if abs(profit_mean) > 1e-4 else 1.0
    stability_val = max(0.0, 100.0 - (profit_std / denom) * 15.0)
    kpi_profit_stability = float(np.clip(stability_val, 10.0, 99.0))

    # 3. Scenario Confidence Score (average across best picks)
    kpi_confidence_score = float(best_picks["confidence_score"].mean())

    # 4. Recommendation Coverage (share of SKUs with viable improvements)
    total_products = int(agg_df["Product Name"].nunique())
    viable_recs = reallocations[
        (reallocations["mean_lead_time_reduction_pct"] > 0.0)
        & (reallocations["mean_profit_impact_pct"] >= -profit_tolerance_pct)
    ]
    covered_products = int(viable_recs["Product Name"].nunique())
    kpi_coverage_pct = float((covered_products / total_products) * 100.0)

    kpis = {
        "lead_time_reduction_pct": kpi_lead_time_reduction_pct,
        "profit_impact_stability": kpi_profit_stability,
        "scenario_confidence_score": kpi_confidence_score,
        "recommendation_coverage_pct": kpi_coverage_pct,
        "total_catalog_products": total_products,
        "products_with_viable_improvements": covered_products,
    }

    return agg_df, top_per_product, kpis


def run_stage_6(
    scenario_path: Path = PRECOMPUTED_SCENARIO_PATH,
    speed_weight: float = 0.5,
    top_n: int = 3,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    """Execute end-to-end Stage 6 optimization and recommendation ranking.

    Args:
        scenario_path: Path to precomputed scenario CSV.
        speed_weight: Speed weight (0.0 to 1.0).
        top_n: Top picks per SKU.

    Returns:
        Tuple of (Full Product Rankings, Top Reallocations, KPIs).
    """
    scenario_df = pd.read_csv(scenario_path)
    full_ranked, top_recs, kpis = rank_factory_recommendations(
        scenario_df, speed_weight=speed_weight, top_n=top_n
    )

    return full_ranked, top_recs, kpis


if __name__ == "__main__":
    ranked, top_picks, kpi_summary = run_stage_6()
    print("STAGE 6 VERIFICATION SUMMARY:")
    print("Operational KPIs:")
    for k, v in kpi_summary.items():
        print(f"  {k}: {v:.2f}" if isinstance(v, float) else f"  {k}: {v}")
    print("\nSample Top Reallocations:")
    sample_cols = [
        "Product Name",
        "current_factory",
        "candidate_factory",
        "mean_lead_time_reduction_pct",
        "mean_profit_impact_pct",
        "confidence_score",
        "composite_score",
    ]
    print(top_picks[sample_cols].head(6).to_string(index=False))
