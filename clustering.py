"""Stage 3: Exploratory data analysis distributions and route KMeans clustering.

Implements Stage 3 of the Nassau Candy Distributor optimization pipeline.
Aggregates shipments to corridor level, selects optimal k using silhouette scoring,
profiles clusters in plain language, and identifies consistently slow routes.
"""

from pathlib import Path
from typing import Dict, List, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.config import (
    CLUSTERING_MODEL_PATH,
    PROCESSED_DATA_PATH,
    RANDOM_STATE,
)


def compute_eda_distributions(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Compute lead time distributions and financial metrics across key dimensions.

    Args:
        df: Processed DataFrame.

    Returns:
        Dictionary of grouped distribution summary DataFrames.
    """
    distributions = {}

    for dim in ["Region", "Ship Mode", "Division", "assigned_factory"]:
        summary = (
            df.groupby(dim)
            .agg(
                order_count=("lead_time_days", "count"),
                mean_lead_time=("lead_time_days", "mean"),
                median_lead_time=("lead_time_days", "median"),
                std_lead_time=("lead_time_days", "std"),
                min_lead_time=("lead_time_days", "min"),
                max_lead_time=("lead_time_days", "max"),
                total_sales=("Sales", "sum"),
                total_gross_profit=("Gross Profit", "sum"),
                mean_margin_pct=("margin_percent", "mean"),
            )
            .reset_index()
        )
        distributions[dim] = summary

    # Correlation of distance with lead time
    valid_corr = df[["distance_km", "lead_time_days"]].dropna()
    correlation_val = float(valid_corr["distance_km"].corr(valid_corr["lead_time_days"]))
    distributions["distance_lead_time_correlation"] = pd.DataFrame(
        [{"metric": "Pearson Correlation (Distance vs Lead Time)", "value": correlation_val}]
    )

    return distributions


def prepare_route_level_data(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate individual shipments to route level (factory to destination state).

    Args:
        df: Processed DataFrame with route_key.

    Returns:
        Route-level aggregated DataFrame.
    """
    route_df = (
        df.groupby(["route_key", "assigned_factory", "State/Province", "Region"])
        .agg(
            mean_lead_time=("lead_time_days", "mean"),
            std_lead_time=("lead_time_days", lambda x: x.std(ddof=0)),
            order_volume=("lead_time_days", "count"),
            mean_margin=("margin_percent", "mean"),
            mean_distance_km=("distance_km", "mean"),
            total_sales=("Sales", "sum"),
            total_gross_profit=("Gross Profit", "sum"),
        )
        .reset_index()
    )
    route_df["std_lead_time"] = route_df["std_lead_time"].fillna(0.0)

    return route_df


def cluster_routes(
    route_df: pd.DataFrame,
    features: List[str] = [
        "mean_lead_time",
        "std_lead_time",
        "order_volume",
        "mean_margin",
    ],
    k_range: range = range(2, 7),
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Cluster routes using KMeans with silhouette score optimization.

    Args:
        route_df: Aggregated route DataFrame.
        features: Feature list for clustering.
        k_range: Candidate cluster counts to evaluate.
        random_state: Seed for reproducibility.

    Returns:
        Tuple of (Clustered DataFrame, metadata dict).
    """
    X_routes = route_df[features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_routes)

    silhouette_scores: Dict[int, float] = {}
    best_k = 3
    best_score = -1.0
    models = {}

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        silhouette_scores[k] = float(score)
        models[k] = kmeans
        if score > best_score:
            best_score = score
            best_k = k

    best_model = models[best_k]
    cluster_labels = best_model.predict(X_scaled)

    route_clustered = route_df.copy()
    route_clustered["cluster"] = cluster_labels

    # Plain language labeling based on centroids
    cluster_profiles = []
    cluster_name_map = {}

    for c in range(best_k):
        subset = route_clustered[route_clustered["cluster"] == c]
        avg_lt = subset["mean_lead_time"].mean()
        avg_vol = subset["order_volume"].mean()
        avg_var = subset["std_lead_time"].mean()
        avg_margin = subset["mean_margin"].mean()

        if avg_lt > route_clustered["mean_lead_time"].mean() and avg_vol < route_clustered["order_volume"].mean():
            label = "Consistently Slow Long-Haul Routes"
        elif avg_vol > route_clustered["order_volume"].mean():
            label = "High-Volume Core Highway Corridors"
        elif avg_var > route_clustered["std_lead_time"].mean():
            label = "Volatile Shipping Corridors"
        elif avg_margin > route_clustered["mean_margin"].mean():
            label = "High-Margin Regional Feeder Routes"
        else:
            label = f"Balanced Standard Distribution Cluster {c}"

        cluster_name_map[c] = label
        cluster_profiles.append({
            "cluster_id": c,
            "cluster_label": label,
            "route_count": len(subset),
            "avg_lead_time_days": float(avg_lt),
            "avg_std_lead_time": float(avg_var),
            "avg_order_volume": float(avg_vol),
            "avg_margin_percent": float(avg_margin),
            "total_cluster_orders": int(subset["order_volume"].sum()),
        })

    route_clustered["cluster_label"] = route_clustered["cluster"].map(cluster_name_map)
    slowest_routes = route_clustered.sort_values(by="mean_lead_time", ascending=False).head(10)

    metadata = {
        "best_k": best_k,
        "best_silhouette": float(best_score),
        "silhouette_scores": silhouette_scores,
        "cluster_profiles": pd.DataFrame(cluster_profiles),
        "scaler": scaler,
        "kmeans_model": best_model,
        "features": features,
        "slowest_routes": slowest_routes[
            ["route_key", "assigned_factory", "State/Province", "mean_lead_time", "order_volume", "cluster_label"]
        ],
    }

    return route_clustered, metadata


def run_stage_3(
    processed_path: Path = PROCESSED_DATA_PATH, save_model: bool = True
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], Dict[str, object]]:
    """Execute end-to-end Stage 3 EDA and Route Clustering.

    Args:
        processed_path: Path to processed CSV.
        save_model: Whether to persist clustering model.

    Returns:
        Tuple of (Route Clustered DataFrame, EDA distributions, metadata).
    """
    df = pd.read_csv(processed_path)
    eda_dists = compute_eda_distributions(df)
    route_df = prepare_route_level_data(df)
    route_clustered, cluster_meta = cluster_routes(route_df)

    if save_model:
        CLUSTERING_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "kmeans_model": cluster_meta["kmeans_model"],
                "scaler": cluster_meta["scaler"],
                "features": cluster_meta["features"],
                "best_k": cluster_meta["best_k"],
                "cluster_profiles": cluster_meta["cluster_profiles"],
            },
            CLUSTERING_MODEL_PATH,
        )

    return route_clustered, eda_dists, cluster_meta


if __name__ == "__main__":
    clustered, dists, meta = run_stage_3()
    print("STAGE 3 VERIFICATION SUMMARY:")
    print(f"Total routes clustered: {len(clustered)}")
    print(f"Optimal k: {meta['best_k']} (Silhouette Score: {meta['best_silhouette']:.4f})")
    print(f"Candidate silhouette scores: {meta['silhouette_scores']}")
    print("\nCluster Profiles:")
    print(meta["cluster_profiles"][["cluster_id", "cluster_label", "route_count", "avg_lead_time_days", "avg_order_volume"]])
    print("\nTop 5 Consistently Slowest Routes:")
    print(meta["slowest_routes"].head(5))
