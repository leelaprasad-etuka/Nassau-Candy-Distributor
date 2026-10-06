"""Streamlit Web Application: Factory Reallocation and Shipping Optimization.

Nassau Candy Distributor Decision Support System.
Provides interactive simulation, what-if trade-off analysis, KPI tracking,
geospatial route mapping, sensitivity stress testing, and risk auditing.
Strict styling guidelines: Zero emojis, professional typography, cached computations.
"""

import sys
from pathlib import Path

# Add current directory and parent directory to Python path
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import (
    BEST_MODEL_PATH,
    DEFAULT_FREIGHT_RATE_PER_UNIT_KM,
    FACTORIES,
    METRICS_PATH,
    PRECOMPUTED_SCENARIO_PATH,
    PROCESSED_DATA_PATH,
    PRODUCT_FACTORY_MAP,
    PROFIT_LOSS_TOLERANCE_PCT,
)
from src.geo import STATE_PROVINCE_CENTROIDS
from src.recommend import (
    compute_scenario_confidence_score,
    rank_factory_recommendations,
)


# Page Configuration
st.set_page_config(
    page_title="Factory Reallocation and Shipping Optimization - Nassau Candy Distributor",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sophisticated Corporate Styling (Zero emojis, high contrast, clean typography)
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #0f172a;
    }
    .main {
        background-color: #f8fafc;
    }
    .hero-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #ffffff;
        padding: 24px 32px;
        border-radius: 8px;
        margin-bottom: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .hero-title {
        font-size: 1.85rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin-bottom: 6px;
        color: #f8fafc;
    }
    .hero-subtitle {
        font-size: 0.95rem;
        color: #94a3b8;
        margin-bottom: 16px;
    }
    .hero-badges {
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
    }
    .hero-badge {
        background-color: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.15);
        color: #e2e8f0;
        padding: 5px 12px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }
    .hero-badge strong {
        color: #38bdf8;
    }
    .card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    }
    .card-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 12px;
        border-bottom: 1px solid #f1f5f9;
        padding-bottom: 8px;
    }
    .stat-box {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 14px 16px;
        text-align: left;
    }
    .stat-label {
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
        margin-bottom: 4px;
    }
    .stat-value {
        font-size: 1.45rem;
        font-weight: 800;
        color: #0f172a;
        line-height: 1.2;
    }
    .stat-sub {
        font-size: 0.78rem;
        color: #059669;
        font-weight: 600;
        margin-top: 3px;
    }
    .stat-sub-neutral {
        font-size: 0.78rem;
        color: #64748b;
        font-weight: 500;
        margin-top: 3px;
    }
    .badge-pill {
        display: inline-block;
        padding: 3px 9px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-emerald {
        background-color: #d1fae5;
        color: #065f46;
    }
    .badge-blue {
        background-color: #dbeafe;
        color: #1e40af;
    }
    .badge-amber {
        background-color: #fef3c7;
        color: #92400e;
    }
    .badge-slate {
        background-color: #f1f5f9;
        color: #475569;
    }
    .callout {
        padding: 14px 18px;
        border-radius: 6px;
        margin-bottom: 16px;
        font-size: 0.9rem;
        line-height: 1.5;
    }
    .callout-info {
        background-color: #f0f9ff;
        border-left: 4px solid #0284c7;
        color: #0369a1;
    }
    .callout-warning {
        background-color: #fffbeb;
        border-left: 4px solid #d97706;
        color: #92400e;
    }
    .callout-danger {
        background-color: #fef2f2;
        border-left: 4px solid #dc2626;
        color: #991b1b;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        border-bottom: 2px solid #e2e8f0;
        margin-bottom: 16px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        font-weight: 600;
        font-size: 0.92rem;
        border-radius: 6px 6px 0 0;
        color: #475569;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff;
        color: #0f172a !important;
        border: 1px solid #e2e8f0;
        border-bottom: 2px solid #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_base_data() -> pd.DataFrame:
    """Load and cache processed historical dataset."""
    if not PROCESSED_DATA_PATH.exists():
        st.error("Processed data file not found. Please run preprocessing pipeline first.")
        st.stop()
    return pd.read_csv(PROCESSED_DATA_PATH)


@st.cache_data
def load_scenario_data() -> pd.DataFrame:
    """Load and cache precomputed simulation grid."""
    if not PRECOMPUTED_SCENARIO_PATH.exists():
        from src.simulation import run_stage_5
        sim_df, _ = run_stage_5()
        return sim_df
    return pd.read_csv(PRECOMPUTED_SCENARIO_PATH)


@st.cache_resource
def load_model_artifacts():
    """Load and cache predictive models and evaluation metadata."""
    if not METRICS_PATH.exists() or not BEST_MODEL_PATH.exists():
        from src.models import run_stage_4
        run_stage_4()
    model = joblib.load(BEST_MODEL_PATH)
    metrics_meta = joblib.load(METRICS_PATH)
    return model, metrics_meta


# Data Loading
df_historical = load_base_data()
df_scenarios = load_scenario_data()
best_model, metrics_meta = load_model_artifacts()

# Executive Hero Banner
st.markdown(
    """
    <div class="hero-header">
        <div class="hero-title">Factory Reallocation and Shipping Optimization System</div>
        <div class="hero-subtitle">
            Enterprise Decision Support Platform for Nassau Candy Distributor | Multi-Criteria Supply Chain Optimization
        </div>
        <div class="hero-badges">
            <span class="hero-badge">Catalog: <strong>15 Confectionery SKUs</strong></span>
            <span class="hero-badge">Production: <strong>5 Manufacturing Facilities</strong></span>
            <span class="hero-badge">Network: <strong>59 Destination Jurisdictions</strong></span>
            <span class="hero-badge">Permutations: <strong>17,700 Precomputed Options</strong></span>
            <span class="hero-badge">Algorithm: <strong>Random Forest + Haversine Spatial Bridge</strong></span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar Configuration Controls
st.sidebar.markdown("### Decision Parameters and Filters")

# Presets Quick Select
preset = st.sidebar.selectbox(
    "Optimization Objective Preset",
    options=[
        "Custom Configuration",
        "Balanced Strategy (50% Speed / 50% Profit)",
        "Aggressive Delivery Acceleration (80% Speed / 20% Profit)",
        "Margin Protection Priority (20% Speed / 80% Profit)",
    ],
    index=1,
)

if preset == "Balanced Strategy (50% Speed / 50% Profit)":
    default_w = 0.50
elif preset == "Aggressive Delivery Acceleration (80% Speed / 20% Profit)":
    default_w = 0.80
elif preset == "Margin Protection Priority (20% Speed / 80% Profit)":
    default_w = 0.20
else:
    default_w = 0.50

# Objective Slider
speed_weight = st.sidebar.slider(
    "Objective Weight: Speed vs Profit",
    min_value=0.0,
    max_value=1.0,
    value=default_w,
    step=0.05,
    help="1.0 prioritizes delivery lead time reduction exclusively. 0.0 prioritizes shipping freight and net profit exclusively.",
)
st.sidebar.caption(
    f"Current Trade-off: Speed Weight = {speed_weight:.2f} | Profit Weight = {(1.0 - speed_weight):.2f}"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Logistics and Geographic Filters")

# Product Selector
product_options = ["All Products"] + sorted(df_historical["Product Name"].unique().tolist())
selected_product = st.sidebar.selectbox("Filter by Product", options=product_options, index=0)

# Region Selector
region_options = ["All Regions"] + sorted(df_historical["Region"].unique().tolist())
selected_region = st.sidebar.selectbox("Filter by Destination Region", options=region_options, index=0)

# Division Selector
division_options = ["All Divisions"] + sorted(df_historical["Division"].unique().tolist())
selected_division = st.sidebar.selectbox("Filter by Product Division", options=division_options, index=0)

# Ship Mode Selector
all_ship_modes = sorted(df_historical["Ship Mode"].unique().tolist())
selected_ship_modes = st.sidebar.multiselect(
    "Filter by Shipping Mode",
    options=all_ship_modes,
    default=all_ship_modes,
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Financial Assumptions")

# Freight Rate Input
freight_rate = st.sidebar.number_input(
    "Base Freight Cost ($ / unit / km)",
    min_value=0.0001,
    max_value=0.0100,
    value=DEFAULT_FREIGHT_RATE_PER_UNIT_KM,
    step=0.0001,
    format="%.4f",
    help="Adjustable shipping transportation cost per unit per kilometer.",
)

# Profit Loss Tolerance
profit_tolerance = st.sidebar.number_input(
    "Max Acceptable Profit Reduction (%)",
    min_value=0.0,
    max_value=25.0,
    value=PROFIT_LOSS_TOLERANCE_PCT,
    step=0.5,
    help="Upper bound threshold on tolerable freight cost increase for viable speed improvements.",
)

# Recommendations Count
top_n = st.sidebar.slider("Top Recommendations per SKU", min_value=1, max_value=4, value=3)

# Filter Data Execution
filtered_scenarios = df_scenarios.copy()

if selected_product != "All Products":
    filtered_scenarios = filtered_scenarios[
        filtered_scenarios["Product Name"] == selected_product
    ]

if selected_region != "All Regions":
    filtered_scenarios = filtered_scenarios[
        filtered_scenarios["Region"] == selected_region
    ]

if selected_division != "All Divisions":
    filtered_scenarios = filtered_scenarios[
        filtered_scenarios["Division"] == selected_division
    ]

if selected_ship_modes:
    filtered_scenarios = filtered_scenarios[
        filtered_scenarios["Ship Mode"].isin(selected_ship_modes)
    ]
else:
    st.warning("Please select at least one Shipping Mode in the sidebar.")
    st.stop()

# Dynamic freight adjustment
if abs(freight_rate - DEFAULT_FREIGHT_RATE_PER_UNIT_KM) > 1e-6:
    rate_factor = freight_rate / DEFAULT_FREIGHT_RATE_PER_UNIT_KM
    filtered_scenarios["estimated_freight_change"] = (
        filtered_scenarios["estimated_freight_change"] * rate_factor
    )
    filtered_scenarios["adjusted_profit_change"] = (
        -filtered_scenarios["estimated_freight_change"]
    )
    filtered_scenarios["profit_impact_pct"] = (
        filtered_scenarios["profit_impact_pct"] * rate_factor
    )

if filtered_scenarios.empty:
    st.warning("No records matched the active filter criteria. Please broaden your selection.")
    st.stop()

# Compute Recommendations and KPIs
ranked_products, top_recs, kpis = rank_factory_recommendations(
    filtered_scenarios,
    speed_weight=speed_weight,
    top_n=top_n,
    profit_tolerance_pct=profit_tolerance,
)

# Main Application Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "1. Factory Optimization Simulator",
    "2. What-If Scenario Analysis",
    "3. Recommendation Dashboard",
    "4. Risk, Sensitivity and Audit Panel",
])


# MODULE 1: FACTORY OPTIMIZATION SIMULATOR
with tab1:
    st.markdown("### Factory Optimization Simulator")
    st.markdown(
        "Simulate performance and financial metrics across all five manufacturing facilities "
        "for any individual SKU. Evaluate predicted delivery speed, adjusted profit margins, and confidence."
    )

    col_sku_sel, col_sku_metric = st.columns([1, 2])

    with col_sku_sel:
        sim_product = st.selectbox(
            "Select Confectionery SKU:",
            options=sorted(df_historical["Product Name"].unique().tolist()),
            index=0,
            key="sim_product_select",
        )
        prod_historical = df_historical[df_historical["Product Name"] == sim_product]
        current_assigned = PRODUCT_FACTORY_MAP.get(sim_product, "Unknown")
        division_name = prod_historical["Division"].iloc[0]
        historical_volume = len(prod_historical)
        avg_units = prod_historical["Units"].mean()
        avg_sales = prod_historical["Sales"].mean()
        avg_margin = prod_historical["margin_percent"].mean()

    with col_sku_metric:
        st.markdown(
            f"""
            <div class="card" style="margin-bottom: 0;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                    <span style="font-size: 1.15rem; font-weight: 800; color: #0f172a;">{sim_product}</span>
                    <div>
                        <span class="badge-pill badge-blue">Division: {division_name}</span>
                        <span class="badge-pill badge-slate">Current Plant: {current_assigned}</span>
                    </div>
                </div>
                <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px;">
                    <div class="stat-box">
                        <div class="stat-label">Historical Orders</div>
                        <div class="stat-value">{historical_volume:,}</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-label">Avg Units/Order</div>
                        <div class="stat-value">{avg_units:.1f}</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-label">Avg Order Value</div>
                        <div class="stat-value">${avg_sales:.2f}</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-label">Gross Margin</div>
                        <div class="stat-value">{avg_margin:.1f}%</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # Compute comparative table across all candidate factories
    prod_scenarios = df_scenarios[df_scenarios["Product Name"] == sim_product]
    factory_comparison = (
        prod_scenarios.groupby("candidate_factory")
        .agg(
            mean_pred_lead_time=("predicted_candidate_lead_time", "mean"),
            lead_time_reduction_pct=("lead_time_reduction_pct", "mean"),
            mean_distance_km=("candidate_distance_km", "mean"),
            mean_freight=("candidate_freight_cost", "mean"),
            profit_impact_pct=("profit_impact_pct", "mean"),
            uncertainty_std=("prediction_uncertainty_std", "mean"),
        )
        .reset_index()
    )

    factory_comparison["is_current"] = (
        factory_comparison["candidate_factory"] == current_assigned
    )

    sample_sz = int(prod_scenarios["product_sample_size"].iloc[0])
    factory_comparison["confidence_score"] = [
        compute_scenario_confidence_score(
            sample_size=sample_sz,
            uncertainty_std=u,
            max_sample_size=2200,
        )
        for u in factory_comparison["uncertainty_std"]
    ]

    # Best alternative candidate
    alts = factory_comparison[~factory_comparison["is_current"]].sort_values(
        by="lead_time_reduction_pct", ascending=False
    )
    best_alt = alts.iloc[0] if not alts.empty else factory_comparison.iloc[0]
    current_row = factory_comparison[factory_comparison["is_current"]].iloc[0]

    # Head-to-Head Comparison Card
    col_h2h_1, col_h2h_2 = st.columns(2)
    with col_h2h_1:
        st.markdown(
            f"""
            <div class="card" style="border-left: 4px solid #64748b;">
                <div class="card-title">Current Baseline Facility: {current_assigned}</div>
                <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px;">
                    <div>
                        <div class="stat-label">Predicted Lead Time Index</div>
                        <div class="stat-value">{current_row['mean_pred_lead_time']:.1f} <span style="font-size:0.85rem; color:#64748b;">days</span></div>
                    </div>
                    <div>
                        <div class="stat-label">Average Distance</div>
                        <div class="stat-value">{current_row['mean_distance_km']:.1f} <span style="font-size:0.85rem; color:#64748b;">km</span></div>
                    </div>
                    <div>
                        <div class="stat-label">Estimated Freight Cost</div>
                        <div class="stat-value">${current_row['mean_freight']:.2f} <span style="font-size:0.85rem; color:#64748b;">/order</span></div>
                    </div>
                    <div>
                        <div class="stat-label">Confidence Score</div>
                        <div class="stat-value">{current_row['confidence_score']:.1f} <span style="font-size:0.85rem; color:#64748b;">/100</span></div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_h2h_2:
        speed_delta_color = "#059669" if best_alt['lead_time_reduction_pct'] >= 0 else "#dc2626"
        profit_delta_color = "#059669" if best_alt['profit_impact_pct'] >= 0 else "#dc2626"
        st.markdown(
            f"""
            <div class="card" style="border-left: 4px solid #0284c7;">
                <div class="card-title">Optimal Alternative Facility: {best_alt['candidate_factory']}</div>
                <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px;">
                    <div>
                        <div class="stat-label">Speed Improvement</div>
                        <div class="stat-value" style="color: {speed_delta_color};">{best_alt['lead_time_reduction_pct']:+.2f}%</div>
                        <div class="stat-sub-neutral">{best_alt['mean_pred_lead_time']:.1f} days relative index</div>
                    </div>
                    <div>
                        <div class="stat-label">Distance Change</div>
                        <div class="stat-value">{(best_alt['mean_distance_km'] - current_row['mean_distance_km']):+.1f} <span style="font-size:0.85rem; color:#64748b;">km</span></div>
                        <div class="stat-sub-neutral">{best_alt['mean_distance_km']:.1f} km avg transit</div>
                    </div>
                    <div>
                        <div class="stat-label">Profit Impact</div>
                        <div class="stat-value" style="color: {profit_delta_color};">{best_alt['profit_impact_pct']:+.2f}%</div>
                        <div class="stat-sub-neutral">${(best_alt['mean_freight'] - current_row['mean_freight']):+.2f} freight differential</div>
                    </div>
                    <div>
                        <div class="stat-label">Confidence Score</div>
                        <div class="stat-value">{best_alt['confidence_score']:.1f} <span style="font-size:0.85rem; color:#64748b;">/100</span></div>
                        <div class="stat-sub-neutral">Uncertainty std: {best_alt['uncertainty_std']:.1f} days</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Detailed Comparison Table
    st.markdown("#### Performance Comparison Across All Facilities")
    disp_table = factory_comparison[[
        "candidate_factory",
        "is_current",
        "mean_pred_lead_time",
        "lead_time_reduction_pct",
        "mean_distance_km",
        "mean_freight",
        "profit_impact_pct",
        "confidence_score",
    ]].copy()
    disp_table["Status"] = disp_table["is_current"].apply(
        lambda x: "Current Baseline" if x else "Candidate Alternative"
    )
    disp_table = disp_table.drop(columns=["is_current"])
    disp_table.columns = [
        "Factory Name",
        "Lead Time Index (Days)",
        "Speed Change (%)",
        "Average Distance (km)",
        "Est Freight ($/order)",
        "Profit Impact (%)",
        "Confidence Score",
        "Role",
    ]
    st.dataframe(
        disp_table.style.format({
            "Lead Time Index (Days)": "{:.1f}",
            "Speed Change (%)": "{:+.2f}%",
            "Average Distance (km)": "{:.1f}",
            "Est Freight ($/order)": "${:.2f}",
            "Profit Impact (%)": "{:+.2f}%",
            "Confidence Score": "{:.1f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    # Interactive Visualizations
    col_plot_1, col_plot_2 = st.columns(2)
    with col_plot_1:
        fig_bar = px.bar(
            factory_comparison,
            x="candidate_factory",
            y="lead_time_reduction_pct",
            color="is_current",
            color_discrete_map={True: "#94a3b8", False: "#0284c7"},
            labels={
                "candidate_factory": "Facility",
                "lead_time_reduction_pct": "Lead Time Reduction (%)",
                "is_current": "Baseline",
            },
            title=f"Delivery Speed Change by Facility: {sim_product}",
        )
        fig_bar.add_hline(y=0, line_dash="dash", line_color="#cbd5e1")
        fig_bar.update_layout(showlegend=False, margin=dict(l=20, r=20, t=40, b=20), height=340)
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_plot_2:
        fig_scatter = px.scatter(
            factory_comparison,
            x="lead_time_reduction_pct",
            y="profit_impact_pct",
            text="candidate_factory",
            size="confidence_score",
            color="is_current",
            color_discrete_map={True: "#64748b", False: "#059669"},
            labels={
                "lead_time_reduction_pct": "Speed Gain (%) [Higher is Better]",
                "profit_impact_pct": "Profit Impact (%) [Higher is Better]",
            },
            title="Multi-Objective Trade-Off Frontier",
        )
        fig_scatter.update_traces(textposition="top center")
        fig_scatter.add_hline(y=0, line_dash="dash", line_color="#cbd5e1")
        fig_scatter.add_vline(x=0, line_dash="dash", line_color="#cbd5e1")
        fig_scatter.update_layout(showlegend=False, margin=dict(l=20, r=20, t=40, b=20), height=340)
        st.plotly_chart(fig_scatter, use_container_width=True)


# MODULE 2: WHAT-IF SCENARIO ANALYSIS & GEOSPATIAL LOGISTICS
with tab2:
    st.markdown("### What-If Scenario Analysis: Geospatial and Corridor Intelligence")
    st.markdown(
        "Interactive geospatial modeling comparing current versus recommended manufacturing plants. "
        "Inspect distribution corridors across 59 North American jurisdictions."
    )

    col_map_ctrl, col_map_view = st.columns([1, 2.5])

    with col_map_ctrl:
        st.markdown("#### Scenario Inspection")
        target_prod = st.selectbox(
            "Select SKU for Geospatial Analysis:",
            options=sorted(df_historical["Product Name"].unique().tolist()),
            key="whatif_prod_select",
        )
        current_fac = PRODUCT_FACTORY_MAP.get(target_prod)

        prod_top = top_recs[top_recs["Product Name"] == target_prod]
        if not prod_top.empty:
            rec_fac = prod_top.iloc[0]["candidate_factory"]
            gain_lt = prod_top.iloc[0]["mean_lead_time_reduction_pct"]
            gain_prof = prod_top.iloc[0]["mean_profit_impact_pct"]
            cand_dist = prod_top.iloc[0]["mean_candidate_distance_km"]
            curr_dist = prod_top.iloc[0]["mean_current_distance_km"]
        else:
            rec_fac = current_fac
            gain_lt = 0.0
            gain_prof = 0.0
            cand_dist = 2000.0
            curr_dist = 2000.0

        st.markdown(
            f"""
            <div class="card">
                <div class="card-title">Route Summary</div>
                <div style="font-size: 0.9rem; margin-bottom: 8px;">
                    <strong>Current Hub:</strong> {current_fac}
                </div>
                <div style="font-size: 0.9rem; margin-bottom: 12px;">
                    <strong>Proposed Hub:</strong> <span style="color:#059669; font-weight:700;">{rec_fac}</span>
                </div>
                <hr style="margin: 8px 0; border: none; border-top: 1px solid #e2e8f0;">
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; font-size: 0.85rem;">
                    <div>Speed Delta: <strong>{gain_lt:+.2f}%</strong></div>
                    <div>Profit Delta: <strong>{gain_prof:+.2f}%</strong></div>
                    <div>Distance Delta: <strong>{(cand_dist - curr_dist):+.1f} km</strong></div>
                    <div>Status: <strong>{'Reallocated' if rec_fac != current_fac else 'Maintained'}</strong></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        show_current_lines = st.checkbox("Show Current Plant Corridors", value=False)
        show_proposed_lines = st.checkbox("Show Proposed Plant Corridors", value=True)
        corridor_limit = st.slider("Corridor Overlay Density (Destinations)", min_value=5, max_value=35, value=15)

    with col_map_view:
        map_fig = go.Figure()

        # Add Plant Nodes
        fac_names = list(FACTORIES.keys())
        fac_lats = [FACTORIES[f][0] for f in fac_names]
        fac_lons = [FACTORIES[f][1] for f in fac_names]
        fac_colors = [
            "#dc2626" if f == current_fac else ("#059669" if f == rec_fac else "#64748b")
            for f in fac_names
        ]
        fac_symbols = [
            "square" if f in [current_fac, rec_fac] else "circle"
            for f in fac_names
        ]

        map_fig.add_trace(
            go.Scattergeo(
                lon=fac_lons,
                lat=fac_lats,
                mode="markers+text",
                text=fac_names,
                textposition="bottom center",
                marker=dict(
                    size=14,
                    color=fac_colors,
                    symbol=fac_symbols,
                    line=dict(width=2, color="#0f172a"),
                ),
                name="Manufacturing Plants",
                hoverinfo="text",
            )
        )

        # Add Destination Centroids
        dest_states = sorted(list(STATE_PROVINCE_CENTROIDS.keys()))
        dest_lats = [STATE_PROVINCE_CENTROIDS[s][0] for s in dest_states]
        dest_lons = [STATE_PROVINCE_CENTROIDS[s][1] for s in dest_states]

        map_fig.add_trace(
            go.Scattergeo(
                lon=dest_lons,
                lat=dest_lats,
                mode="markers",
                marker=dict(size=5, color="#94a3b8", opacity=0.7),
                hovertext=dest_states,
                name="Customer State Centroids",
            )
        )

        # Connect Corridors
        sample_dest = dest_states[:corridor_limit]

        if show_current_lines and current_fac in FACTORIES:
            c_lat, c_lon = FACTORIES[current_fac]
            for s in sample_dest:
                d_lat, d_lon = STATE_PROVINCE_CENTROIDS[s]
                map_fig.add_trace(
                    go.Scattergeo(
                        lon=[c_lon, d_lon],
                        lat=[c_lat, d_lat],
                        mode="lines",
                        line=dict(width=1, color="rgba(220, 38, 38, 0.35)"),
                        showlegend=False,
                        hoverinfo="none",
                    )
                )

        if show_proposed_lines and rec_fac in FACTORIES:
            r_lat, r_lon = FACTORIES[rec_fac]
            for s in sample_dest:
                d_lat, d_lon = STATE_PROVINCE_CENTROIDS[s]
                map_fig.add_trace(
                    go.Scattergeo(
                        lon=[r_lon, d_lon],
                        lat=[r_lat, d_lat],
                        mode="lines",
                        line=dict(width=1.2, color="rgba(5, 150, 105, 0.45)"),
                        showlegend=False,
                        hoverinfo="none",
                    )
                )

        map_fig.update_layout(
            title_text=f"Logistics Network Corridors: {target_prod} (Red: Current | Green: Proposed)",
            geo=dict(
                scope="north america",
                showland=True,
                landcolor="#f8fafc",
                subunitcolor="#cbd5e1",
                countrycolor="#94a3b8",
                showsubunits=True,
                showcountries=True,
                center=dict(lat=41.0, lon=-97.0),
            ),
            margin=dict(l=0, r=0, t=35, b=0),
            height=480,
        )
        st.plotly_chart(map_fig, use_container_width=True)

    # Destination State Mileage Savings Bar Chart
    st.markdown("#### Transit Distance Comparison Across Sample Destinations")
    if current_fac in FACTORIES and rec_fac in FACTORIES:
        sample_dist_data = []
        for s in dest_states[:12]:
            d_lat, d_lon = STATE_PROVINCE_CENTROIDS[s]
            from src.geo import haversine_distance
            c_dist = float(haversine_distance(FACTORIES[current_fac][0], FACTORIES[current_fac][1], d_lat, d_lon))
            r_dist = float(haversine_distance(FACTORIES[rec_fac][0], FACTORIES[rec_fac][1], d_lat, d_lon))
            sample_dist_data.append({
                "Destination": s,
                "Current Distance (km)": c_dist,
                "Proposed Distance (km)": r_dist,
                "Distance Reduction (km)": c_dist - r_dist,
            })
        df_dist_sample = pd.DataFrame(sample_dist_data)
        fig_dist_bar = px.bar(
            df_dist_sample,
            x="Destination",
            y=["Current Distance (km)", "Proposed Distance (km)"],
            barmode="group",
            color_discrete_sequence=["#ef4444", "#10b981"],
            title=f"Transit Distance Comparison per Destination State: {target_prod}",
        )
        fig_dist_bar.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=300)
        st.plotly_chart(fig_dist_bar, use_container_width=True)


# MODULE 3: RECOMMENDATION DASHBOARD & SUPPLY CHAIN REBALANCING
with tab3:
    st.markdown("### Strategic Recommendation Dashboard")
    st.markdown(
        "Prioritized supply chain rebalancing proposals. Review portfolio-level KPI metric cards, "
        "inspect multi-criteria scores, and download actionable recommendations."
    )

    # 4 Key Operational Metric Cards
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

    with kpi_col1:
        st.markdown(
            f"""
            <div class="stat-box" style="border-top: 4px solid #0284c7;">
                <div class="stat-label">Avg Lead Time Reduction</div>
                <div class="stat-value">{kpis['lead_time_reduction_pct']:+.2f}%</div>
                <div class="stat-sub">Delivery Efficiency Acceleration</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with kpi_col2:
        st.markdown(
            f"""
            <div class="stat-box" style="border-top: 4px solid #059669;">
                <div class="stat-label">Profit Impact Stability</div>
                <div class="stat-value">{kpis['profit_impact_stability']:.1f}%</div>
                <div class="stat-sub">Cross-Territory Resilience</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with kpi_col3:
        st.markdown(
            f"""
            <div class="stat-box" style="border-top: 4px solid #d97706;">
                <div class="stat-label">Confidence Score</div>
                <div class="stat-value">{kpis['scenario_confidence_score']:.1f} <span style="font-size:0.85rem; color:#64748b;">/ 100</span></div>
                <div class="stat-sub-neutral">Sample-weighted statistical rigor</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with kpi_col4:
        st.markdown(
            f"""
            <div class="stat-box" style="border-top: 4px solid #7c3aed;">
                <div class="stat-label">Recommendation Coverage</div>
                <div class="stat-value">{kpis['recommendation_coverage_pct']:.1f}%</div>
                <div class="stat-sub">{kpis['products_with_viable_improvements']} of {kpis['total_catalog_products']} SKUs optimized</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # Portfolio Reallocation Proposals Table
    st.markdown("#### Ranked Supply Chain Reallocation Proposals")

    # Filter recommendations for top-1 best pick per SKU
    best_recs = top_recs[top_recs["realloc_rank"] == 1].copy()
    best_recs["Action"] = best_recs.apply(
        lambda r: "Reallocate Plant" if (r["mean_lead_time_reduction_pct"] > 0 and r["mean_profit_impact_pct"] >= -profit_tolerance) else "Maintain Status Quo",
        axis=1,
    )

    table_data = best_recs[[
        "Product Name",
        "Division",
        "current_factory",
        "candidate_factory",
        "Action",
        "mean_lead_time_reduction_pct",
        "mean_profit_impact_pct",
        "mean_candidate_distance_km",
        "confidence_score",
        "composite_score",
    ]].copy()

    table_data.columns = [
        "Product Name",
        "Division",
        "Current Plant",
        "Recommended Plant",
        "Recommendation",
        "Lead Time Gain (%)",
        "Profit Impact (%)",
        "Avg Distance (km)",
        "Confidence",
        "Score",
    ]

    st.dataframe(
        table_data.style.format({
            "Lead Time Gain (%)": "{:+.2f}%",
            "Profit Impact (%)": "{:+.2f}%",
            "Avg Distance (km)": "{:.1f}",
            "Confidence": "{:.1f}",
            "Score": "{:.3f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    # Action Summary
    realloc_count = (table_data["Recommendation"] == "Reallocate Plant").sum()
    maintain_count = (table_data["Recommendation"] == "Maintain Status Quo").sum()
    st.caption(
        f"Portfolio Summary: {realloc_count} products recommended for physical facility shifts, "
        f"{maintain_count} products recommended to remain at current manufacturing facilities."
    )

    # Download Buttons
    col_dl1, col_dl2 = st.columns([1, 3])
    with col_dl1:
        csv_bytes = table_data.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Recommendations CSV",
            data=csv_bytes,
            file_name="nassau_candy_recommendations.csv",
            mime="text/csv",
        )


# MODULE 4: RISK, SENSITIVITY, AND AUDIT PANEL
with tab4:
    st.markdown("### Risk, Sensitivity, and Governance Audit Panel")
    st.markdown(
        "Supply chain sensitivity stress-testing, data volume auditing, "
        "predictive model validation, and governance disclosures."
    )

    # Stress-Testing Section
    st.markdown("#### 1. Fuel and Freight Inflation Sensitivity Stress-Test")
    st.write(
        "Evaluate how potential increases in freight contract rates (e.g., fuel surcharges, diesel inflation) "
        "alter the financial trade-off for producing closer to customer demand centers."
    )

    stress_factor = st.slider(
        "Freight Transportation Cost Escalation (%)",
        min_value=0,
        max_value=200,
        value=0,
        step=25,
        help="Simulates an industry-wide rise in carrier freight rates.",
    )

    effective_rate = DEFAULT_FREIGHT_RATE_PER_UNIT_KM * (1.0 + stress_factor / 100.0)
    st.info(
        f"Simulated Freight Rate: ${effective_rate:.5f} / unit / km (Base: ${DEFAULT_FREIGHT_RATE_PER_UNIT_KM:.4f}). "
        f"Higher freight rates substantially increase the financial return on manufacturing SKUs closer to regional demand."
    )

    # Risk Alerts Section
    st.markdown("#### 2. Operational Risk Alerts")

    low_conf_items = top_recs[top_recs["confidence_score"] < 25.0]["Product Name"].unique().tolist()
    if low_conf_items:
        items_str = ", ".join(low_conf_items)
        st.markdown(
            f"""
            <div class="callout callout-warning">
                <strong>Low Sample Volume Warning:</strong> The following SKUs possess limited historical transaction volume or elevated model dispersion:
                <br><em>{items_str}</em>. Implement small-scale pilot reallocations prior to nationwide line cutovers.
            </div>
            """,
            unsafe_allow_html=True,
        )

    excess_cost_items = top_recs[top_recs["mean_profit_impact_pct"] < -profit_tolerance]["Product Name"].unique().tolist()
    if excess_cost_items:
        cost_str = ", ".join(excess_cost_items)
        st.markdown(
            f"""
            <div class="callout callout-danger">
                <strong>Financial Expense Alert:</strong> Reallocating the following SKUs increases shipping freight beyond the tolerable {profit_tolerance:.1f}% threshold:
                <br><em>{cost_str}</em>. Reallocation should only proceed if rapid customer SLA delivery demands override transportation cost.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Model Evaluation Metrics
    st.markdown("#### 3. Predictive Model Architecture and Cross-Validation")
    metrics_table = metrics_meta["metrics_df"]
    st.dataframe(
        metrics_table.style.format({
            "CV_RMSE": "{:.2f}",
            "CV_R2": "{:.4f}",
            "Test_RMSE": "{:.2f}",
            "Test_MAE": "{:.2f}",
            "Test_R2": "{:.4f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    # Feature Importance Plot
    feat_imp = metrics_meta["feature_importances"].head(6)
    fig_imp = px.bar(
        feat_imp,
        x="Importance",
        y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale="Blues",
        title="Top Predictive Features Powering the Simulation (Random Forest)",
    )
    fig_imp.update_layout(yaxis=dict(autorange="reversed"), margin=dict(l=20, r=20, t=40, b=20), height=260)
    st.plotly_chart(fig_imp, use_container_width=True)

    # Mandatory Formal Data Limitations Notice
    st.markdown("#### 4. Mandatory Data Limitations Notice")
    st.markdown(
        """
        <div class="callout callout-info">
            <strong>Formal Governance Disclosures:</strong>
            <ol style="margin-top: 6px; padding-left: 20px;">
                <li><strong>Relative Lead Time Indexing:</strong> Derived lead time spans from 904 to 1,642 days (mean 1,320.8 days) because historical order records date from 2024-2025 while recorded ship dates fall between 2026 and 2030. All values are utilized strictly as relative benchmarks; all efficiency gains are reported in percentage terms.</li>
                <li><strong>Spatial Bridge for Historical Collinearity:</strong> Every historical SKU was manufactured exclusively at a single factory. Great-circle haversine distance in kilometers (distance_km) is engineered as the unbiased bridge feature to evaluate counterfactual facilities.</li>
                <li><strong>Carrier Mode Statistical Invariance:</strong> Historical mean lead times across shipping classes vary by less than 1.7 days (0.13%). Consequently, carrier mode selection provides minimal predictive differentiation.</li>
                <li><strong>Synthetic Freight Formulation:</strong> Gross profit in the dataset reflects Sales minus Cost without explicit carrier charges. Transportation economics are dynamically computed via the configurable unit-km rate parameter.</li>
            </ol>
        </div>
        """,
        unsafe_allow_html=True,
    )
