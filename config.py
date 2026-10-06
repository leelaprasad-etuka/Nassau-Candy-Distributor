"""Configuration parameters and settings for the Nassau Candy Distributor project.

Consolidates all file paths, geospatial coordinates, SKU-factory assignments,
and simulation parameters to eliminate hard-coded magic values across modules.
"""

from pathlib import Path
from typing import Dict, List, Tuple

# Base Project Paths
BASE_DIR: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = BASE_DIR / "data"
MODELS_DIR: Path = BASE_DIR / "models"
REPORTS_DIR: Path = BASE_DIR / "reports"
NOTEBOOKS_DIR: Path = BASE_DIR / "notebooks"

# Data File Paths
RAW_DATA_PATH: Path = DATA_DIR / "Nassau_Candy_Distributor.csv"
CLEANED_DATA_PATH: Path = DATA_DIR / "cleaned_nassau_candy.csv"
PROCESSED_DATA_PATH: Path = DATA_DIR / "processed_nassau_candy.csv"

# Model and Scenario Artifact Paths
BEST_MODEL_PATH: Path = MODELS_DIR / "best_lead_time_model.joblib"
SCALER_PATH: Path = MODELS_DIR / "feature_scaler.joblib"
ENCODERS_PATH: Path = MODELS_DIR / "feature_encoders.joblib"
METRICS_PATH: Path = MODELS_DIR / "model_evaluation_metrics.joblib"
CLUSTERING_MODEL_PATH: Path = MODELS_DIR / "route_clustering_model.joblib"
PRECOMPUTED_SCENARIO_PATH: Path = MODELS_DIR / "precomputed_scenarios.csv"

# Reproducibility Parameters
RANDOM_STATE: int = 42
TEST_SIZE: float = 0.20
CV_FOLDS: int = 5
IQR_MULTIPLIER: float = 1.5

# Financial and Logistics Defaults
DEFAULT_FREIGHT_RATE_PER_UNIT_KM: float = 0.0005
PROFIT_LOSS_TOLERANCE_PCT: float = 5.0

# Nassau Candy Manufacturing Plant Coordinates (Latitude, Longitude)
FACTORIES: Dict[str, Tuple[float, float]] = {
    "Lot's O' Nuts": (32.881893, -111.768036),
    "Wicked Choccy's": (32.076176, -81.088371),
    "Sugar Shack": (48.119140, -96.181150),
    "Secret Factory": (41.446333, -90.565487),
    "The Other Factory": (35.117500, -89.971107),
}

# Canonical Product to Historical Manufacturing Plant Assignment
PRODUCT_FACTORY_MAP: Dict[str, str] = {
    "Wonka Bar - Nutty Crunch Surprise": "Lot's O' Nuts",
    "Wonka Bar - Fudge Mallows": "Lot's O' Nuts",
    "Wonka Bar - Scrumdiddlyumptious": "Lot's O' Nuts",
    "Wonka Bar - Milk Chocolate": "Wicked Choccy's",
    "Wonka Bar - Triple Dazzle Caramel": "Wicked Choccy's",
    "Laffy Taffy": "Sugar Shack",
    "SweeTARTS": "Sugar Shack",
    "Nerds": "Sugar Shack",
    "Fun Dip": "Sugar Shack",
    "Fizzy Lifting Drinks": "Sugar Shack",
    "Everlasting Gobstopper": "Secret Factory",
    "Lickable Wallpaper": "Secret Factory",
    "Wonka Gum": "Secret Factory",
    "Hair Toffee": "The Other Factory",
    "Kazookles": "The Other Factory",
}

# Product Name Normalization Mapping
PRODUCT_NAME_CLEAN_MAP: Dict[str, str] = {
    "Wonka Bar -Scrumdiddlyumptious": "Wonka Bar - Scrumdiddlyumptious",
}

# Volume Division Classifications
LOW_VOLUME_DIVISIONS: List[str] = ["Sugar", "Other"]
SUGAR_VOLUME_THRESHOLD: int = 50
OTHER_VOLUME_THRESHOLD: int = 350
