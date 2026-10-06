"""Geographic centroids and spherical haversine distance utilities.

Provides complete coordinate lookups for all 50 US states, District of Columbia,
and Canadian provinces/territories. Computes vectorised great-circle distances
between manufacturing plants and customer delivery centroids.
"""

from typing import Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd

# Centroids for US States and District of Columbia
US_STATE_CENTROIDS: Dict[str, Tuple[float, float]] = {
    "Alabama": (32.806671, -86.791130),
    "Alaska": (61.370716, -152.404419),
    "Arizona": (33.729759, -111.431221),
    "Arkansas": (34.969704, -92.373123),
    "California": (36.116203, -119.681564),
    "Colorado": (39.059811, -105.311104),
    "Connecticut": (41.597782, -72.755371),
    "Delaware": (39.318523, -75.507141),
    "District of Columbia": (38.897438, -77.026817),
    "Florida": (27.766279, -81.686783),
    "Georgia": (33.040619, -83.643074),
    "Hawaii": (21.094318, -157.498337),
    "Idaho": (44.240459, -114.478828),
    "Illinois": (40.349457, -88.986137),
    "Indiana": (39.849426, -86.258278),
    "Iowa": (42.011539, -93.210526),
    "Kansas": (38.526600, -96.726486),
    "Kentucky": (37.668140, -84.670067),
    "Louisiana": (31.169546, -91.867805),
    "Maine": (44.693947, -69.381927),
    "Maryland": (39.063946, -76.802101),
    "Massachusetts": (42.230171, -71.530106),
    "Michigan": (43.326618, -84.536095),
    "Minnesota": (45.694454, -93.900192),
    "Mississippi": (32.741646, -89.678696),
    "Missouri": (38.456085, -92.288368),
    "Montana": (46.921925, -110.454353),
    "Nebraska": (41.125370, -98.268082),
    "Nevada": (38.313515, -117.055374),
    "New Hampshire": (43.452492, -71.563896),
    "New Jersey": (40.298904, -74.521011),
    "New Mexico": (34.840515, -106.248482),
    "New York": (42.165726, -74.948051),
    "North Carolina": (35.630066, -79.806419),
    "North Dakota": (47.528912, -99.784012),
    "Ohio": (40.388783, -82.764915),
    "Oklahoma": (35.565342, -96.928917),
    "Oregon": (44.572021, -122.070938),
    "Pennsylvania": (40.590752, -77.209755),
    "Rhode Island": (41.680893, -71.511780),
    "South Carolina": (33.856892, -80.945007),
    "South Dakota": (44.299782, -99.438828),
    "Tennessee": (35.747845, -86.692345),
    "Texas": (31.054487, -97.563461),
    "Utah": (40.150032, -111.862434),
    "Vermont": (44.045876, -72.710686),
    "Virginia": (37.769337, -78.169968),
    "Washington": (47.400902, -121.490494),
    "West Virginia": (38.491226, -80.954453),
    "Wisconsin": (44.268543, -89.616508),
    "Wyoming": (42.755966, -107.302490),
}

# Centroids for Canadian Provinces and Territories
CANADA_PROVINCE_CENTROIDS: Dict[str, Tuple[float, float]] = {
    "Alberta": (53.933271, -116.576504),
    "British Columbia": (53.726668, -127.647621),
    "Manitoba": (53.760861, -98.813876),
    "New Brunswick": (46.565316, -66.461916),
    "Newfoundland and Labrador": (53.135509, -57.660436),
    "Nova Scotia": (44.681987, -63.744311),
    "Ontario": (51.253775, -85.323214),
    "Prince Edward Island": (46.510712, -63.416814),
    "Quebec": (52.939916, -73.549136),
    "Saskatchewan": (52.939916, -106.450864),
    "Northwest Territories": (64.825546, -124.845733),
    "Nunavut": (70.299771, -83.107577),
    "Yukon": (64.282327, -135.000000),
}

# Master Combined Centroid Lookup
STATE_PROVINCE_CENTROIDS: Dict[str, Tuple[float, float]] = {
    **US_STATE_CENTROIDS,
    **CANADA_PROVINCE_CENTROIDS,
}


def get_centroid(state_or_province: str) -> Optional[Tuple[float, float]]:
    """Retrieve latitude and longitude for a given state or province name.

    Args:
        state_or_province: State or province name string.

    Returns:
        Tuple of (latitude, longitude) or None if not found.
    """
    clean_name = state_or_province.strip() if isinstance(state_or_province, str) else ""
    return STATE_PROVINCE_CENTROIDS.get(clean_name)


def haversine_distance(
    lat1: Union[float, np.ndarray, pd.Series],
    lon1: Union[float, np.ndarray, pd.Series],
    lat2: Union[float, np.ndarray, pd.Series],
    lon2: Union[float, np.ndarray, pd.Series],
) -> Union[float, np.ndarray, pd.Series]:
    """Calculate great-circle distance between two points in kilometers.

    Implements the vectorised spherical haversine formula (Earth radius = 6371.0088 km).

    Args:
        lat1: Latitude of origin point(s) in degrees.
        lon1: Longitude of origin point(s) in degrees.
        lat2: Latitude of destination point(s) in degrees.
        lon2: Longitude of destination point(s) in degrees.

    Returns:
        Distance in kilometers.
    """
    radius_earth_km = 6371.0088

    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    a = (
        np.sin(delta_phi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2
    )
    a = np.clip(a, 0.0, 1.0)
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))

    return radius_earth_km * c
