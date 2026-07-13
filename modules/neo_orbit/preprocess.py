"""
NEO Orbit — Data Preprocessing Module
"""

import os
import numpy as np
import pandas as pd


def load_and_preprocess(data_dir):
    """Load NEO orbital and close approach datasets."""
    processed_dir = os.path.join(data_dir, "datasets", "processed")
    raw_dir = os.path.join(data_dir, "data", "raw")

    # Try processed first
    orb_path = os.path.join(processed_dir, "neo_orbital_processed.csv")
    close_path = os.path.join(processed_dir, "neo_close_processed.csv")

    if os.path.exists(orb_path) and os.path.exists(close_path):
        df_orb = pd.read_csv(orb_path)
        df_close = pd.read_csv(close_path)
    else:
        # Load from raw
        orb_files = [f for f in os.listdir(raw_dir) if "orbital" in f.lower() and f.endswith(".csv")]
        close_files = [f for f in os.listdir(raw_dir) if ("close" in f.lower() or "approach" in f.lower()) and f.endswith(".csv")]

        df_orb = pd.read_csv(os.path.join(raw_dir, orb_files[0])) if orb_files else pd.DataFrame()
        df_close = pd.read_csv(os.path.join(raw_dir, close_files[0])) if close_files else pd.DataFrame()

    return df_orb, df_close


def get_orbital_elements(df_orb):
    """Extract orbital elements for propagation."""
    params = {
        "a": "semi-major axis (AU)",
        "e": "eccentricity",
        "i": "inclination (deg)",
        "om": "longitude of ascending node (deg)",
        "w": "argument of perihelion (deg)",
        "ma": "mean anomaly (deg)"
    }
    available = {k: v for k, v in params.items() if k in df_orb.columns}
    return df_orb[list(available.keys())].values, list(available.keys())
