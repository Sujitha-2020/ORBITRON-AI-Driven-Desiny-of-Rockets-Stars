"""
Satellite Health — Data Preprocessing Module
Handles CMAPSS / telemetry data preprocessing.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_and_preprocess(data_dir):
    """Load satellite telemetry data and return train/test splits."""
    processed_dir = os.path.join(data_dir, "datasets", "processed")
    train_path = os.path.join(processed_dir, "telemetry_train.csv")
    test_path = os.path.join(processed_dir, "telemetry_test.csv")

    if os.path.exists(train_path) and os.path.exists(test_path):
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
    else:
        # Fallback to raw
        raw_files = [f for f in os.listdir(os.path.join(data_dir, "data", "raw"))
                     if ("telemetry" in f.lower() or "cmapss" in f.lower()) and f.endswith(".csv")]
        if not raw_files:
            raise FileNotFoundError("No satellite telemetry dataset found")
        df = pd.read_csv(os.path.join(data_dir, "data", "raw", raw_files[0]))

        # Compute RUL if not present
        max_cycles = df.groupby("engine_id")["cycle"].max().reset_index()
        max_cycles.columns = ["engine_id", "max_cycle"]
        df = df.merge(max_cycles, on="engine_id")
        if "RUL" not in df.columns:
            df["RUL"] = df["max_cycle"] - df["cycle"]
        df = df.drop(columns=["max_cycle"])

        # Health index
        df["health_index"] = df.groupby("engine_id")["cycle"].transform(
            lambda x: 1.0 - (x - x.min()) / max(x.max() - x.min(), 1)
        )

        engine_ids = df["engine_id"].unique()
        np.random.seed(42)
        np.random.shuffle(engine_ids)
        split = int(len(engine_ids) * 0.8)
        train_df = df[df["engine_id"].isin(engine_ids[:split])].copy()
        test_df = df[df["engine_id"].isin(engine_ids[split:])].copy()

    sensor_cols = [c for c in train_df.columns if c.startswith("s") and c[1:].isdigit()]
    return train_df, test_df, sensor_cols


def get_sensor_data(df, sensor_cols):
    """Extract normalized sensor data matrix."""
    scaler = StandardScaler()
    X = scaler.fit_transform(df[sensor_cols].values)
    return X, scaler
