"""
Launch Failure — Data Preprocessing Module
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split


def load_and_preprocess(data_dir):
    """Load and preprocess launch dataset."""
    processed_dir = os.path.join(data_dir, "datasets", "processed")

    # Try loading pre-processed data first
    train_path = os.path.join(processed_dir, "launch_train.npz")
    test_path = os.path.join(processed_dir, "launch_test.npz")

    if os.path.exists(train_path) and os.path.exists(test_path):
        train = np.load(train_path)
        test = np.load(test_path)
        return train["X"], test["X"], train["y"], test["y"]

    # Fallback: preprocess from raw
    raw_files = [f for f in os.listdir(os.path.join(data_dir, "data", "raw"))
                 if "launch" in f.lower() and f.endswith(".csv")]
    if not raw_files:
        raise FileNotFoundError("No launch dataset found")

    df = pd.read_csv(os.path.join(data_dir, "data", "raw", raw_files[0]))

    drop_cols = ["Detail", "Datum", "Status Rocket", "Status Mission"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    if "failure_label" not in df.columns:
        df["failure_label"] = 1 - df.get("mission_success", 0)

    for col in df.select_dtypes(include="object").columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    df = df.fillna(df.median(numeric_only=True))
    y = df["failure_label"].values
    X = df.drop(columns=["failure_label", "mission_success"], errors="ignore")

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.25, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test


def get_feature_names(data_dir):
    """Get feature names from raw launch dataset."""
    raw_files = [f for f in os.listdir(os.path.join(data_dir, "data", "raw"))
                 if "launch" in f.lower() and f.endswith(".csv")]
    if not raw_files:
        return []
    df = pd.read_csv(os.path.join(data_dir, "data", "raw", raw_files[0]))
    drop_cols = ["Detail", "Datum", "Status Rocket", "Status Mission",
                 "failure_label", "mission_success"]
    return [c for c in df.columns if c not in drop_cols]
