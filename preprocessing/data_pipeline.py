"""
ORBITRON — Data Preprocessing Pipeline
Automatically detects and preprocesses all datasets for the three ML modules:
  1. Launch Failure Prediction
  2. Satellite Health Monitoring (CMAPSS)
  3. NEO Orbit Risk Analysis
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIRS = [
    os.path.join(BASE_DIR, "datasets", "raw"),
    os.path.join(BASE_DIR, "data", "raw")
]
PROCESSED_DIR = os.path.join(BASE_DIR, "datasets", "processed")


def _ensure_dirs():
    os.makedirs(PROCESSED_DIR, exist_ok=True)


# ---------- Dataset Detection ---------- #

def detect_datasets():
    """Scan raw directories and categorize datasets automatically."""
    mapping = {"launch": None, "satellite": None, "neo_orbital": None, "neo_close": None}
    
    for rdir in RAW_DIRS:
        if not os.path.exists(rdir):
            continue
            
        files = [f for f in os.listdir(rdir) if f.endswith(".csv")]
        for f in files:
            fl = f.lower()
            fpath = os.path.join(rdir, f)
            
            if ("launch" in fl or "space_missions" in fl) and mapping["launch"] is None:
                mapping["launch"] = fpath
            elif ("telemetry" in fl or "cmapss" in fl or "satellite" in fl) and mapping["satellite"] is None:
                mapping["satellite"] = fpath
            elif "orbital" in fl and mapping["neo_orbital"] is None:
                mapping["neo_orbital"] = fpath
            elif ("close" in fl or "approach" in fl or "neo" in fl) and mapping["neo_close"] is None:
                mapping["neo_close"] = fpath
                
    print("[DataPipeline] Detected datasets:")
    for k, v in mapping.items():
        print(f"  {k}: {os.path.basename(v) if v else 'NOT FOUND'}")
    return mapping


# ---------- Launch Data Preprocessing ---------- #

def preprocess_launch(path):
    """Handle missing values, feature engineering (year, month, price, rocket_active), SMOTE."""
    print("[DataPipeline] Preprocessing Launch dataset from space_missions_large...")
    df = pd.read_csv(path)

    # Clean column names (remove leading spaces)
    df.columns = [c.strip() for c in df.columns]

    # Target variable: failure_label = 1 if "Failure" else 0
    # Drop rows with unknown status
    df = df[df["Status Mission"].str.contains("Success|Failure", na=False)]
    df["failure_label"] = df["Status Mission"].apply(lambda x: 1 if "Failure" in str(x) else 0)

    # Feature engineering: launch_year, launch_month
    # Datum example: "Fri Aug 07, 2020 05:12 UTC"
    try:
        df["Datum_parsed"] = pd.to_datetime(df["Datum"], utc=True, errors="coerce")
        df["launch_year"] = df["Datum_parsed"].dt.year
        df["launch_month"] = df["Datum_parsed"].dt.month
    except Exception:
        # Fallback regex extraction if parsing fails
        df["launch_year"] = df["Datum"].str.extract(r'(\d{4})').astype(float)
        # Simplified month extraction mapping
        month_map = {"Jan":1, "Feb":2, "Mar":3, "Apr":4, "May":5, "Jun":6, 
                     "Jul":7, "Aug":8, "Sep":9, "Oct":10, "Nov":11, "Dec":12}
        df["launch_month"] = df["Datum"].str.extract(r'([A-Za-z]{3})')[0].map(month_map)

    df["launch_year"] = df["launch_year"].fillna(df["launch_year"].median()).fillna(2000)
    df["launch_month"] = df["launch_month"].fillna(1)

    # Convert Price to numeric cost_usd
    if "Price" in df.columns:
        df["cost_usd"] = pd.to_numeric(df["Price"].astype(str).str.replace(',', ''), errors='coerce')
    else:
        df["cost_usd"] = np.nan
    df["cost_usd"] = df["cost_usd"].fillna(df["cost_usd"].median()).fillna(50.0)

    # Create rocket_active feature if rocket appears multiple times
    rocket_counts = df["Rocket"].value_counts()
    df["rocket_active"] = df["Rocket"].apply(lambda r: 1 if rocket_counts.get(r, 0) > 1 else 0)

    # Label Encoding
    le = LabelEncoder()
    df["company_encoded"] = le.fit_transform(df["Company Name"].astype(str))
    df["location_encoded"] = le.fit_transform(df["Location"].astype(str))
    df["rocket_encoded"] = le.fit_transform(df["Rocket"].astype(str))

    # Final feature matrix X
    features = ["company_encoded", "location_encoded", "rocket_encoded", 
                "launch_year", "launch_month", "cost_usd", "rocket_active"]
    X = df[features]
    y = df["failure_label"].values

    # Normalize numeric features
    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=features)

    # SMOTE Oversampling to handle imbalance in training data
    from imblearn.over_sampling import SMOTE
    try:
        sm = SMOTE(random_state=42)
        X_resampled, y_resampled = sm.fit_resample(X_scaled, y)
        print(f"    [SMOTE] Training data resampled to {len(y_resampled)} samples")
    except Exception as e:
        print(f"    [Warning] SMOTE failed: {e}")
        X_resampled, y_resampled = X_scaled, y

    # Train test split stratified
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled.values, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
    )

    _ensure_dirs()
    np.savez(os.path.join(PROCESSED_DIR, "launch_train.npz"), X=X_train, y=y_train)
    np.savez(os.path.join(PROCESSED_DIR, "launch_test.npz"), X=X_test, y=y_test)

    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test, features


# ---------- Satellite / CMAPSS Preprocessing ---------- #

def preprocess_satellite(path):
    """Compute Health Index, normalize sensors, generate RUL sequences."""
    print("[DataPipeline] Preprocessing Satellite Telemetry (CMAPSS) dataset...")
    df = pd.read_csv(path)

    sensor_cols = [c for c in df.columns if c.startswith("s") and c[1:].isdigit()]
    setting_cols = [c for c in df.columns if c.startswith("setting")]

    # Compute max cycle per engine for RUL
    max_cycles = df.groupby("engine_id")["cycle"].max().reset_index()
    max_cycles.columns = ["engine_id", "max_cycle"]
    df = df.merge(max_cycles, on="engine_id")
    if "RUL" not in df.columns:
        df["RUL"] = df["max_cycle"] - df["cycle"]
    df.drop(columns=["max_cycle"], inplace=True)

    # Health Index: linearly interpolated from 1 (start) to 0 (failure) per engine
    df["health_index"] = df.groupby("engine_id")["cycle"].transform(
        lambda x: 1.0 - (x - x.min()) / max(x.max() - x.min(), 1)
    )

    # Normalize sensor readings
    scaler = StandardScaler()
    df[sensor_cols] = scaler.fit_transform(df[sensor_cols])

    # Anomaly label — if not present, derive from health_index
    if "anomaly_label" not in df.columns:
        df["anomaly_label"] = (df["health_index"] < 0.15).astype(int)

    # Split by engine_id
    engine_ids = df["engine_id"].unique()
    np.random.seed(42)
    np.random.shuffle(engine_ids)
    split_idx = int(len(engine_ids) * 0.8)
    train_engines = engine_ids[:split_idx]
    test_engines = engine_ids[split_idx:]

    train_df = df[df["engine_id"].isin(train_engines)].copy()
    test_df = df[df["engine_id"].isin(test_engines)].copy()

    _ensure_dirs()
    train_df.to_csv(os.path.join(PROCESSED_DIR, "telemetry_train.csv"), index=False)
    test_df.to_csv(os.path.join(PROCESSED_DIR, "telemetry_test.csv"), index=False)
    df.to_csv(os.path.join(PROCESSED_DIR, "telemetry_with_health.csv"), index=False)

    print(f"  Total: {df.shape}, Train engines: {len(train_engines)}, Test engines: {len(test_engines)}")
    return train_df, test_df, sensor_cols


# ---------- NEO Orbit Preprocessing ---------- #

def preprocess_neo(orbital_path, close_path):
    """Parse orbital parameters, compute MOID and orbital vectors."""
    print("[DataPipeline] Preprocessing NEO datasets...")
    df_orb = pd.read_csv(orbital_path)
    df_close = pd.read_csv(close_path)

    # --- Orbital dataset ---
    orbital_params = ["e", "a", "q", "ad", "i", "om", "w", "ma", "per"]
    available = [c for c in orbital_params if c in df_orb.columns]

    # Compute MOID if not present
    if "moid_au" not in df_orb.columns:
        if "q" in df_orb.columns:
            df_orb["moid_au"] = (df_orb["q"] - 1.0).abs()
        else:
            df_orb["moid_au"] = 0.05

    # Collision risk label
    if "collision_risk" not in df_orb.columns:
        df_orb["collision_risk"] = (df_orb["moid_au"] < 0.05).astype(float)

    # Compute orbital vectors (position in orbital plane)
    if "a" in df_orb.columns and "e" in df_orb.columns:
        df_orb["semi_latus_rectum"] = df_orb["a"] * (1 - df_orb["e"] ** 2)
        if "ma" in df_orb.columns:
            # True anomaly approximation
            df_orb["true_anomaly"] = np.radians(df_orb["ma"])
            df_orb["r_orbital"] = df_orb["semi_latus_rectum"] / (
                1 + df_orb["e"] * np.cos(df_orb["true_anomaly"])
            )

    # --- Close approaches dataset ---
    df_close["dist"] = pd.to_numeric(df_close["dist"], errors="coerce")
    df_close["v_rel"] = pd.to_numeric(df_close["v_rel"], errors="coerce")
    df_close["v_inf"] = pd.to_numeric(df_close["v_inf"], errors="coerce")
    df_close["h"] = pd.to_numeric(df_close["h"], errors="coerce")
    df_close = df_close.dropna(subset=["dist", "v_rel"])

    # Compute close-approach risk score
    df_close["approach_risk"] = np.clip(1.0 / (df_close["dist"] * 1000 + 1e-6), 0, 1)

    _ensure_dirs()
    df_orb.to_csv(os.path.join(PROCESSED_DIR, "neo_orbital_processed.csv"), index=False)
    df_close.to_csv(os.path.join(PROCESSED_DIR, "neo_close_processed.csv"), index=False)

    # Prepare training data for residual model from close approaches
    feature_cols = ["dist", "v_rel", "v_inf", "h"]
    available_feats = [c for c in feature_cols if c in df_close.columns]
    X_neo = df_close[available_feats].fillna(0).values
    y_neo = df_close["approach_risk"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X_neo, y_neo, test_size=0.2, random_state=42
    )
    np.savez(os.path.join(PROCESSED_DIR, "neo_train.npz"), X=X_train, y=y_train)
    np.savez(os.path.join(PROCESSED_DIR, "neo_test.npz"), X=X_test, y=y_test)

    print(f"  Orbital: {df_orb.shape}, Close Approaches: {df_close.shape}")
    return df_orb, df_close


# ---------- Master Pipeline ---------- #

def run_pipeline():
    """Run the complete data preprocessing pipeline."""
    print("=" * 60)
    print("ORBITRON — Data Preprocessing Pipeline")
    print("=" * 60)

    datasets = detect_datasets()
    results = {}

    # Launch
    if datasets["launch"]:
        X_train, X_test, y_train, y_test, feat_names = preprocess_launch(datasets["launch"])
        results["launch"] = {
            "X_train": X_train, "X_test": X_test,
            "y_train": y_train, "y_test": y_test,
            "feature_names": feat_names
        }

    # Satellite
    if datasets["satellite"]:
        train_df, test_df, sensor_cols = preprocess_satellite(datasets["satellite"])
        results["satellite"] = {
            "train_df": train_df, "test_df": test_df,
            "sensor_cols": sensor_cols
        }

    # NEO
    if datasets["neo_orbital"] or datasets["neo_close"]:
        orb_path = datasets["neo_orbital"]
        close_path = datasets["neo_close"]
        if orb_path and close_path:
            df_orb, df_close = preprocess_neo(orb_path, close_path)
            results["neo"] = {"orbital": df_orb, "close": df_close}

    print("\n[DataPipeline] Preprocessing complete. Processed files saved to:", PROCESSED_DIR)
    return results


if __name__ == "__main__":
    run_pipeline()
