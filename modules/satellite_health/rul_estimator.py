"""
Satellite Health — Remaining Useful Life (RUL) Estimator
Uses RandomForestRegressor to predict RUL from sensor data and cycle info.
"""

import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def train_rul_model(train_df, test_df, sensor_cols, model_dir):
    """Train RUL regression model using RandomForestRegressor."""
    print("\n[SatelliteHealth] Training RUL Estimator...")

    feature_cols = ["cycle"] + sensor_cols
    available = [c for c in feature_cols if c in train_df.columns]

    X_train = train_df[available].values
    y_train = train_df["RUL"].values

    X_test = test_df[available].values
    y_test = test_df["RUL"].values

    # Cap RUL at 125 (common practice for CMAPSS)
    y_train = np.minimum(y_train, 125)
    y_test = np.minimum(y_test, 125)

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"  MAE:  {mae:.2f} cycles")
    print(f"  RMSE: {rmse:.2f} cycles")
    print(f"  R²:   {r2:.4f}")

    # Save model
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "rul_estimator.joblib")
    joblib.dump(model, model_path)
    print(f"  Model saved to: {model_path}")

    return model, y_pred, {"mae": mae, "rmse": rmse, "r2": r2}


def load_model(model_dir):
    """Load trained RUL model."""
    return joblib.load(os.path.join(model_dir, "rul_estimator.joblib"))
