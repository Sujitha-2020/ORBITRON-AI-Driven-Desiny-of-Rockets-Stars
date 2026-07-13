"""
NEO Orbit — ML Residual Correction Model
Layer 2: GradientBoostingRegressor trained on residual MOID errors
to correct physics-based orbit predictions.
"""

import os
import numpy as np
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def train_residual_model(X_train, y_train, X_test, y_test, model_dir):
    """
    Train GradientBoostingRegressor to predict residual errors in MOID.
    corrected_orbit = physics_orbit + residual_prediction
    """
    print("\n[NEOOrbit] Training Residual Correction Model (GBR)...")

    model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"  MAE:  {mae:.6f}")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  R²:   {r2:.4f}")

    # Save model
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "neo_residual_gbr.joblib")
    joblib.dump(model, model_path)
    print(f"  Model saved to: {model_path}")

    return model, y_pred, {"mae": mae, "rmse": rmse, "r2": r2}


def correct_prediction(physics_prediction, residual_prediction):
    """Apply residual correction: corrected = physics + residual."""
    return physics_prediction + residual_prediction


def load_model(model_dir):
    """Load trained residual model."""
    return joblib.load(os.path.join(model_dir, "neo_residual_gbr.joblib"))
