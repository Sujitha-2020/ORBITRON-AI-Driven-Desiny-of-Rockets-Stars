"""
Launch Failure — Ensemble Module
Combines XGBoost (60%) and LSTM (40%) predictions into a weighted launch risk score.
"""

import numpy as np


def compute_ensemble(xgb_probs, lstm_probs, xgb_weight=0.6, lstm_weight=0.4):
    """
    Compute weighted ensemble of XGBoost and LSTM predictions.
    launch_risk = 0.6 * XGB_prediction + 0.4 * LSTM_prediction
    """
    # Ensure arrays
    xgb_probs = np.asarray(xgb_probs).flatten()
    lstm_probs = np.asarray(lstm_probs).flatten()

    # Handle shape mismatches
    min_len = min(len(xgb_probs), len(lstm_probs))
    xgb_probs = xgb_probs[:min_len]
    lstm_probs = lstm_probs[:min_len]

    launch_risk = xgb_weight * xgb_probs + lstm_weight * lstm_probs
    launch_risk = np.clip(launch_risk, 0.0, 1.0)

    print(f"\n[Ensemble] Launch Risk Score:")
    print(f"  Mean: {launch_risk.mean():.4f}")
    print(f"  Min:  {launch_risk.min():.4f}")
    print(f"  Max:  {launch_risk.max():.4f}")

    return launch_risk


def get_launch_risk_single(xgb_model, lstm_model, X_sample):
    """Get launch risk for a single sample."""
    xgb_prob = xgb_model.predict_proba(X_sample.reshape(1, -1))[:, 1][0]

    X_3d = X_sample.reshape(1, 1, -1)
    lstm_prob = lstm_model.predict(X_3d, verbose=0).flatten()[0]

    risk = 0.6 * xgb_prob + 0.4 * lstm_prob
    return float(np.clip(risk, 0.0, 1.0))
