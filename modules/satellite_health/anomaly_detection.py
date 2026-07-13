"""
Satellite Health — Anomaly Detection
Uses autoencoder reconstruction error to detect anomalies.
Threshold: mean + 2 * std
"""

import numpy as np


def detect_anomalies(autoencoder, X_data, threshold):
    """
    Detect anomalies using reconstruction error.
    If error > threshold → anomaly.
    """
    reconstructed = autoencoder.predict(X_data, verbose=0)
    recon_errors = np.mean((X_data - reconstructed) ** 2, axis=1)
    anomalies = (recon_errors > threshold).astype(int)

    anomaly_rate = anomalies.mean()
    print(f"\n[AnomalyDetection] Results:")
    print(f"  Total samples: {len(X_data)}")
    print(f"  Anomalies detected: {anomalies.sum()} ({anomaly_rate:.2%})")
    print(f"  Threshold: {threshold:.6f}")
    print(f"  Max reconstruction error: {recon_errors.max():.6f}")

    return anomalies, recon_errors


def compute_satellite_risk(anomalies, recon_errors, threshold):
    """
    Compute a satellite risk score (0-1).
    Based strictly on anomaly rate as requested.
    """
    anomaly_rate = anomalies.mean() if len(anomalies) > 0 else 0.0
    # Map anomaly rate slightly to a high risk score, e.g. 5% anomalies = 0.5 risk
    risk = min(anomaly_rate * 10.0, 1.0)
    return float(np.clip(risk, 0.0, 1.0))
