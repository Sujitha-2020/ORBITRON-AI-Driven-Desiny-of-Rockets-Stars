"""
ORBITRON — Master Pipeline (Research-Grade)
Runs the complete ML aerospace risk intelligence system end-to-end,
including full model evaluation and scientific visualizations.

Usage:
    python run.py
"""

import os
import sys
import json
import warnings
import subprocess
import numpy as np


warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

MODEL_DIR = os.path.join(ROOT, "models")
OUTPUT_DIR = os.path.join(ROOT, "outputs")
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
SHAP_DIR = os.path.join(OUTPUT_DIR, "shap")
PROCESSED_DIR = os.path.join(ROOT, "datasets", "processed")


def setup_dirs():
    for d in [MODEL_DIR, OUTPUT_DIR, PLOT_DIR, SHAP_DIR, PROCESSED_DIR]:
        os.makedirs(d, exist_ok=True)


def step_1_preprocess():
    """Step 1: Data Preprocessing."""
    print("\n" + "=" * 60)
    print("STEP 1 — DATA PREPROCESSING")
    print("=" * 60)
    from preprocessing.data_pipeline import run_pipeline
    return run_pipeline()


def step_2_launch_failure(data):
    """Step 2: Launch Failure Prediction (XGBoost + LSTM Ensemble)."""
    print("\n" + "=" * 60)
    print("STEP 2 — LAUNCH FAILURE PREDICTION")
    print("=" * 60)

    launch = data.get("launch")
    if launch is None:
        print("[SKIP] No launch data available.")
        return None, None, None, None, None, None

    X_train = launch["X_train"]
    X_test = launch["X_test"]
    y_train = launch["y_train"]
    y_test = launch["y_test"]

    if not isinstance(X_train, np.ndarray):
        X_train = X_train.values
        X_test = X_test.values

    # XGBoost
    from modules.launch_failure.xgboost_model import train_xgboost
    xgb_model, xgb_probs = train_xgboost(X_train, y_train, X_test, y_test, MODEL_DIR)

    # LSTM
    from modules.launch_failure.lstm_model import train_lstm
    lstm_model, lstm_probs = train_lstm(X_train, y_train, X_test, y_test, MODEL_DIR)

    # Ensemble
    from modules.launch_failure.ensemble import compute_ensemble
    launch_risk = compute_ensemble(xgb_probs, lstm_probs)

    return xgb_model, lstm_model, launch_risk, X_train, X_test, (y_train, y_test, xgb_probs, lstm_probs)


def step_3_satellite_health(data):
    """Step 3: Satellite Health Monitoring."""
    print("\n" + "=" * 60)
    print("STEP 3 — SATELLITE HEALTH MONITORING")
    print("=" * 60)

    sat = data.get("satellite")
    if sat is None:
        print("[SKIP] No satellite data available.")
        return None, None, None, None, None, None, None

    train_df = sat["train_df"]
    test_df = sat["test_df"]
    sensor_cols = sat["sensor_cols"]

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_sensors = scaler.fit_transform(train_df[sensor_cols].values)
    X_test_sensors = scaler.transform(test_df[sensor_cols].values)

    # Normal data for autoencoder
    if "health_index" in train_df.columns:
        normal_mask = train_df["health_index"] > 0.5
        X_train_normal = X_train_sensors[normal_mask.values]
    else:
        X_train_normal = X_train_sensors[:int(len(X_train_sensors) * 0.7)]

    from modules.satellite_health.autoencoder import train_autoencoder
    ae_model, threshold, train_errors = train_autoencoder(
        X_train_normal, MODEL_DIR, epochs=80, batch_size=64
    )

    from modules.satellite_health.anomaly_detection import detect_anomalies, compute_satellite_risk
    anomalies, recon_errors = detect_anomalies(ae_model, X_test_sensors, threshold)
    sat_risk = compute_satellite_risk(anomalies, recon_errors, threshold)
    print(f"  Satellite Risk Score: {sat_risk:.4f}")

    from modules.satellite_health.rul_estimator import train_rul_model
    rul_model, rul_pred, rul_metrics = train_rul_model(
        train_df, test_df, sensor_cols, MODEL_DIR
    )

    return ae_model, sat_risk, rul_metrics, anomalies, recon_errors, threshold, rul_pred


def step_4_neo_orbit(data):
    """Step 4: NEO Orbit Prediction."""
    print("\n" + "=" * 60)
    print("STEP 4 — NEO ORBIT RISK ANALYSIS")
    print("=" * 60)

    neo = data.get("neo")
    if neo is None:
        print("[SKIP] No NEO data available.")
        return None, None, None, None, None

    df_orb = neo["orbital"]
    df_close = neo["close"]

    from modules.neo_orbit.kepler_propagator import propagate_all_neos
    neo_results = propagate_all_neos(df_orb)

    # Residual model
    train_path = os.path.join(PROCESSED_DIR, "neo_train.npz")
    test_path = os.path.join(PROCESSED_DIR, "neo_test.npz")
    res_pred = None
    res_y_test = None

    if os.path.exists(train_path) and os.path.exists(test_path):
        train_data = np.load(train_path)
        test_data = np.load(test_path)
        from modules.neo_orbit.residual_model import train_residual_model
        res_model, res_pred, res_metrics = train_residual_model(
            train_data["X"], train_data["y"],
            test_data["X"], test_data["y"],
            MODEL_DIR
        )
        res_y_test = test_data["y"]

    from modules.neo_orbit.collision_probability import compute_all_collision_probabilities
    collision_results = compute_all_collision_probabilities(neo_results, n_simulations=2000)

    max_prob = max(c["collision_probability"] for c in collision_results) if collision_results else 0.0

    return collision_results, max_prob, neo_results, (res_y_test, res_pred), df_orb


def step_5_explainability(xgb_model, lstm_model, X_data, feature_names):
    """Step 5: SHAP Explainability."""
    print("\n" + "=" * 60)
    print("STEP 5 — EXPLAINABILITY (SHAP)")
    print("=" * 60)

    if xgb_model is None:
        print("[SKIP] No models to explain.")
        return

    from explainability.shap_analysis import run_shap_analysis
    run_shap_analysis(xgb_model, lstm_model, X_data, feature_names, OUTPUT_DIR)


def step_6_evaluation(data, xgb_model, lstm_model, launch_extras,
                       sat_data, sat_extras, neo_extras, neo_propagation,
                       launch_risk_mean, sat_risk, neo_risk):
    """Step 6: Full Model Evaluation & Scientific Visualizations."""
    print("\n" + "=" * 60)
    print("STEP 6 — RESEARCH-GRADE EVALUATION & VISUALIZATION")
    print("=" * 60)

    launch_metrics = None

    # --- Launch evaluation plots ---
    if xgb_model is not None and launch_extras is not None:
        y_train, y_test, xgb_probs, lstm_probs = launch_extras
        X_train = data["launch"]["X_train"]
        X_test = data["launch"]["X_test"]
        if not isinstance(X_train, np.ndarray):
            X_train = X_train.values
            X_test = X_test.values
        feature_names = data["launch"].get("feature_names",
                                            [f"f{i}" for i in range(X_train.shape[1])])

        from outputs.evaluation.launch_eval import generate_all_launch_plots
        launch_metrics, _ = generate_all_launch_plots(
            xgb_model, X_train, y_train, X_test, y_test,
            xgb_probs, lstm_probs, feature_names, PLOT_DIR
        )

    # --- Satellite evaluation plots ---
    if sat_data is not None and sat_extras is not None:
        train_df, test_df, sensor_cols = sat_data
        anomalies, recon_errors, threshold, rul_pred = sat_extras

        from outputs.evaluation.satellite_eval import generate_all_satellite_plots
        generate_all_satellite_plots(
            train_df, test_df, sensor_cols, None,
            threshold, rul_pred, anomalies, recon_errors, PLOT_DIR
        )

    # --- NEO evaluation plots ---
    if neo_extras is not None:
        collision_results, neo_prop_results, (res_y_test, res_pred), df_orb_eval = neo_extras

        from outputs.evaluation.neo_eval import generate_all_neo_plots
        generate_all_neo_plots(
            neo_prop_results, collision_results,
            res_y_test, res_pred, df_orb_eval, PLOT_DIR
        )

    # --- Mission intelligence plots ---
    from outputs.evaluation.mission_eval import generate_all_mission_plots
    generate_all_mission_plots(launch_risk_mean, sat_risk, neo_risk, PLOT_DIR)

    return launch_metrics


def step_7_fusion(launch_risk, satellite_risk, neo_risk):
    """Step 7: Mission Risk Fusion."""
    print("\n" + "=" * 60)
    print("STEP 7 — MISSION RISK FUSION")
    print("=" * 60)

    from modules.fusion.mission_risk_index import generate_risk_report
    return generate_risk_report(launch_risk, satellite_risk, neo_risk)


def step_8_dashboard():
    """Step 8: Launch Streamlit Dashboard."""
    print("\n" + "=" * 60)
    print("STEP 8 — LAUNCHING DASHBOARD")
    print("=" * 60)
    print("\nStarting Streamlit dashboard...")
    print("Open your browser at: http://localhost:8501\n")

    app_path = os.path.join(ROOT, "app", "app.py")
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", app_path,
        "--server.headless", "true",
        "--server.port", "8501",
        "--browser.gatherUsageStats", "false"
    ])


def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║              ORBITRON v2.0 (Research Grade)             ║")
    print("║         Mission Risk Intelligence System                ║")
    print("║         Aerospace AI — Full Evaluation Pipeline         ║")
    print("╚══════════════════════════════════════════════════════════╝")

    setup_dirs()

    # Step 1: Preprocess
    data = step_1_preprocess()

    # Step 2: Launch Failure
    result = step_2_launch_failure(data)
    xgb_model, lstm_model, launch_risk_scores = result[0], result[1], result[2]
    X_train_l, X_test_l = result[3], result[4]
    launch_extras = result[5]  # (y_train, y_test, xgb_probs, lstm_probs)
    mean_launch_risk = float(np.mean(launch_risk_scores)) if launch_risk_scores is not None else 0.0

    # Step 3: Satellite Health
    sat_result = step_3_satellite_health(data)
    ae_model = sat_result[0]
    satellite_risk = sat_result[1] if sat_result[1] is not None else 0.0
    rul_metrics = sat_result[2]
    anomalies_arr = sat_result[3]
    recon_errors_arr = sat_result[4]
    ae_threshold = sat_result[5]
    rul_pred = sat_result[6]

    sat_data_tuple = None
    sat_extras_tuple = None
    if data.get("satellite"):
        sat_data_tuple = (data["satellite"]["train_df"],
                          data["satellite"]["test_df"],
                          data["satellite"]["sensor_cols"])
        sat_extras_tuple = (anomalies_arr, recon_errors_arr, ae_threshold, rul_pred)

    # Step 4: NEO Orbit
    neo_result = step_4_neo_orbit(data)
    collision_results = neo_result[0] if neo_result else None
    neo_risk = neo_result[1] if (neo_result and neo_result[1] is not None) else 0.0
    neo_prop_results = neo_result[2] if neo_result else None
    neo_residual_data = neo_result[3] if neo_result else None
    df_orb_main = neo_result[4] if neo_result else None

    neo_extras_tuple = None
    if collision_results is not None:
        neo_extras_tuple = (collision_results, neo_prop_results, neo_residual_data, df_orb_main)

    # Step 5: Explainability
    if data.get("launch"):
        feature_names = data["launch"].get("feature_names",
                                            [f"f{i}" for i in range(X_test_l.shape[1])])
    else:
        feature_names = []
    step_5_explainability(xgb_model, lstm_model, X_test_l, feature_names)

    # Step 6: Full Evaluation & Visualization
    launch_metrics = step_6_evaluation(
        data, xgb_model, lstm_model, launch_extras,
        sat_data_tuple, sat_extras_tuple, neo_extras_tuple, neo_prop_results,
        mean_launch_risk, satellite_risk, neo_risk
    )

    # Step 7: Fusion
    report = step_7_fusion(mean_launch_risk, satellite_risk, neo_risk)

    # Save results
    results = {
        "mission_risk": report,
        "launch_risk_scores": launch_risk_scores.tolist() if launch_risk_scores is not None else [],
        "launch_metrics": launch_metrics if launch_metrics else {},
        "satellite_risk": satellite_risk,
        "anomaly_rate": float(anomalies_arr.mean()) if anomalies_arr is not None else 0.0,
        "neo_risk": neo_risk,
        "rul_metrics": rul_metrics if rul_metrics else {},
        "neo_collision_probs": [
            {"name": c["name"], "collision_probability": c["collision_probability"],
             "moid_mean": c["moid_mean"]}
            for c in (collision_results or [])
        ]
    }
    results_path = os.path.join(OUTPUT_DIR, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Pipeline] Results saved to: {results_path}")

    # List all generated plots
    plot_files = [f for f in os.listdir(PLOT_DIR) if f.endswith(".png")]
    shap_files = [f for f in os.listdir(SHAP_DIR) if f.endswith(".png")] if os.path.exists(SHAP_DIR) else []
    print(f"\n[Pipeline] Generated {len(plot_files)} evaluation plots in {PLOT_DIR}")
    print(f"[Pipeline] Generated {len(shap_files)} SHAP plots in {SHAP_DIR}")
    for f in sorted(plot_files):
        print(f"  📊 {f}")

    # Step 8: Dashboard
    step_8_dashboard()


if __name__ == "__main__":
    main()
