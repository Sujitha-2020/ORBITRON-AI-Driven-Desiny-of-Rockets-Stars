"""
ORBITRON — Interactive Streamlit Dashboard (Research-Grade)
Pages: Mission Overview, Launch Risk Analysis, Satellite Health Monitor,
NEO Orbit Visualization, Explainability, Model Evaluation
"""

import os
import sys
import json
import glob
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

PLOT_DIR = os.path.join(PROJECT_ROOT, "outputs", "plots")
SHAP_DIR = os.path.join(PROJECT_ROOT, "outputs", "shap")


def load_results():
    results_path = os.path.join(PROJECT_ROOT, "outputs", "results.json")
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            return json.load(f)
    return None


def risk_gauge(value, title):
    colors = {0.2: "#00CC66", 0.4: "#66CC00", 0.6: "#FFCC00", 0.8: "#FF6600", 1.01: "#FF0000"}
    color = "#888"
    for thresh, c in colors.items():
        if value < thresh:
            color = c
            break

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100,
        number={"suffix": "%", "font": {"size": 36, "color": "white"}},
        title={"text": title, "font": {"size": 15, "color": "#ccc"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#555"},
            "bar": {"color": color},
            "bgcolor": "#030E40",
            "steps": [
                {"range": [0, 20], "color": "#0a2e0a"},
                {"range": [20, 40], "color": "#2e2e0a"},
                {"range": [40, 60], "color": "#3e2e0a"},
                {"range": [60, 80], "color": "#3e1a0a"},
                {"range": [80, 100], "color": "#3e0a0a"},
            ],
            "threshold": {"line": {"color": "#FF4444", "width": 3}, "thickness": 0.75, "value": 80},
        },
    ))
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=50, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
    return fig


# ─────────────── PAGE: Mission Overview ─────────────── #

def page_mission_overview(results):
    st.title("🛰️ ORBITRON — Mission Risk Intelligence")
    st.markdown("---")

    if results is None:
        st.warning("No results. Run `python run.py` first.")
        return

    risk = results.get("mission_risk", {})
    mri = risk.get("mission_risk_index", 0)
    category = risk.get("risk_category", "UNKNOWN")
    recommendation = risk.get("recommendation", "")
    components = risk.get("components", {})

    cat_colors = {"LOW": "#00CC66", "MODERATE": "#4FC3F7", "ELEVATED": "#FFD54F",
                  "HIGH": "#FF6600", "CRITICAL": "#FF0000"}
    c = cat_colors.get(category, "#888")

    st.markdown(f"""
    <div style='background: linear-gradient(135deg, {c}22, {c}11);
    border-left: 4px solid {c}; padding: 18px; border-radius: 10px; margin-bottom: 20px;'>
    <h2 style='margin:0; color: {c};'>Mission Status: {category}</h2>
    <p style='margin:5px 0 0 0; font-size: 17px; color: #ddd;'>{recommendation}</p>
    </div>""", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.plotly_chart(risk_gauge(mri, "Mission Risk Index"), use_container_width=True)
    with c2:
        st.plotly_chart(risk_gauge(components.get("launch_risk", 0), "Launch Risk"), use_container_width=True)
    with c3:
        st.plotly_chart(risk_gauge(components.get("satellite_risk", 0), "Satellite Risk"), use_container_width=True)

    c4, c5, c6 = st.columns(3)
    with c4:
        st.plotly_chart(risk_gauge(components.get("neo_risk", 0), "NEO Risk"), use_container_width=True)
    with c5:
        fig = go.Figure(data=[go.Pie(
            labels=["Launch (40%)", "Satellite (35%)", "NEO (25%)"],
            values=[0.4, 0.35, 0.25],
            marker_colors=["#FF6B6B", "#4ECDC4", "#45B7D1"], hole=0.4
        )])
        
        fig.update_layout(title=dict(text="Risk Weight Distribution",font = dict(color = "white",size = 10, weight = "normal")
        ),
        height=260,
        margin=dict(l=20, r=20, t=50, b=10),
        paper_bgcolor="rgba(0,0,0,0)")
        fig.update_layout(font_color="white")
        st.plotly_chart(fig, width='stretch')
    with c6:
        st.metric("Risk Score", f"{mri:.4f}")
        st.metric("Category", category)
        if "rul_metrics" in results and results["rul_metrics"]:
            st.metric("RUL MAE", f"{results['rul_metrics'].get('mae', 0):.1f} cycles")

    # Show mission intelligence plots
    _show_plot_gallery("Mission Intelligence", [
        "mission_risk_pie.png", "mission_risk_radar.png",
        "mission_risk_sensitivity.png", "mission_risk_timeline.png"
    ])


# ─────────────── PAGE: Launch Risk ─────────────── #

def page_launch_risk(results):
    st.title(" Launch Failure Prediction")
    st.markdown("---")

    if results and "launch_risk_scores" in results and results["launch_risk_scores"]:
        scores = results["launch_risk_scores"]
        c1, c2, c3 = st.columns([1, 2, 1]) # Center the gauge without the noisy bar chart
        with c2:
            st.plotly_chart(risk_gauge(np.mean(scores), "Mean Launch Risk"), use_container_width=True)

    if results and "launch_metrics" in results and results["launch_metrics"]:
        st.subheader(" Classification Metrics")
        m = results["launch_metrics"]
        cols = st.columns(5)
        metric_names = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
        metric_labels = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"]
        for col, name, label in zip(cols, metric_names, metric_labels):
            col.metric(label, f"{m.get(name, 0):.4f}")

    st.subheader(" Model Evaluation Plots")
    st.markdown("**Ensemble**: 60% XGBoost + 40% LSTM")
    _show_plot_gallery("Launch Evaluation", [
        "launch_prob_distribution.png", "launch_precision_recall.png",
        "launch_roc_curve.png", "launch_confusion_matrix.png", 
        "launch_feature_correlation.png", "launch_calibration_curve.png",
        "launch_class_distribution.png", "launch_learning_curve.png", 
        "launch_feature_importance.png"
    ])

    raw_dir = os.path.join(PROJECT_ROOT, "data", "raw")
    lf = os.path.join(raw_dir, "launch_data.csv")
    if os.path.exists(lf):
        st.subheader(" Launch Data Overview")
        st.dataframe(pd.read_csv(lf).head(20), use_container_width=True)


# ─────────────── PAGE: Satellite Health ─────────────── #

def page_satellite_health(results):
    st.title("🛰️ Satellite Health Monitor")
    st.markdown("---")

    processed_dir = os.path.join(PROJECT_ROOT, "datasets", "processed")
    train_path = os.path.join(processed_dir, "telemetry_train.csv")

    if os.path.exists(train_path):
        df = pd.read_csv(train_path)
        engines = sorted(df["engine_id"].unique())

       


        st.markdown(
            """
            <style>
            /* Forces EVERY element inside all select boxes to have a black background */
            div[data-testid="stSelectbox"] [data-baseweb="select"] * {
                background-color: #000000 !important;
                color: #ffffff !important;
            }
            
            /* The 2 lines added to turn the large open white selection dropdown box black */
            div[data-baseweb="popover"] * { background-color: #000000 !important; }
            ul[role="listbox"] * { background-color: #000000 !important; }
            
            /* Ensures the outer border of both input boxes is visible */
            div[data-testid="stSelectbox"] [data-baseweb="select"] > div {
                border: 1px solid #444444 !important;
                border-radius: 4px;
            }

            /* Turns the dropdown arrow icon white */
            div[data-testid="stSelectbox"] svg {
                fill: #000000 !important;
            }
            </style>
            """,
            unsafe_allow_html=True
        )


        selected_engine = st.selectbox("Select Engine", engines)
        engine_data = df[df["engine_id"] == selected_engine]

        sensor_cols = [c for c in df.columns if c.startswith("s") and c[1:].isdigit()]
        selected_sensors = st.multiselect("Select Sensors", sensor_cols, default=sensor_cols[:4])

        if selected_sensors:
            fig = go.Figure()
            for s in selected_sensors:
                fig.add_trace(go.Scatter(x=engine_data["cycle"], y=engine_data[s],
                                         mode="lines", name=s))
            fig.update_layout(title=f"Sensor Readings — Engine {selected_engine}",
                              xaxis_title="Cycle", yaxis_title="Value", height=400,
                              paper_bgcolor="rgba(0,0,0,0)", 
                              plot_bgcolor="#161F33",
                              font=dict(color="white"))
            st.plotly_chart(fig, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            if "health_index" in engine_data.columns:
                fig = go.Figure(data=[go.Scatter(
                    x=engine_data["cycle"], y=engine_data["health_index"],
                    mode="lines", fill="tozeroy", line=dict(color="#4ECDC4")
                )])
                fig.update_layout(title=dict(text="Health Index", font = dict(color= "white")
                ),
                xaxis_title="Cycle", height=300,
                                  paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#111827",
                                  font=dict(color="white"))
                st.plotly_chart(fig, use_container_width=True)
        with c2:
            if "RUL" in engine_data.columns:
                fig = go.Figure(data=[go.Scatter(
                    x=engine_data["cycle"], y=engine_data["RUL"],
                    mode="lines", line=dict(color="#FF6B6B")
                )])
                fig.update_layout(title="Remaining Useful Life",
                                  xaxis_title="Cycle", yaxis_title="RUL (cycles)",
                                  height=300, paper_bgcolor="rgba(0,0,0,0)",
                                  plot_bgcolor="#111827", font=dict(color="white"))
                st.plotly_chart(fig, use_container_width=True)

    if results:
        c1, c2, c3 = st.columns(3)
        c1.metric("Anomaly Rate", f"{results.get('anomaly_rate', 0):.2%}")
        c2.metric("Satellite Risk", f"{results.get('satellite_risk', 0):.4f}")
        if results.get("rul_metrics"):
            m = results["rul_metrics"]
            c3.metric("RUL R²", f"{m.get('r2', 0):.4f}")

    st.subheader(" Satellite Health Analysis Plots")
    _show_plot_gallery("Satellite Analysis", [
        "satellite_sensor_correlation.png", "satellite_pca_projection.png",
        "satellite_recon_error_hist.png", "satellite_anomaly_timeline.png",
        "satellite_health_degradation.png", "satellite_rul_scatter.png",
        "satellite_sensor_trends.png"
    ])


# ─────────────── PAGE: NEO Orbit ─────────────── #

def page_neo_orbit(results):
    st.title(" Near-Earth Object Risk Analysis")
    st.markdown("---")

    raw_dir = os.path.join(PROJECT_ROOT, "data", "raw")
    orb_path = os.path.join(raw_dir, "neo_orbital.csv")
    close_path = os.path.join(raw_dir, "neo_close_approaches.csv")

    if os.path.exists(orb_path):
        df_orb = pd.read_csv(orb_path)
        st.subheader(" NEO Orbital Elements")
        st.dataframe(df_orb, use_container_width=True)

        if "a" in df_orb.columns and "e" in df_orb.columns:
            fig = go.Figure()
            theta = np.linspace(0, 2 * np.pi, 200)
            fig.add_trace(go.Scatter3d(
                x=np.cos(theta), y=np.sin(theta), z=np.zeros(200),
                mode="lines", name="Earth Orbit", line=dict(color="#4ECDC4", width=3)
            ))
            fig.add_trace(go.Scatter3d(
                x=[0], y=[0], z=[0], mode="markers", name="Sun",
                marker=dict(size=10, color="yellow")
            ))
            colors = px.colors.qualitative.Set1
            for idx, row in df_orb.iterrows():
                a, e_val = row.get("a", 1.5), row.get("e", 0.3)
                i_val = np.radians(row.get("i", 10))
                nu = np.linspace(0, 2 * np.pi, 200)
                r = a * (1 - e_val**2) / (1 + e_val * np.cos(nu))
                x_o, y_o = r * np.cos(nu), r * np.sin(nu)
                z_o = r * np.sin(nu) * np.sin(i_val)
                y_r = y_o * np.cos(i_val)
                fig.add_trace(go.Scatter3d(
                    x=x_o, y=y_r, z=z_o, mode="lines",
                    name=str(row.get("full_name", f"NEO_{idx}")),
                    line=dict(color=colors[idx % len(colors)], width=2)
                ))
            fig.update_layout(
                title="NEO Orbital Trajectories (3D)", height=600,
                scene=dict(xaxis_title="X (AU)", yaxis_title="Y (AU)", zaxis_title="Z (AU)",
                           aspectmode="cube",
                           xaxis=dict(backgroundcolor="#111827", gridcolor="#333"),
                           yaxis=dict(backgroundcolor="#111827", gridcolor="#333"),
                           zaxis=dict(backgroundcolor="#111827", gridcolor="#333")),
                paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white")
            )
            st.plotly_chart(fig, use_container_width=True)

    if os.path.exists(close_path):
        df_close = pd.read_csv(close_path)
        df_close["dist"] = pd.to_numeric(df_close["dist"], errors="coerce")
        st.subheader(" Close Approach Distribution")
        fig = px.histogram(df_close, x="dist", nbins=50,
                           title="Distribution of Close Approach Distances",
                           labels={"dist": "Distance (AU)"})
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#111827",
                          font=dict(color="white"))
        st.plotly_chart(fig, use_container_width=True)

    if results and "neo_collision_probs" in results:
        st.subheader(" Monte Carlo Collision Probabilities")
        for neo in results["neo_collision_probs"]:
            col1, col2 = st.columns(2)
            col1.metric(f"{neo['name']} — P(collision)", f"{neo['collision_probability']:.6f}")
            col2.metric("Mean MOID", f"{neo['moid_mean']:.4f} AU")

    st.subheader(" NEO Analysis Plots")
    _show_plot_gallery("NEO Analysis", [
        "neo_moid_distribution.png", "neo_monte_carlo_orbits.png",
        "neo_inclination_axis.png", "neo_diameter_distribution.png",
        "neo_pha_count.png", "neo_prediction_error.png", 
        "neo_residual_errors.png"
    ])


# ─────────────── PAGE: Explainability ─────────────── #

def page_explainability():
    st.title(" Model Explainability (SHAP)")
    st.markdown("---")

    if not os.path.exists(SHAP_DIR):
        st.warning("No SHAP plots found. Run `python run.py` first.")
        return

    shap_files = sorted([f for f in os.listdir(SHAP_DIR) if f.endswith(".png")])
    if not shap_files:
        st.warning("No SHAP images generated yet.")
        return

    # Categorize SHAP files
    summary_files = [f for f in shap_files if "summary" in f or "bar" in f]
    dependence_files = [f for f in shap_files if "dependence" in f]
    other_files = [f for f in shap_files if f not in summary_files and f not in dependence_files]

    st.subheader("Summary & Feature Importance")
    for f in summary_files:
        st.markdown(f"**{f.replace('_', ' ').replace('.png', '').title()}**")
        st.image(os.path.join(SHAP_DIR, f), use_container_width=True)

    if dependence_files:
        st.markdown("---")
        st.subheader("Feature Dependence Plots")
        for f in dependence_files:
            st.markdown(f"**{f.replace('_', ' ').replace('.png', '').title()}**")
            st.image(os.path.join(SHAP_DIR, f), use_container_width=True)

    if other_files:
        st.markdown("---")
        st.subheader("Other SHAP Visualizations")
        for f in other_files:
            st.markdown(f"**{f.replace('_', ' ').replace('.png', '').title()}**")
            st.image(os.path.join(SHAP_DIR, f), use_container_width=True)


# ─────────────── PAGE: Model Evaluation ─────────────── #

def page_model_evaluation(results):
    st.title("Model Evaluation & Scientific Analysis")
    st.markdown("---")

    # Metrics summary
    if results:
        st.subheader(" Performance Summary")

        if results.get("launch_metrics"):
            st.markdown("#### Launch Failure Prediction (Ensemble)")
            m = results["launch_metrics"]
            cols = st.columns(5)
            for col, (k, label) in zip(cols, [
                ("accuracy", "Accuracy"), ("precision", "Precision"),
                ("recall", "Recall"), ("f1_score", "F1 Score"), ("roc_auc", "ROC AUC")
            ]):
                val = m.get(k, 0)
                col.metric(label, f"{val:.4f}")

        if results.get("rul_metrics"):
            st.markdown("#### Satellite RUL Estimation")
            m = results["rul_metrics"]
            c1, c2, c3 = st.columns(3)
            c1.metric("MAE", f"{m.get('mae', 0):.2f} cycles")
            c2.metric("RMSE", f"{m.get('rmse', 0):.2f} cycles")
            c3.metric("R²", f"{m.get('r2', 0):.4f}")

        st.metric("Satellite Anomaly Rate", f"{results.get('anomaly_rate', 0):.2%}")

    # All generated plots
    st.subheader(" Complete Visualization Gallery")

    tabs = st.tabs([" Launch", " Satellite", " NEO", "Mission"])

    with tabs[0]:
        _show_plot_gallery("Launch Failure Evaluation", [
            "launch_prob_distribution.png", "launch_precision_recall.png",
            "launch_roc_curve.png", "launch_confusion_matrix.png", 
            "launch_feature_correlation.png", "launch_calibration_curve.png",
            "launch_class_distribution.png", "launch_learning_curve.png", 
            "launch_feature_importance.png"
        ])

    with tabs[1]:
        _show_plot_gallery("Satellite Health Analysis", [
            "satellite_sensor_correlation.png", "satellite_pca_projection.png",
            "satellite_recon_error_hist.png", "satellite_anomaly_timeline.png",
            "satellite_health_degradation.png", "satellite_rul_scatter.png",
            "satellite_sensor_trends.png"
        ])

    with tabs[2]:
        _show_plot_gallery("NEO Orbit Analysis", [
            "neo_moid_distribution.png", "neo_monte_carlo_orbits.png",
            "neo_inclination_axis.png", "neo_diameter_distribution.png",
            "neo_pha_count.png", "neo_prediction_error.png", 
            "neo_residual_errors.png"
        ])

    with tabs[3]:
        _show_plot_gallery("Mission Intelligence", [
            "mission_risk_pie.png", "mission_risk_radar.png",
            "mission_risk_sensitivity.png", "mission_risk_timeline.png"
        ])


# ─────────────── HELPERS ─────────────── #

def _show_plot_gallery(section_title, filenames):
    """Display plots from outputs/plots/ in a responsive 2-column grid."""
    available = [f for f in filenames if os.path.exists(os.path.join(PLOT_DIR, f))]
    if not available:
        st.info(f"No {section_title} plots found yet. Run `python run.py` to generate.")
        return

    for i in range(0, len(available), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            idx = i + j
            if idx < len(available):
                fpath = os.path.join(PLOT_DIR, available[idx])
                label = available[idx].replace("_", " ").replace(".png", "").title()
                col.image(fpath, caption=label, width="stretch" if st.__version__ >= "1.30.0" else None, use_container_width=True)



# ─────────────── MAIN ─────────────── #

def main():
    st.set_page_config(
        page_title="ORBITRON — Mission Risk Intelligence",
        page_icon="🛰️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown("""
    <style>
    .stApp {
        background : #05386b;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
    }
    [data-testid="stSidebar"] h1 { font-size: 1.5rem !important; }
    .stMetric {
        background: rgba(255,255,255,0.04);
        padding: 0.8rem;
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.08);
    }
    h1, h2, h3,p, span, div , label { color: #ffffff !important; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background: rgba(255,255,255,0.05);
        border-radius: 8px;
        padding: 8px 16px;
        color: white !important;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(79, 195, 247, 0.15) !important;
        border-bottom: 2px solid #4FC3F7 !important;
    }
    </style>""", unsafe_allow_html=True)

    st.sidebar.title("🛰️ ORBITRON")
    st.sidebar.markdown("*Mission Risk Intelligence*")
    st.sidebar.markdown("---")
    page = st.sidebar.radio("Navigate", [
        " Mission Overview",
        " Launch Risk Analysis",
        "Satellite Health Monitor",
        " NEO Orbit Visualization",
        " Explainability",
        " Model Evaluation"
    ])

    results = load_results()

    if "Mission Overview" in page:
        page_mission_overview(results)
    elif "Launch Risk" in page:
        page_launch_risk(results)
    elif "Satellite Health" in page:
        page_satellite_health(results)
    elif "NEO Orbit" in page:
        page_neo_orbit(results)
    elif "Explainability" in page:
        page_explainability()
    elif "Model Evaluation" in page:
        page_model_evaluation(results)


if __name__ == "__main__":
    main()
