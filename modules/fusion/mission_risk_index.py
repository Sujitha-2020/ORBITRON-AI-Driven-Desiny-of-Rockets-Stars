"""
ORBITRON — Mission Risk Fusion Module
Combines launch risk, satellite anomaly risk, and NEO collision probability
into a unified Mission Risk Index.
"""

import numpy as np


def compute_mission_risk(launch_risk, satellite_risk, neo_risk,
                         w_launch=0.4, w_satellite=0.35, w_neo=0.25):
    """
    Compute Mission Risk Index as weighted combination:
        mission_risk = 0.4 * launch_risk + 0.35 * satellite_risk + 0.25 * neo_risk
    Normalized to [0, 1].
    """
    mission_risk = (
        w_launch * float(launch_risk) +
        w_satellite * float(satellite_risk) +
        w_neo * float(neo_risk)
    )
    mission_risk = float(np.clip(mission_risk, 0.0, 1.0))
    return mission_risk


def risk_category(risk_value):
    """Classify risk into operational categories."""
    if risk_value < 0.2:
        return "LOW", "✅ Mission is GO"
    elif risk_value < 0.4:
        return "MODERATE", "⚠️ Proceed with caution"
    elif risk_value < 0.6:
        return "ELEVATED", "🔶 Additional review required"
    elif risk_value < 0.8:
        return "HIGH", "🔴 Mission hold recommended"
    else:
        return "CRITICAL", "🚨 Mission abort recommended"


def generate_risk_report(launch_risk, satellite_risk, neo_risk):
    """Generate a comprehensive mission risk report."""
    mission_risk = compute_mission_risk(launch_risk, satellite_risk, neo_risk)
    category, recommendation = risk_category(mission_risk)

    report = {
        "mission_risk_index": round(mission_risk, 4),
        "risk_category": category,
        "recommendation": recommendation,
        "components": {
            "launch_risk": round(float(launch_risk), 4),
            "satellite_risk": round(float(satellite_risk), 4),
            "neo_risk": round(float(neo_risk), 4)
        },
        "weights": {
            "launch": 0.4,
            "satellite": 0.35,
            "neo": 0.25
        }
    }

    print("\n" + "=" * 60)
    print("ORBITRON — Mission Risk Intelligence Report")
    print("=" * 60)
    print(f"  Launch Risk:        {launch_risk:.4f}")
    print(f"  Satellite Risk:     {satellite_risk:.4f}")
    print(f"  NEO Risk:           {neo_risk:.4f}")
    print(f"  ─────────────────────────────────")
    print(f"  Mission Risk Index: {mission_risk:.4f}")
    print(f"  Category:           {category}")
    print(f"  {recommendation}")
    print("=" * 60)

    return report
