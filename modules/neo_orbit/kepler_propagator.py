"""
NEO Orbit — Keplerian Orbit Propagator
Solves Kepler's equation using Newton-Raphson iteration.
Computes future orbital positions and MOID (Minimum Orbit Intersection Distance).
"""

import numpy as np


def solve_kepler(M, e, tol=1e-10, max_iter=100):
    """
    Solve Kepler's equation M = E - e * sin(E) using Newton-Raphson.
    M: Mean anomaly (radians)
    e: Eccentricity
    Returns: Eccentric anomaly E (radians)
    """
    E = M.copy() if isinstance(M, np.ndarray) else np.array([M])
    for _ in range(max_iter):
        dE = (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
        E -= dE
        if np.all(np.abs(dE) < tol):
            break
    return E


def eccentric_to_true_anomaly(E, e):
    """Convert eccentric anomaly to true anomaly."""
    nu = 2 * np.arctan2(
        np.sqrt(1 + e) * np.sin(E / 2),
        np.sqrt(1 - e) * np.cos(E / 2)
    )
    return nu


def orbital_position(a, e, i, om, w, nu):
    """
    Compute 3D position from orbital elements.
    a: semi-major axis, e: eccentricity, i: inclination,
    om: longitude of ascending node, w: argument of perihelion,
    nu: true anomaly. All angles in radians.
    """
    # Radius
    r = a * (1 - e**2) / (1 + e * np.cos(nu))

    # Position in orbital plane
    x_orb = r * np.cos(nu)
    y_orb = r * np.sin(nu)

    # Rotation to 3D (Perifocal → Inertial)
    cos_om, sin_om = np.cos(om), np.sin(om)
    cos_w, sin_w = np.cos(w), np.sin(w)
    cos_i, sin_i = np.cos(i), np.sin(i)

    x = (cos_om * cos_w - sin_om * sin_w * cos_i) * x_orb + \
        (-cos_om * sin_w - sin_om * cos_w * cos_i) * y_orb
    y = (sin_om * cos_w + cos_om * sin_w * cos_i) * x_orb + \
        (-sin_om * sin_w + cos_om * cos_w * cos_i) * y_orb
    z = (sin_w * sin_i) * x_orb + (cos_w * sin_i) * y_orb

    return np.column_stack([x, y, z]) if isinstance(x, np.ndarray) else np.array([x, y, z])


def propagate_orbit(a, e, i_deg, om_deg, w_deg, ma_deg, n_steps=360):
    """
    Propagate a full orbit over n_steps mean anomaly steps.
    Returns array of (n_steps, 3) positions.
    """
    i = np.radians(i_deg)
    om = np.radians(om_deg)
    w = np.radians(w_deg)

    M_values = np.linspace(0, 2 * np.pi, n_steps)
    positions = []

    for M in M_values:
        E = solve_kepler(np.array([M]), e)[0]
        nu = eccentric_to_true_anomaly(E, e)
        pos = orbital_position(a, e, i, om, w, nu)
        positions.append(pos.flatten())

    return np.array(positions)


def compute_moid(positions_neo, a_earth=1.0, n_earth_steps=360):
    """
    Compute approximate Minimum Orbit Intersection Distance (MOID).
    Compares NEO orbit positions to Earth's approximate circular orbit.
    """
    # Earth approximate positions (circular orbit in ecliptic plane)
    theta_earth = np.linspace(0, 2 * np.pi, n_earth_steps)
    earth_pos = np.column_stack([
        a_earth * np.cos(theta_earth),
        a_earth * np.sin(theta_earth),
        np.zeros(n_earth_steps)
    ])

    # Compute minimum distance
    min_dist = np.inf
    for p_neo in positions_neo:
        dists = np.linalg.norm(earth_pos - p_neo, axis=1)
        min_dist = min(min_dist, dists.min())

    return min_dist


def propagate_all_neos(df_orb):
    """Propagate orbits for all NEOs in the dataframe."""
    print("\n[NEOOrbit] Propagating Keplerian orbits...")
    results = []

    for idx, row in df_orb.iterrows():
        a = row.get("a", 1.5)
        e = row.get("e", 0.3)
        i = row.get("i", 10.0)
        om = row.get("om", 0.0)
        w = row.get("w", 0.0)
        ma = row.get("ma", 0.0)

        positions = propagate_orbit(a, e, i, om, w, ma)
        moid = compute_moid(positions)

        name = row.get("full_name", f"NEO_{idx}")
        print(f"  {name}: MOID = {moid:.6f} AU")

        results.append({
            "name": name,
            "positions": positions,
            "moid": moid,
            "a": a, "e": e, "i": i
        })

    return results
