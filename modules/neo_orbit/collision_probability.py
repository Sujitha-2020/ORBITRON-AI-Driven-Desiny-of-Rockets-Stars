"""
NEO Orbit — Collision Probability via Monte Carlo Simulation
Simulates 10,000 orbital variations and computes Earth intersection probability.
"""

import numpy as np
from modules.neo_orbit.kepler_propagator import propagate_orbit, compute_moid


def monte_carlo_collision(a, e, i, om, w, ma, n_simulations=10000,
                          perturbation_scale=0.01, earth_radius_au=4.26e-5):
    """
    Monte Carlo collision probability estimation.
    Perturbs orbital elements and computes fraction of orbits
    that intersect Earth's sphere of influence.

    Parameters:
        a, e, i, om, w, ma: Orbital elements (a in AU, angles in degrees)
        n_simulations: Number of Monte Carlo samples
        perturbation_scale: Scale of Gaussian perturbation (fraction)
        earth_radius_au: Earth's radius in AU (~4.26e-5 AU)

    Returns:
        collision_probability: Float in [0, 1]
        moid_distribution: Array of MOIDs from simulations
    """
    print(f"\n[CollisionProb] Running Monte Carlo ({n_simulations} simulations)...")

    # Earth's Hill sphere radius in AU (broader collision check)
    collision_threshold = 0.01  # ~1.5 million km — gravitational capture zone

    moid_distribution = []
    collision_count = 0

    for sim in range(n_simulations):
        # Perturb orbital elements
        a_p = a * (1 + np.random.normal(0, perturbation_scale))
        e_p = np.clip(e + np.random.normal(0, perturbation_scale * 0.1), 0.001, 0.999)
        i_p = i + np.random.normal(0, perturbation_scale * 5)
        om_p = om + np.random.normal(0, perturbation_scale * 5)
        w_p = w + np.random.normal(0, perturbation_scale * 5)
        ma_p = ma + np.random.normal(0, perturbation_scale * 10)

        # Propagate perturbed orbit (use fewer steps for speed)
        try:
            positions = propagate_orbit(a_p, e_p, i_p, om_p, w_p, ma_p, n_steps=72)
            moid = compute_moid(positions, n_earth_steps=72)
            moid_distribution.append(moid)

            if moid < collision_threshold:
                collision_count += 1
        except Exception:
            continue

    collision_probability = collision_count / max(len(moid_distribution), 1)
    moid_distribution = np.array(moid_distribution)

    print(f"  Simulations completed: {len(moid_distribution)}")
    print(f"  Collision probability: {collision_probability:.6f}")
    print(f"  MOID — Mean: {moid_distribution.mean():.6f}, Min: {moid_distribution.min():.6f}")

    return collision_probability, moid_distribution


def compute_all_collision_probabilities(neo_results, n_simulations=10000):
    """Compute collision probabilities for all propagated NEOs."""
    print("\n[CollisionProb] Computing collision probabilities for all NEOs...")
    results = []

    for neo in neo_results:
        name = neo["name"]
        a, e, i = neo["a"], neo["e"], neo["i"]
        # Use default values for missing elements
        om = neo.get("om", 0.0)
        w = neo.get("w", 0.0)
        ma = neo.get("ma", 0.0)

        prob, moid_dist = monte_carlo_collision(
            a, e, i, om, w, ma, n_simulations=n_simulations
        )

        results.append({
            "name": name,
            "collision_probability": prob,
            "moid_mean": float(moid_dist.mean()),
            "moid_min": float(moid_dist.min()),
            "moid_distribution": moid_dist
        })

    return results
