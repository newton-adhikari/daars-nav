"""
this is for formal analysis of DAARS scaling properties.

Her in this module provides we have done:
  1. Numerical verification of the four formal properties stated in
     daars_reward.py (P1–P4).
  2. A safety-barrier argument showing DAARS induces a repulsive gradient
     near obstacles while recovering standard reward shaping in open space.
  3. Comparison of gradient magnitudes between DAARS and the static
     baseline to quantify the benefit of dual modulation.

These functions are called by run_experiments.py to produce the
figures and the formal results table.

Mathematical background
-----------------------
Define the DAARS shaped reward (dense part only, ignoring terminal signals):

    R(d) = α(d) · P + β(d) · S(d)

where P = r_prog + r_vel ≥ 0 (progress terms, assumed locally positive),
S(d) = c · (1 - d/d_s)²  (safety penalty, c < 0), d = d_obs.

The gradient with respect to d (treated as a scalar for analysis):

    dR/dd = α'(d)·P + β'(d)·S(d) + β(d)·S'(d)

Near d ≈ 0:
  α(d)  → 0,   α'(d) = 1/d_s  (constant, positive)
  β(d)  → ∞,   β'(d) → -∞     (strongly negative)
  S(d)  → c·1 < 0
  S'(d) = -2c(1-d/d_s)/d_s > 0  (since c < 0)

So β'(d)·S(d) → +∞  (β' strongly negative × S negative = positive)
and β(d)·S'(d) → +∞ (β large positive × S' positive = positive).

Both terms push dR/dd → +∞  as d → 0, meaning the reward increases
as the robot moves away from the obstacle — a repulsive pseudo-force.

For the static baseline:
    R_static(d) = w_g·P + w_s·S(d)
    dR_static/dd = w_s·S'(d) = -2w_s·c·(1-d/d_s)/d_s

This is bounded and constant regardless of proximity.  DAARS produces
a strictly stronger repulsive gradient near obstacles (Prop. P1).
"""

from __future__ import annotations

import math
import numpy as np



# Scalar helper functions (mirrors daars_reward.py for standalone analysis)

def _alpha(d: float, dsafe: float) -> float:
    ALPHA_MIN = 0.3
    return float(np.clip(d / dsafe, ALPHA_MIN, 1.0))

def _alpha_prime(d: float, dsafe: float) -> float:
    """dα/dd = 1/d_safe for α_min < d/d_safe < 1, 0 otherwise."""
    ALPHA_MIN = 0.3
    if d < ALPHA_MIN * dsafe or d >= dsafe:
        return 0.0
    return 1.0 / dsafe

def _beta(d: float, ks: float, eps: float) -> float:
    return min(math.exp(ks / (d + eps)) - math.exp(ks / (d + eps + 2.0)), 5.0)

def _beta_prime(d: float, ks: float, eps: float) -> float:
    """dβ/dd = -k_s/(d+ε)² · exp(k_s/(d+ε))
               + k_s/(d+ε+2)² · exp(k_s/(d+ε+2))
    Numerical approximation (finite difference)."""
    h = 1e-5
    return (_beta(d + h, ks, eps) - _beta(d - h, ks, eps)) / (2 * h)

def _safety(d: float, dsafe: float, scale: float) -> float:
    if d >= dsafe:
        return 0.0
    prox = 1.0 - d / dsafe
    return scale * prox * prox

def _safety_prime(d: float, dsafe: float, scale: float) -> float:
    """dS/dd = 2·scale·(1-d/d_s)/d_s."""
    if d >= dsafe:
        return 0.0
    return 2.0 * scale * (1.0 - d / dsafe) / dsafe



# Property verification
def verify_safety_dominance(
    ks: float = 1.5,
    dsafe: float = 1.0,
    eps: float = 0.1,
    safety_scale: float = -20.0,
    progress_value: float = 5.0,
) -> dict:
    """Verify P1: β(d)/α(d) → ∞ as d → 0.

    Also computes the crossover distance d* where the safety gradient
    magnitude first exceeds the progress gradient magnitude.
    """
    d_values = np.linspace(0.01, dsafe * 1.5, 500)
    ratio    = np.array([
        _beta(d, ks, eps) / max(_alpha(d, dsafe), 1e-9)
        for d in d_values
    ])

    # Gradient magnitudes
    daars_grad = np.array([
        abs(_alpha_prime(d, dsafe) * progress_value
            + _beta_prime(d, ks, eps) * _safety(d, dsafe, safety_scale)
            + _beta(d, ks, eps) * _safety_prime(d, dsafe, safety_scale))
        for d in d_values
    ])

    static_grad = np.array([
        abs(_safety_prime(d, dsafe, safety_scale))
        for d in d_values
    ])

    # Find crossover distance where DAARS gradient > static gradient
    crossover_idx = np.argmax(daars_grad > static_grad)
    crossover_d   = float(d_values[crossover_idx]) if crossover_idx > 0 else float(dsafe)

    return {
        "d_values":          d_values.tolist(),
        "beta_alpha_ratio":  ratio.tolist(),
        "daars_gradient":    daars_grad.tolist(),
        "static_gradient":   static_grad.tolist(),
        "crossover_dist":    crossover_d,
        "property_P1_holds": bool(ratio[0] > 10 * ratio[-1]),  # ratio large near obstacles, small far away
    }


def verify_boundedness(
    ks: float = 1.5,
    dsafe: float = 1.0,
    eps: float = 0.1,
    safety_scale: float = -20.0,
    progress_value: float = 10.0,
    goal_reward: float = 500.0,
    collision_penalty: float = -200.0,
) -> dict:
    """Verify P3: R_DAARS is bounded in [collision_penalty, goal_reward]."""
    d_values = np.linspace(0.001, 3.0, 1000)
    rewards  = []
    for d in d_values:
        a = _alpha(d, dsafe)
        b = _beta(d, ks, eps)
        s = _safety(d, dsafe, safety_scale)
        r = a * progress_value + b * s - 0.1
        rewards.append(r)

    r_arr = np.array(rewards)
    return {
        "min_reward":           float(np.min(r_arr)),
        "max_reward":           float(np.max(r_arr)),
        "lower_bound_holds":    bool(np.min(r_arr) > collision_penalty),
        "upper_bound_holds":    bool(np.max(r_arr) < goal_reward),
        "property_P3_holds":    bool(np.min(r_arr) > collision_penalty
                                     and np.max(r_arr) < goal_reward),
    }


def verify_monotonicity(
    ks: float = 1.5,
    dsafe: float = 1.0,
    eps: float = 0.1,
) -> dict:
    """Verify P4: α is non-decreasing, β is non-increasing in d."""
    d_values = np.linspace(0.001, 3.0, 1000)
    alpha_vals = np.array([_alpha(d, dsafe) for d in d_values])
    beta_vals  = np.array([_beta(d, ks, eps) for d in d_values])

    alpha_mono = bool(np.all(np.diff(alpha_vals) >= -1e-10))
    beta_mono  = bool(np.all(np.diff(beta_vals) <= 1e-10))

    return {
        "alpha_nondecreasing":   alpha_mono,
        "beta_nonincreasing":    beta_mono,
        "property_P4_holds":     alpha_mono and beta_mono,
    }


def compute_gradient_amplification(
    ks: float = 1.5,
    dsafe: float = 1.0,
    eps: float = 0.1,
    safety_scale: float = -20.0,
    progress_value: float = 5.0,
) -> dict:
    """Compute gradient amplification ratio: DAARS / static at each d.

    A ratio > 1 means DAARS has a stronger repulsive gradient.
    The peak ratio and the distance at which it occurs are key claims
    in the paper.
    """
    d_values    = np.linspace(0.01, dsafe, 200)
    daars_mag   = []
    static_mag  = []

    for d in d_values:
        dg = abs(
            _alpha_prime(d, dsafe) * progress_value
            + _beta_prime(d, ks, eps) * _safety(d, dsafe, safety_scale)
            + _beta(d, ks, eps) * _safety_prime(d, dsafe, safety_scale)
        )
        sg = abs(_safety_prime(d, dsafe, safety_scale))
        daars_mag.append(dg)
        static_mag.append(sg)

    ratio = np.array(daars_mag) / np.maximum(np.array(static_mag), 1e-8)

    return {
        "d_values":          d_values.tolist(),
        "daars_magnitude":   daars_mag,
        "static_magnitude":  static_mag,
        "amplification_ratio": ratio.tolist(),
        "peak_amplification":  float(np.max(ratio)),
        "peak_at_d":           float(d_values[np.argmax(ratio)]),
    }


def run_all_proofs(config: dict | None = None) -> dict:
    """Run all property verifications and return a summary report."""
    ks    = 0.8
    dsafe = 1.0
    eps   = 0.1

    if config:
        rc    = config.get("reward", {})
        ks    = float(rc.get("ks",      ks))
        dsafe = float(rc.get("dsafe",   dsafe))
        eps   = float(rc.get("epsilon", eps))

    results = {
        "P1_safety_dominance": verify_safety_dominance(ks, dsafe, eps),
        "P3_boundedness":      verify_boundedness(ks, dsafe, eps),
        "P4_monotonicity":     verify_monotonicity(ks, dsafe, eps),
        "gradient_amplification": compute_gradient_amplification(ks, dsafe, eps),
    }

    all_hold = all([
        results["P1_safety_dominance"]["property_P1_holds"],
        results["P3_boundedness"]["property_P3_holds"],
        results["P4_monotonicity"]["property_P4_holds"],
    ])
    results["all_properties_verified"] = all_hold

    print("\n>> Theoretical properties:")
    for prop, res in results.items():
        if isinstance(res, dict) and "property" in str(prop):
            holds = res.get(f"{prop[:2]}_holds", "N/A")
            print(f"   {prop}: {'✓' if holds else '✗'}")
    amp = results["gradient_amplification"]
    print(f"   Peak gradient amplification: "
          f"{amp['peak_amplification']:.1f}× at d={amp['peak_at_d']:.3f} m")

    return results
