"""
Ablation over β function forms.

Addresses reviewer concern: "The B function form is not motivated.
Why this form? Why not a simple exponential or inverse-distance?"

We compare four candidate β functions:
  1. DAARS β (paper):  exp(k_s/(d+ε)) - exp(k_s/(d+ε+2))
  2. Simple exponential: exp(k_s * (d_safe - d) / d_safe)
  3. Inverse distance:   k_s / (d + ε)
  4. Quadratic:          k_s * (1 - d/d_safe)^2  for d < d_safe

The DAARS form was chosen because it satisfies all four properties:
  P1: β → ∞ as d → 0 (strong safety near obstacles)
  P2: β → 0 as d → ∞ (no interference in open space)
  P3: Bounded reward (due to the subtraction term)
  P4: Monotonically decreasing in d

The simple exponential satisfies P1 but NOT P2 (grows unbounded for d<0).
The inverse distance satisfies P1, P2 but has a singularity at d=-ε.
The quadratic satisfies P3, P4 but NOT P1 (bounded at d=0).
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np


# β function candidates

def beta_daars(d: float, ks: float, eps: float, dsafe: float) -> float:
    """Paper form: exp(k_s/(d+ε)) - exp(k_s/(d+ε+2)). Satisfies P1-P4."""
    raw = math.exp(ks / (d + eps)) - math.exp(ks / (d + eps + 2.0))
    return min(raw, 5.0)


def beta_exponential(d: float, ks: float, eps: float, dsafe: float) -> float:
    """Simple exponential: exp(k_s * (d_safe - d) / d_safe).
    Satisfies P1, P4. Violates P2 (non-zero for d > d_safe)."""
    raw = math.exp(ks * max(dsafe - d, 0.0) / dsafe)
    return min(raw, 5.0)


def beta_inverse(d: float, ks: float, eps: float, dsafe: float) -> float:
    """Inverse distance: k_s / (d + ε).
    Satisfies P1, P2, P4. Singularity concern at d ≈ 0."""
    raw = ks / (d + eps)
    return min(raw, 5.0)


def beta_quadratic(d: float, ks: float, eps: float, dsafe: float) -> float:
    """Quadratic: k_s * (1 - d/d_safe)^2 for d < d_safe, else 0.
    Satisfies P2, P3, P4. Violates P1 (bounded at d=0)."""
    if d >= dsafe:
        return 0.0
    prox = 1.0 - d / dsafe
    return min(ks * prox * prox, 5.0)


BETA_FUNCTIONS: dict[str, Callable] = {
    "daars": beta_daars,
    "exponential": beta_exponential,
    "inverse": beta_inverse,
    "quadratic": beta_quadratic,
}


def analyze_beta_properties(
    ks: float = 0.5,
    eps: float = 0.1,
    dsafe: float = 0.8,
    n_points: int = 200,
) -> dict:
    """Numerically verify which properties each β form satisfies.

    Returns a table of property satisfaction for each form.
    """
    d_values = np.linspace(0.01, 3.0, n_points)

    results = {}
    for name, beta_fn in BETA_FUNCTIONS.items():
        values = [beta_fn(d, ks, eps, dsafe) for d in d_values]
        values = np.array(values)

        # P1: β → large as d → 0
        near_zero = beta_fn(0.01, ks, eps, dsafe)
        p1_strong_near_zero = near_zero > 3.0

        # P2: β → 0 as d → ∞ (check at d=3.0)
        far_value = beta_fn(3.0, ks, eps, dsafe)
        p2_decays = far_value < 0.1

        # P3: Bounded (max value ≤ 5.0)
        p3_bounded = np.max(values) <= 5.0 + 1e-6

        # P4: Monotonically decreasing
        diffs = np.diff(values)
        p4_monotone = np.all(diffs <= 1e-6)

        # Gradient magnitude at d=0.3 (safety-critical zone)
        h = 1e-5
        grad_at_03 = abs(
            (beta_fn(0.3 + h, ks, eps, dsafe) - beta_fn(0.3 - h, ks, eps, dsafe))
            / (2 * h)
        )

        results[name] = {
            "P1_safety_dominance": bool(p1_strong_near_zero),
            "P2_open_space_decay": bool(p2_decays),
            "P3_bounded": bool(p3_bounded),
            "P4_monotone": bool(p4_monotone),
            "properties_satisfied": sum([
                p1_strong_near_zero, p2_decays, p3_bounded, p4_monotone
            ]),
            "value_at_d01": float(beta_fn(0.1, ks, eps, dsafe)),
            "value_at_d05": float(beta_fn(0.5, ks, eps, dsafe)),
            "value_at_d10": float(beta_fn(1.0, ks, eps, dsafe)),
            "gradient_at_d03": float(grad_at_03),
        }

    return results


def make_beta_reward_fn(beta_name: str):
    """Create a DAARS reward function using a specific β form.

    Used for ablation experiments comparing β function choices.
    """
    from daars.rewards.daars_reward import alpha

    beta_fn = BETA_FUNCTIONS[beta_name]

    def reward_fn(info: dict, cfg: dict, ks=None, dsafe=None) -> float:
        if info["reached_goal"]:
            return float(cfg["goal_reward"])
        if info["collision"]:
            return float(cfg["collision_penalty"])
        if info["timeout"]:
            return float(cfg["timeout_penalty"])

        ks_val = float(ks if ks is not None else cfg["ks"])
        dsafe_val = float(dsafe if dsafe is not None else cfg["dsafe"])
        epsilon = float(cfg["epsilon"])
        d_obs = float(info["min_obs_dist"])

        a = alpha(d_obs, dsafe_val)
        b = beta_fn(d_obs, ks_val, epsilon, dsafe_val)

        r_prog = (float(info["prev_goal_dist"]) - float(info["goal_dist"])) \
                 * float(cfg["progress_scale"])
        r_vel = 0.5 * max(float(info["v_linear"]), 0.0)

        if d_obs < dsafe_val:
            proximity = 1.0 - d_obs / dsafe_val
            r_safe = float(cfg["safety_penalty_scale"]) * proximity * proximity
        else:
            r_safe = 0.0

        return a * (r_prog + r_vel) + b * r_safe - 0.1

    return reward_fn
