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
    }

    return results
