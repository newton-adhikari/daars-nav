"""
Baseline reward functions for comparison with DAARS.
"""

from __future__ import annotations
import math


# 1. Static baseline
def static_reward(info: dict, cfg: dict) -> float:
    """
    Fixed-weight reward function (upper bound on non-adaptive shaping).

    R_static = w_g · (r_prog + r_vel) + w_s · r_safe + r_time

    where w_g = static_goal_weight, w_s = static_safety_weight are
    constants throughout training.  This is the strongest non-adaptive
    baseline: if DAARS does not outperform this, the adaptive mechanism
    is useless[well technically not but our core idea will not be validated].
    """
    if info["reached_goal"]:
        return float(cfg["goal_reward"])
    if info["collision"]:
        return float(cfg["collision_penalty"])
    if info["timeout"]:
        return float(cfg["timeout_penalty"])

    wg = float(cfg["static_goal_weight"])
    ws = float(cfg["static_safety_weight"])

    r_prog = (float(info["prev_goal_dist"]) - float(info["goal_dist"])) \
             * float(cfg["progress_scale"])
    r_vel  = 0.5 * max(float(info["v_linear"]), 0.0)

    d_obs  = float(info["min_obs_dist"])
    dsafe  = float(cfg["dsafe"])
    if d_obs < dsafe:
        proximity = 1.0 - d_obs / dsafe
        r_safe = float(cfg["safety_penalty_scale"]) * proximity * proximity
    else:
        r_safe = 0.0

    return wg * (r_prog + r_vel) + ws * r_safe - 0.1

# 2. Task-only reward (used with Lagrangian agents)
def task_reward(info: dict, cfg: dict) -> float:
    """
    It is mainly, task reward without safety shaping.

    Used with PPO-Lagrangian and SAC-Lagrangian, where safety is enforced
    via an explicit cost constraint rather than reward shaping.  Providing
    the Lagrangian baselines with the same task reward as static.
    
    """
    if info["reached_goal"]:
        return float(cfg["goal_reward"])
    if info["collision"]:
        return float(cfg["collision_penalty"])
    if info["timeout"]:
        return float(cfg["timeout_penalty"])

    r_prog = (float(info["prev_goal_dist"]) - float(info["goal_dist"])) \
             * float(cfg["progress_scale"])
    r_vel  = 0.5 * max(float(info["v_linear"]), 0.0)
    return r_prog + r_vel - 0.1

# 3. Ablation: α-only (progress suppression, static safety weight)

# 4. Ablation: β-only (safety amplification, static progress weight)
