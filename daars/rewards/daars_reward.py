import math

"""
Standard reward shaping for safe navigation uses *fixed* weights for the
progress and safety components.  DAARS replaces these with two distance-
dependent scaling functions that modulate the *dominant* terms:

    R_DAARS = α(d) · [r_prog + r_vel] + β(d) · r_safe + r_time

here d = d_obs is the minimum obstacle clearance.
"""

# first create scaling functions
#alpha
# the idea is: Reduce movement as obstacles get closer, but never reduce below 30%
def alpha(d_obs: float, d_safe: float) -> float:
    """Goal/progress modulator: ∈ [α_min, 1], linear in d_obs.

    α(d) = max(α_min, min(d / d_safe, 1))

    """
    ALPHA_MIN = 0.3          
    if d_obs >= d_safe:
        return 1.0
    
    return max(ALPHA_MIN, d_obs / d_safe)

#beta
# safety weight of 5.0, to prevent max safety weight i.e. cap it
def beta(d_obs: float, ks: float, epsilon: float) -> float:
    """Safety amplifier: > 0, monotonically decreasing in d_obs.

    β(d) = exp(k_s / (d + ε)) − exp(k_s / (d + ε + 2))

    """
    BETA_MAX = 5.0          
    raw = math.exp(ks / (d_obs + epsilon)) - math.exp(ks / (d_obs + epsilon + 2.0))
    return min(raw, BETA_MAX)


def daars_reward(info: dict, cfg: dict,
                 ks: float | None = None,
                 dsafe: float | None = None) -> float:
    """Compute one-step DAARS reward.

    Formula
    -------
        R = α(d) · (r_prog + r_vel) + β(d) · r_safe + r_time

    where:
        r_prog  = (d_prev − d_goal) · progress_scale   [potential shaping]
        r_vel   = 0.5 · max(v, 0)                      [forward-motion bonus]
        r_safe  = safety_penalty_scale · (1 − d/d_s)²  [quadratic proximity]
        r_time  = −0.1                                  [time penalty]
    """
    # Terminal signals take priority
    if info["reached_goal"]:
        return float(cfg["goal_reward"])
    if info["collision"]:
        return float(cfg["collision_penalty"])
    if info["timeout"]:
        return float(cfg["timeout_penalty"])

    ks_val    = float(ks    if ks    is not None else cfg["ks"])
    dsafe_val = float(dsafe if dsafe is not None else cfg["dsafe"])
    epsilon   = float(cfg["epsilon"])
    d_obs     = float(info["min_obs_dist"])

    a = alpha(d_obs, dsafe_val)
    b = beta(d_obs, ks_val, epsilon)

    ### Dense components
    r_prog = (float(info["prev_goal_dist"]) - float(info["goal_dist"])) \
             * float(cfg["progress_scale"])
    r_vel  = 0.5 * max(float(info["v_linear"]), 0.0)

    if d_obs < dsafe_val:
        proximity = 1.0 - d_obs / dsafe_val
        r_safe = float(cfg["safety_penalty_scale"]) * proximity * proximity
    else:
        r_safe = 0.0

    r_time = -0.1

    return a * (r_prog + r_vel) + b * r_safe + r_time

# cost signal for larangian baseline
def daars_cost(info: dict, cfg: dict) -> float:
    """Binary safety cost c_t ∈ {0, 1}.

    c_t = 1 if the robot is within d_safe of any obstacle, else 0.
    The Lagrangian baselines constrain E[Σ c_t / T] ≤ cost_limit.

    here, separating the cost from the reward allows Lagrangian methods to
    optimise task performance subject to an explicit safety constraint,
    which is the standard CPO / SAC-Lag formulation.
    """
    if info["collision"]:
        return 1.0
    d_obs = float(info["min_obs_dist"])
    dsafe = float(cfg["dsafe"])
    return 1.0 if d_obs < dsafe else 0.0
