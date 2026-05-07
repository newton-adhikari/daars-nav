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