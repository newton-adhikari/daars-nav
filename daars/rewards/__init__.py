from .daars_reward import daars_reward, daars_cost
from .baselines import static_reward, task_reward, alpha_only_reward, beta_only_reward

# rewards used by train.py and evaluate.py
REWARD_FUNCTIONS = {
    "static":      static_reward,
    "daars":       daars_reward,
    "alpha_only":  alpha_only_reward,
    "beta_only":   beta_only_reward,
    "ppo_lag":     task_reward,
    "sac_lag":     task_reward,
    "daars_lag":   daars_reward,
    "focops":      task_reward,  # FOCOPS uses task reward + constraint projection
}

COST_FUNCTIONS = {
    name: daars_cost for name in REWARD_FUNCTIONS
}

__all__ = [
    "REWARD_FUNCTIONS", "COST_FUNCTIONS",
    "daars_reward", "daars_cost", "directional_daars_reward",
    "learned_daars_reward",
    "static_reward", "task_reward",
    "alpha_only_reward", "beta_only_reward",
]
 