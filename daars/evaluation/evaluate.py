"""
Evaluation of trained agents on fixed test scenarios.
"""
from stable_baselines3 import PPO, SAC


def evaluate_agent(
    config: dict,
    model_path: str
) -> dict:
    """Evaluate a trained agent on fixed test scenarios.
    """
    # Load model (try PPO first, fall back to SAC)
    try:
        model = PPO.load(model_path)
    except Exception:
        model = SAC.load(model_path)

    