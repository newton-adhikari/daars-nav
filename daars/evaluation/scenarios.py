"""
NOTE: This is fixed test scenario generation 
for reproducible evaluation.
"""

import numpy as np


def generate_test_scenarios(num_episodes: int = 100, seed: int = 42) -> list[dict]:
    """Returns list of dicts with 'seed' for each episode, ensuring
    all methods are evaluated on identical obstacle configurations.
    """
    rng = np.random.default_rng(seed)
    scenarios = []
    for i in range(num_episodes):
        scenarios.append({
            "episode_id": i,
            "seed": int(rng.integers(0, 2**31)),
        })
    return scenarios
