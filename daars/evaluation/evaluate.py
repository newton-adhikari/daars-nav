"""
Evaluation of trained agents on fixed test scenarios.
"""

from __future__ import annotations

import numpy as np
from stable_baselines3 import PPO, SAC

from daars.envs import RobotNavEnv
from daars.rewards import REWARD_FUNCTIONS, COST_FUNCTIONS
from daars.evaluation.scenarios import generate_test_scenarios


def evaluate_agent(
    config: dict,
    model_path: str,
    reward_type: str,
    scenario: str = "simple",
    num_episodes: int = 200,
    test_seed: int = 42,
    daars_ks: float | None = None,
    daars_dsafe: float | None = None,
) -> dict:
    """Evaluate a trained agent on fixed test scenarios.

    Returns per-episode metrics and aggregate summary
    """
    # Load model (try PPO first, fall back to SAC)
    try:
        model = PPO.load(model_path)
    except Exception:
        model = SAC.load(model_path)

    reward_fn = REWARD_FUNCTIONS[reward_type]
    cost_fn   = COST_FUNCTIONS[reward_type]

    if reward_type == "daars" and (daars_ks is not None or daars_dsafe is not None):
        _base = reward_fn
        _ks, _ds = daars_ks, daars_dsafe
        reward_fn = lambda i, c: _base(i, c, ks=_ks, dsafe=_ds)

    # Fixed test scenarios (identical across all methods)
    test_scenarios = generate_test_scenarios(num_episodes, seed=test_seed)

    records: dict[str, list] = {
        "success":        [],
        "collision":      [],
        "timeout":        [],
        "path_length":    [],
        "start_goal_dist": [],
        "min_clearance":  [],
        "ep_reward":      [],
        "ep_cost":        [],
        "ep_steps":       [],
    }

    for sc in test_scenarios:
        env = RobotNavEnv(
            config,
            reward_fn=reward_fn,
            cost_fn=cost_fn,
            scenario=scenario,
            seed=sc["seed"],
            domain_rand=False,  # eval without noise for reproducibility
        )
        obs, info = env.reset(seed=sc["seed"])

        total_reward = 0.0
        done         = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        records["success"].append(bool(info.get("reached_goal", False)))
        records["collision"].append(bool(info.get("collision",   False)))
        records["timeout"].append(
            not info.get("reached_goal", False) and not info.get("collision", False))
        records["path_length"].append(float(info.get("path_length", 0.0)))

        records["start_goal_dist"].append(float(info.get("start_goal_dist", 0.0)))
        records["min_clearance"].append(float(info.get("min_clearance", float("inf"))))
        records["ep_reward"].append(total_reward)
        records["ep_cost"].append(float(info.get("cumulative_cost", 0.0)))
        records["ep_steps"].append(int(info.get("steps", 0)))

    # Aggregate metrics
    successes     = np.array(records["success"])
    collisions    = np.array(records["collision"])
    path_lengths  = np.array(records["path_length"])
    start_dists   = np.array(records["start_goal_dist"])
    clearances    = np.array(records["min_clearance"])
    ep_costs      = np.array(records["ep_cost"])
    ep_steps      = np.array(records["ep_steps"])

    # Path efficiency — only on successful episodes
    suc_mask   = successes.astype(bool)
    if suc_mask.any():
        path_eff = start_dists[suc_mask] / np.maximum(path_lengths[suc_mask], 1e-6)
        path_efficiency = float(np.mean(path_eff))
    else:
        path_efficiency = 0.0

    # Clearances (exclude inf for episodes with no obstacles)
    valid_clr = clearances[clearances < 1e9]
    avg_min_clearance = float(np.mean(valid_clr)) if len(valid_clr) > 0 else 0.0
    std_min_clearance = float(np.std(valid_clr))  if len(valid_clr) > 0 else 0.0

    # Constraint violation rate: fraction of steps in danger zone
    total_steps = float(np.sum(ep_steps))
    total_cost  = float(np.sum(ep_costs))
    constraint_violation_rate = total_cost / max(total_steps, 1)

    summary = {
        "success_rate":              float(np.mean(successes)),
        "collision_rate":            float(np.mean(collisions)),
        "timeout_rate":              float(np.mean(records["timeout"])),
        "path_efficiency":           path_efficiency,
        "avg_min_clearance":         avg_min_clearance,
        "std_min_clearance":         std_min_clearance,
        "constraint_violation_rate": constraint_violation_rate,
        "avg_reward":                float(np.mean(records["ep_reward"])),
        "avg_steps":                 float(np.mean(ep_steps)),
        "num_episodes":              len(records["success"]),
    }

    return {
        "summary":   summary,
        # Per-episode arrays for statistical tests
        "successes":  [float(v) for v in records["success"]],
        "collisions": [float(v) for v in records["collision"]],
        "min_clearances": [float(v) for v in records["min_clearance"]],
        "path_efficiencies": (
            [float(s / max(p, 1e-6))
             for s, p, ok in zip(records["start_goal_dist"],
                                 records["path_length"],
                                 records["success"]) if ok]
        ),
        "ep_costs":   [float(v) for v in records["ep_cost"]],
    }
