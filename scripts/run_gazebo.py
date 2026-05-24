#!/usr/bin/env python3
"""
Run a single trained model in Gazebo and collect episode metrics.

Usage:
    python3 scripts/run_gazebo.py \
        --model results/models/daars_seed0 \
        --method daars \
        --episodes 20 \
        --output results/gazebo/daars_seed0.json
"""

import os
import sys
import json
import argparse
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import yaml
from stable_baselines3 import PPO, SAC
from daars.rewards import REWARD_FUNCTIONS, COST_FUNCTIONS


def load_config(path=None):
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "..", "daars", "config", "default.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


def ts():
    return time.strftime("%H:%M:%S")


def main():
    parser = argparse.ArgumentParser(description="Run one model in Gazebo")
    parser.add_argument("--model", required=True, help="Path to .zip model (without .zip)")
    parser.add_argument("--method", required=True,
                        choices=list(REWARD_FUNCTIONS.keys()),
                        help="Method name (for reward function selection)")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    reward_fn = REWARD_FUNCTIONS[args.method]
    cost_fn = COST_FUNCTIONS[args.method]

    from daars.gazebo.gazebo_env import GazeboNavEnv

    print(f"[{ts()}] Creating env (fast_mode=True)")
    env = GazeboNavEnv(
        config,
        reward_fn=reward_fn,
        cost_fn=cost_fn,
        fast_mode=True,
    )

    print(f"[{ts()}] Loading model: {args.model}")
    custom_objects = {
        "observation_space": env.observation_space,
        "action_space": env.action_space,
    }
    try:
        model = PPO.load(args.model, custom_objects=custom_objects)
    except Exception:
        model = SAC.load(args.model, custom_objects=custom_objects)

    print(f"[{ts()}] Running {args.episodes} episodes...")

    records = []
    t_start = time.time()

    for ep in range(args.episodes):
        ep_start = time.time()
        obs, info = env.reset()
        done = False
        ep_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated

        ep_time = time.time() - ep_start

        rec = {
            "episode": ep,
            "success": bool(info.get("reached_goal", False)),
            "collision": bool(info.get("collision", False)),
            "timeout": not info.get("reached_goal", False) and not info.get("collision", False),
            "path_length": float(info.get("path_length", 0.0)),
            "min_clearance": float(info.get("min_clearance", 0.0)),
            "start_goal_dist": float(info.get("start_goal_dist", 0.0)),
            "steps": int(info.get("steps", 0)),
            "reward": float(ep_reward),
            "cumulative_cost": float(info.get("cumulative_cost", 0.0)),
            "wall_time_s": round(ep_time, 2),
        }
        records.append(rec)

        # Running stats
        sr = np.mean([r["success"] for r in records])
        cr = np.mean([r["collision"] for r in records])
        elapsed = time.time() - t_start
        eta = (elapsed / (ep + 1)) * (args.episodes - ep - 1)

        status = "OK" if rec["success"] else ("COL" if rec["collision"] else "T/O")
        print(f"[{ts()}] ep {ep+1:3d}/{args.episodes} {status} "
              f"clr={rec['min_clearance']:.3f}m "
              f"steps={rec['steps']:3d} "
              f"t={ep_time:.1f}s "
              f"| SR={sr:.2f} CR={cr:.2f} ETA={eta:.0f}s")

    env.close()

    # Shutdown ROS (single-env script)
    try:
        import rclpy
        rclpy.shutdown()
    except Exception:
        pass

    # Summary
    total_time = time.time() - t_start
    sr = np.mean([r["success"] for r in records])
    cr = np.mean([r["collision"] for r in records])
    mc = np.mean([r["min_clearance"] for r in records])
    succ_clr = [r["min_clearance"] for r in records if r["success"]]
    avg_succ_clr = np.mean(succ_clr) if succ_clr else 0.0

    summary = {
        "method": args.method,
        "model": args.model,
        "num_episodes": args.episodes,
        "success_rate": float(sr),
        "collision_rate": float(cr),
        "timeout_rate": float(np.mean([r["timeout"] for r in records])),
        "avg_min_clearance": float(mc),
        "avg_clearance_successful": float(avg_succ_clr),
        "avg_path_length": float(np.mean([r["path_length"] for r in records if r["success"]])) if succ_clr else 0.0,
        "avg_steps": float(np.mean([r["steps"] for r in records])),
        "total_wall_time_s": round(total_time, 1),
        "avg_time_per_episode_s": round(total_time / args.episodes, 1),
    }

    print(f"\n[{ts()}] DONE in {total_time:.0f}s ({total_time/args.episodes:.1f}s/ep)")
    print(f"  SR={sr:.3f}  CR={cr:.3f}  Clearance={mc:.3f}m  SuccClearance={avg_succ_clr:.3f}m")

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    output_data = {"summary": summary, "episodes": records}
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"  Saved → {args.output}")


if __name__ == "__main__":
    main()
