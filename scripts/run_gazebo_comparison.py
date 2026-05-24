#!/usr/bin/env python3
"""
Gazebo TurtleBot3 comparison — single-session multi-method evaluation.

Alternative to run_all_gazebo.sh for when you want to run everything
from Python (e.g., on a machine where bash scripting is awkward).

Assumes Gazebo is already running with the robot spawned:
    ros2 launch daars_gazebo daars_world.launch.py

Usage:
    python3 scripts/run_gazebo_comparison.py \
        --model-dir results/models \
        --seeds 0 1 2 3 4 \
        --episodes 20 \
        --output-dir results/gazebo
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
from scipy import stats

from daars.rewards import REWARD_FUNCTIONS, COST_FUNCTIONS


METHODS = ["daars", "static", "alpha_only", "beta_only", "ppo_lag", "focops"]


def load_config(path=None):
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "..", "daars", "config", "default.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


def ts():
    return time.strftime("%H:%M:%S")


def run_episodes(env, model, episodes: int) -> list:
    """Run N episodes and return records."""
    records = []
    for ep in range(episodes):
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

        status = "✓" if rec["success"] else ("✗" if rec["collision"] else "T")
        sr = np.mean([r["success"] for r in records])
        print(f"      [{status}] ep {ep+1:2d}/{episodes}  "
              f"clr={rec['min_clearance']:.3f}m  "
              f"steps={rec['steps']:3d}  "
              f"t={ep_time:.1f}s  SR={sr:.2f}")

    return records


def main():
    parser = argparse.ArgumentParser(description="Gazebo multi-method comparison")
    parser.add_argument("--model-dir", default="results/models")
    parser.add_argument("--config", default=None)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--methods", nargs="+", default=METHODS)
    parser.add_argument("--output-dir", default="results/gazebo")
    args = parser.parse_args()

    config = load_config(args.config)
    os.makedirs(args.output_dir, exist_ok=True)

    from daars.gazebo.gazebo_env import GazeboNavEnv

    total_jobs = len(args.methods) * len(args.seeds)
    done = 0
    skipped = 0
    t_global = time.time()

    print(f"\n[{ts()}] Gazebo Comparison: {len(args.methods)} methods × "
          f"{len(args.seeds)} seeds × {args.episodes} eps = "
          f"{total_jobs * args.episodes} total episodes")
    print()

    for method in args.methods:
        if method not in REWARD_FUNCTIONS:
            print(f"[{ts()}] WARNING: Unknown method '{method}', skipping")
            continue

        for seed in args.seeds:
            done += 1
            model_path = os.path.join(args.model_dir, f"{method}_seed{seed}")
            output_path = os.path.join(args.output_dir, f"{method}_seed{seed}.json")

            if not os.path.exists(model_path + ".zip"):
                print(f"[{ts()}] [{done}/{total_jobs}] SKIP {method}/seed{seed} — no model")
                skipped += 1
                continue

            # Check if already done
            if os.path.exists(output_path):
                try:
                    with open(output_path) as f:
                        existing = json.load(f)
                    eps = existing.get("episodes", existing) if isinstance(existing, dict) else existing
                    if len(eps) >= args.episodes:
                        print(f"[{ts()}] [{done}/{total_jobs}] SKIP {method}/seed{seed} — already done")
                        skipped += 1
                        continue
                except Exception:
                    pass

            print(f"[{ts()}] [{done}/{total_jobs}] {method}/seed{seed} ({args.episodes} eps)")

            # Create env for this method
            reward_fn = REWARD_FUNCTIONS[method]
            cost_fn = COST_FUNCTIONS[method]

            env = GazeboNavEnv(
                config,
                reward_fn=reward_fn,
                cost_fn=cost_fn,
                fast_mode=True,
            )

            # Load model
            custom_objects = {
                "observation_space": env.observation_space,
                "action_space": env.action_space,
            }
            try:
                model = PPO.load(model_path, custom_objects=custom_objects)
            except Exception:
                model = SAC.load(model_path, custom_objects=custom_objects)

            # Run
            job_start = time.time()
            records = run_episodes(env, model, args.episodes)
            job_time = time.time() - job_start

            env.close()

            # Save
            sr = np.mean([r["success"] for r in records])
            cr = np.mean([r["collision"] for r in records])
            mc = np.mean([r["min_clearance"] for r in records])

            summary = {
                "method": method,
                "seed": seed,
                "success_rate": float(sr),
                "collision_rate": float(cr),
                "avg_min_clearance": float(mc),
                "num_episodes": len(records),
                "wall_time_s": round(job_time, 1),
            }

            with open(output_path, "w") as f:
                json.dump({"summary": summary, "episodes": records}, f, indent=2)

            elapsed = time.time() - t_global
            jobs_run = done - skipped
            eta = (elapsed / max(jobs_run, 1)) * (total_jobs - done) if jobs_run > 0 else 0

            print(f"    → SR={sr:.2f} CR={cr:.2f} Clr={mc:.3f}m "
                  f"({job_time:.0f}s) ETA={eta:.0f}s")
            print()

    # Final summary
    total_time = time.time() - t_global
    print(f"\n[{ts()}] {'=' * 60}")
    print(f"[{ts()}] COMPLETE in {total_time:.0f}s ({total_time/60:.1f}m)")
    print(f"[{ts()}] Jobs: {done - skipped} run, {skipped} skipped")
    print(f"[{ts()}] Results: {args.output_dir}/")
    print(f"[{ts()}] Next: python3 scripts/analyze_gazebo.py")

    # Cleanup ROS
    try:
        import rclpy
        rclpy.shutdown()
    except Exception:
        pass


if __name__ == "__main__":
    main()
