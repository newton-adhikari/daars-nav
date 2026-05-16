#!/usr/bin/env python3
# use daars policy in TB3 in gazebo


import os
import sys
import json
import argparse
import numpy as np
 
# for root path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
 
import yaml
from stable_baselines3 import PPO, SAC
 
from daars.rewards.daars_reward import daars_reward, daars_cost
 
 
def load_config(path=None):
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "..", "daars", "config", "default.yaml")
    with open(path) as f:
        return yaml.safe_load(f)
 
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    required=True, help="Path to .zip model (no extension)")
    parser.add_argument("--config",   default=None)
    parser.add_argument("--goal",     nargs=2, type=float, default=None,
                        metavar=("X","Y"), help="Fixed goal position in metres")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--output",   default="gazebo_results.json")
    args = parser.parse_args()
 
    config   = load_config(args.config)
    goal_pos = tuple(args.goal) if args.goal else None
 
    # for non-ROS machines 
    from daars.gazebo.gazebo_env import GazeboNavEnv
 
    env = GazeboNavEnv(
        config,
        reward_fn = daars_reward,
        cost_fn   = daars_cost,
        goal_pos  = goal_pos,
    )
 
    print(f"\n>> Loading model: {args.model}")
    # to handle SB3 version mismatches
    custom_objects = {
        "observation_space": env.observation_space,
        "action_space": env.action_space,
    }
    try:
        model = PPO.load(args.model, custom_objects=custom_objects)
    except Exception:
        model = SAC.load(args.model, custom_objects=custom_objects)
 
    records = []
    print(f">> Running {args.episodes} episodes in Gazebo...\n")
 
    for ep in range(args.episodes):
        obs, info = env.reset()
        done      = False
        ep_reward = 0.0
 
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated
 
        rec = {
            "episode":       ep,
            "success":       bool(info.get("reached_goal", False)),
            "collision":     bool(info.get("collision", False)),
            "path_length":   float(info.get("path_length", 0.0)),
            "min_clearance": float(info.get("min_clearance", 0.0)),
            "steps":         int(info.get("steps", 0)),
            "reward":        float(ep_reward),
            "goal_pos":      env.goal_pos.tolist(),
        }
        records.append(rec)
 
        # Get final robot position
        with env._lock:
            rx, ry = env._robot_x, env._robot_y
        goal_dist = float(np.linalg.norm(env.goal_pos - [rx, ry]))
 
        status = "SUCCESS" if rec["success"] else ("COLLISION" if rec["collision"] else "TIMEOUT")
        print(f"  ep {ep+1:3d}/{args.episodes}  {status}  "
              f"clr={rec['min_clearance']:.3f}m  "
              f"len={rec['path_length']:.2f}m  "
              f"steps={rec['steps']}  "
              f"robot=({rx:.1f},{ry:.1f})  "
              f"goal=({env.goal_pos[0]:.1f},{env.goal_pos[1]:.1f})  "
              f"d_goal={goal_dist:.2f}m")
 
    env.close()
 
    sr = np.mean([r["success"]   for r in records])
    cr = np.mean([r["collision"] for r in records])
    mc = np.mean([r["min_clearance"] for r in records])
    print(f"\n  SR={sr:.3f}  CR={cr:.3f}  Avg-clearance={mc:.3f}m")
 
    with open(args.output, "w") as f:
        json.dump(records, f, indent=2)
    print(f"\n  Results saved → {args.output}")
 
 
if __name__ == "__main__":
    main()
