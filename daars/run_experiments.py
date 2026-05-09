#!/usr/bin/env python3

"""
DAARS Experiment Pipeline
"""

import os
import json
import argparse

import numpy as np
import yaml

from daars.training.train      import train_agent, train_all_parallel


# these are the baselines
ALL_METHODS = ["static", "alpha_only", "daars", "beta_only", "ppo_lag"]

def load_config(path=None):
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "config", "default.yaml")
    with open(path) as f:
        return yaml.safe_load(f)
    
def _load(p):
    return json.load(open(p)) if os.path.exists(p) else {}

def _save(obj, p):
    with open(p, "w") as f:
        json.dump(obj, f, indent=2, default=_serial)

def _serial(o):
    if isinstance(o, (np.floating,)):  return float(o)
    if isinstance(o, (np.integer,)):   return int(o)
    if isinstance(o, (np.bool_,)):     return bool(o)
    if isinstance(o, (np.ndarray,)):   return o.tolist()
    raise TypeError(f"Not serialisable: {type(o)}")

def _model_exists(d, name):
    return os.path.exists(os.path.join(d, name + ".zip"))


# training starts
def run_training(config, output_dir, max_workers=None, num_envs=None):
    """Train all methods × seeds using parallel launcher."""
    if num_envs is not None:
        config["training"]["num_envs"] = num_envs

    model_dir   = os.path.join(output_dir, "models")
    curves_path = os.path.join(output_dir, "training_curves.json")
    curves      = _load(curves_path)
    for m in ALL_METHODS:
        curves.setdefault(m, [])

    # Parallel training
    all_results = train_all_parallel(
        config, ALL_METHODS, model_dir,
        scenario="simple",
        max_workers=max_workers,
    )

    # Merge training curves
    for method, res_list in all_results.items():
        for res in res_list:
            seed = res["seed"]
            while len(curves[method]) <= seed:
                curves[method].append(None)
            curves[method][seed] = res["training_curves"]
    _save(curves, curves_path)

    print(f"\n   Training curves → {curves_path}")
    return curves

# project entry point

def main():
    parser = argparse.ArgumentParser(description="DAARS pipeline")
    parser.add_argument("--config",  default=None)
    parser.add_argument("--output",  default="results")
    parser.add_argument("--phase",   default="all",
        choices=["all","train","evaluate","ablation","theory","plot","safetygym"])
    parser.add_argument("--fast",    action="store_true",
        help="Smoke test: 2 seeds, 20k steps, 20 eps, 2 envs")
    parser.add_argument("--workers", type=int, default=None,
        help="Override max_parallel_jobs")
    parser.add_argument("--envs",    type=int, default=None,
        help="Override num_envs")
    args = parser.parse_args()

    config = load_config(args.config)

if __name__ == "__main__":
    main()
 