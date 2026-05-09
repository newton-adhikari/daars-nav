#!/usr/bin/env python3

"""
DAARS Experiment Pipeline
"""

import os
import json
import argparse

import numpy as np
import yaml


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
 