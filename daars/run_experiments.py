#!/usr/bin/env python3

"""
DAARS Experiment Pipeline
"""

from __future__ import annotations

import os
import json
import argparse
import traceback

import numpy as np
import yaml
from tqdm import tqdm

from daars.training.train      import train_agent, train_all_parallel
from daars.evaluation.evaluate import evaluate_agent
from daars.analysis.statistics import (
    compute_statistics, print_results_table,
    analyze_convergence, analyze_seed_variance,
)
from daars.analysis.plots      import generate_all_figures

# these are the baselines
ALL_METHODS = ["static", "alpha_only", "daars", "beta_only", "ppo_lag", "focops"]

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

def _eval_worker(args):
    """Worker for parallel evaluation."""
    config, model_path, method, scenario, n_ep = args
    try:
        r = evaluate_agent(config, model_path, reward_type=method,
                           scenario=scenario, num_episodes=n_ep)
        return {"result": r, "method": method, "scenario": scenario, "error": None}
    except Exception as e:
        traceback.print_exc()
        return {"result": None, "method": method, "scenario": scenario,
                "error": str(e)}


def run_evaluation(config, output_dir, max_workers=None):
    scenarios = config["evaluation"]["scenarios"]
    seeds     = list(range(int(config["training"]["num_seeds"])))
    n_ep      = int(config["evaluation"]["num_episodes"])
    model_dir = os.path.join(output_dir, "models")
    res_path  = os.path.join(output_dir, "eval_results.json")
    results   = _load(res_path)

    for m in ALL_METHODS:
        results.setdefault(m, {sc: [] for sc in scenarios})

    # Build list of jobs, skipping already-completed ones
    jobs = []
    job_seeds = []
    for method in ALL_METHODS:
        for scenario in scenarios:
            existing = len(results[method].get(scenario, []))
            for si, seed in enumerate(seeds):
                if si < existing:
                    continue
                mp = os.path.join(model_dir, f"{method}_seed{seed}")
                if not os.path.exists(mp + ".zip"):
                    continue
                jobs.append((config, mp, method, scenario, n_ep))
                job_seeds.append((method, scenario, seed))

    total_skipped = len(ALL_METHODS) * len(scenarios) * len(seeds) - len(jobs)
    if not jobs:
        print("   All evaluations already done.")
        return results

    max_w = max_workers or 4  # eval is inference-only, safe to parallelize
    print(f"\n>> Parallel evaluation: {len(jobs)} jobs  "
          f"max_workers={max_w}  ({total_skipped} skipped)\n")

    import multiprocessing as mp_mod
    from concurrent.futures import ProcessPoolExecutor, as_completed
    ctx = mp_mod.get_context("spawn")

    errors = []
    done_count = 0
    with ProcessPoolExecutor(max_workers=max_w, mp_context=ctx) as pool:
        futures = {pool.submit(_eval_worker, j): (j, js)
                   for j, js in zip(jobs, job_seeds)}
        for fut in as_completed(futures):
            _, (method, scenario, seed) = futures[fut]
            res = fut.result()
            done_count += 1
            if res["error"] is None:
                results[res["method"]][res["scenario"]].append(res["result"])
                _save(results, res_path)
                print(f"   ✓ [{done_count}/{len(jobs)}] "
                      f"{method}/{scenario}/seed{seed}")
            else:
                errors.append(f"{method}/{scenario}/s{seed}: {res['error']}")
                print(f"   ✗ [{done_count}/{len(jobs)}] "
                      f"{method}/{scenario}/seed{seed}: {res['error']}")

    if errors:
        print(f"\n!! {len(errors)} evaluation errors:")
        for e in errors: print(f"   {e}")
    return results



# Ablation phase

def run_ablation(config, output_dir, max_workers=None):
    ks_vals    = config["ablation"]["ks_values"]
    ds_vals    = config["ablation"]["dsafe_values"]
    n_ep       = int(config["ablation"].get("eval_episodes", 50))
    sc         = config["ablation"].get("eval_scenario", "complex")
    n_envs_abl = int(config["ablation"].get("num_envs", 4))
    n_abl_seeds = int(config["ablation"].get("num_ablation_seeds", 5))
    model_dir  = os.path.join(output_dir, "models")
    abl_path   = os.path.join(output_dir, "ablation_results.json")

    abl = _load(abl_path) or {
        "ks_values":    ks_vals, "dsafe_values": ds_vals,
        "collision_rate_grid": [[None]*len(ds_vals) for _ in ks_vals],
        "success_rate_grid":   [[None]*len(ds_vals) for _ in ks_vals],
        "num_seeds": n_abl_seeds,
    }

    total  = len(ks_vals) * len(ds_vals)
    errors = []
    pbar   = tqdm(total=total, desc="Ablation")

    for i, ks in enumerate(ks_vals):
        for j, dsafe in enumerate(ds_vals):
            pbar.set_postfix(ks=ks, ds=dsafe)
            if abl["collision_rate_grid"][i][j] is not None:
                pbar.update(1); continue
            cr_seeds, sr_seeds = [], []
            for seed in range(n_abl_seeds):
                mn = f"daars_ks{ks}_ds{dsafe}_seed{seed}"
                try:
                    if not _model_exists(model_dir, mn):
                        train_agent(config, "daars", seed, "simple", model_dir,
                                    daars_ks=ks, daars_dsafe=dsafe,
                                    num_envs=n_envs_abl)
                    r = evaluate_agent(
                        config, os.path.join(model_dir, mn),
                        reward_type="daars", scenario=sc,
                        num_episodes=n_ep, daars_ks=ks, daars_dsafe=dsafe)
                    cr_seeds.append(r["summary"]["collision_rate"])
                    sr_seeds.append(r["summary"]["success_rate"])
                except Exception as e:
                    errors.append(f"ks={ks} ds={dsafe} s{seed}: {e}")
                    traceback.print_exc()
            if cr_seeds:
                abl["collision_rate_grid"][i][j] = float(np.mean(cr_seeds))
                abl["success_rate_grid"][i][j]   = float(np.mean(sr_seeds))
                _save(abl, abl_path)
            pbar.update(1)

    pbar.close()
    if errors:
        print(f"\n!! {len(errors)} ablation errors:")
        for e in errors: print(f"   {e}")
    return abl

# Safety-Gymnasium benchmark phase
def run_safety_gym_benchmark_phase(config, output_dir, max_workers=None, force_retrain=False):
    """Run Safety-Gymnasium standard benchmark for reviewer comparison."""
    from daars.evaluation.safety_gym_benchmark import (
        run_safety_gym_benchmark, SAFETY_GYM_AVAILABLE,
    )
    if not SAFETY_GYM_AVAILABLE:
        print("\n!! safety-gymnasium not installed — skipping benchmark.")
        print("   Install: pip install safety-gymnasium")
        return {}

    sg_cfg = config.get("safety_gym", {})
    num_seeds = int(sg_cfg.get("num_seeds", 10))
    total_ts = int(sg_cfg.get("timesteps", 1_000_000))
    num_envs = int(sg_cfg.get("num_envs", 4))
    return run_safety_gym_benchmark(
        config, output_dir,
        num_seeds=num_seeds,
        total_timesteps=total_ts,
        num_envs=num_envs,
        force_retrain=force_retrain,
    )


# verify the novelty
def run_theory(config, output_dir):
    from daars.theory.analysis import run_all_proofs
    r = run_all_proofs(config)
    _save(r, os.path.join(output_dir, "theory_verification.json"))
    return r

# plots

def run_plotting(config, output_dir):
    curves  = _load(os.path.join(output_dir, "training_curves.json"))
    results = _load(os.path.join(output_dir, "eval_results.json"))
    ablation= _load(os.path.join(output_dir, "ablation_results.json"))
    if not results:
        print("No evaluation results — run evaluate first.")
        return
    stats = compute_statistics(results)

    # Add convergence analysis (reviewer: "convergence unclear")
    if curves:
        stats["convergence_analysis"] = analyze_convergence(curves)

    # Add variance analysis (reviewer: "extremely high variance")
    variance = analyze_seed_variance(stats)
    stats["variance_analysis"] = variance
    if variance["warnings"]:
        print("\n  ⚠ High-variance scenarios:")
        for w in variance["warnings"]:
            print(f"    {w}")

    _save(stats, os.path.join(output_dir, "statistics.json"))
    generate_all_figures(stats, curves, ablation,
                         os.path.join(output_dir, "figures"), config=config)
    print_results_table(stats)

def main():
    parser = argparse.ArgumentParser(description="DAARS pipeline")
    parser.add_argument("--config",  default=None)
    parser.add_argument("--output",  default="results")
    parser.add_argument("--phase",   default="all",
        choices=["all","train","evaluate","ablation","theory","plot","safetygym"])
    parser.add_argument("--fast",    action="store_true",
        help="Smoke test: 2 seeds, 20k steps, 20 eps, 2 envs")
    parser.add_argument("--extended", action="store_true",
        help="Extended training: 1M steps for convergence verification")
    parser.add_argument("--workers", type=int, default=None,
        help="Override max_parallel_jobs")
    parser.add_argument("--envs",    type=int, default=None,
        help="Override num_envs")
    parser.add_argument("--force-retrain", action="store_true",
        help="Force retrain Safety-Gym models (ignore existing .zip files)")
    args = parser.parse_args()

    config = load_config(args.config)

    os.makedirs(args.output, exist_ok=True)

    if args.fast:
        config["training"]["num_seeds"]       = 2
        config["training"]["total_timesteps"] = 20_000
        config["training"]["n_steps"]         = 512
        config["training"]["num_envs"]        = 2
        config["training"]["max_parallel_jobs"] = 2
        config["evaluation"]["num_episodes"]  = 10
        config["ablation"]["ks_values"]       = [1.0, 1.5]
        config["ablation"]["dsafe_values"]    = [0.8, 1.0]
        config["ablation"]["eval_episodes"]   = 5
        # Safety-Gym fast mode
        config.setdefault("safety_gym", {})
        config["safety_gym"]["timesteps"]     = 20_000
        config["safety_gym"]["num_seeds"]     = 2
        config["safety_gym"]["num_envs"]      = 2
        print("*** FAST MODE (smoke test) ***\n")

    if args.extended:
        config["training"]["total_timesteps"] = 1_000_000
        print("*** EXTENDED MODE (1M steps for convergence) ***\n")

    if args.phase in ("all", "train"):
        run_training(config, args.output, args.workers, args.envs)
    if args.phase in ("all", "evaluate"):
        run_evaluation(config, args.output, args.workers)
    if args.phase in ("all", "ablation"):
        run_ablation(config, args.output, args.workers)
    if args.phase in ("all", "theory"):
        run_theory(config, args.output)
    if args.phase in ("all", "safetygym"):
        run_safety_gym_benchmark_phase(config, args.output, args.workers,
                                       force_retrain=args.force_retrain)
    if args.phase in ("all", "plot"):
        run_plotting(config, args.output)

    print("\nDone.")

if __name__ == "__main__":
    main()
 