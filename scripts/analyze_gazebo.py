#!/usr/bin/env python3
"""
Analyze Gazebo evaluation results and produce publication-ready output.

Generates:
  1. Summary table (mean ± std across seeds)
  2. Wilcoxon signed-rank tests (DAARS vs each baseline, paired by seed)
  3. Effect sizes (Cohen's d)
  4. LaTeX table for paper
  5. JSON summary for programmatic use

Usage:
    python3 scripts/analyze_gazebo.py
    python3 scripts/analyze_gazebo.py --output results/gazebo/analysis.json
"""

import os
import sys
import json
import glob
import argparse
import numpy as np
from scipy.stats import wilcoxon, mannwhitneyu

GAZEBO_DIR = "results/gazebo"
METHODS = ["daars", "static", "alpha_only", "beta_only", "ppo_lag", "focops"]
LABELS = {
    "daars": "DAARS",
    "static": "Static",
    "alpha_only": r"$\alpha$-only",
    "beta_only": r"$\beta$-only",
    "ppo_lag": "PPO-Lag",
    "focops": "FOCOPS",
}


def load_method_data(method: str, gazebo_dir: str = GAZEBO_DIR) -> dict | None:
    """Load all seed results for a method. Returns per-seed aggregates."""
    files = sorted(glob.glob(f"{gazebo_dir}/{method}_seed*.json"))
    if not files:
        return None

    seed_metrics = {
        "sr": [], "cr": [], "clearance": [], "path_length": [],
        "steps": [], "cost": [], "n_episodes": 0,
    }

    for f in files:
        with open(f) as fh:
            data = json.load(fh)

        # Handle both formats: {summary, episodes} or raw list
        if isinstance(data, dict) and "episodes" in data:
            episodes = data["episodes"]
        elif isinstance(data, list):
            episodes = data
        else:
            continue

        n = len(episodes)
        seed_metrics["n_episodes"] += n

        sr = sum(1 for e in episodes if e.get("success", False)) / n
        cr = sum(1 for e in episodes if e.get("collision", False)) / n

        # Clearance: use all episodes (not just successful ones) for fair comparison
        clearances = [e["min_clearance"] for e in episodes
                      if "min_clearance" in e and e["min_clearance"] < float("inf")]
        avg_clr = np.mean(clearances) if clearances else float("nan")

        # Path length and steps for successful episodes only
        succ = [e for e in episodes if e.get("success", False)]
        avg_pl = np.mean([e["path_length"] for e in succ]) if succ else float("nan")
        avg_steps = np.mean([e["steps"] for e in episodes])

        # Cost
        costs = [e.get("cumulative_cost", 0) for e in episodes]
        avg_cost = np.mean(costs)

        seed_metrics["sr"].append(sr)
        seed_metrics["cr"].append(cr)
        seed_metrics["clearance"].append(avg_clr)
        seed_metrics["path_length"].append(avg_pl)
        seed_metrics["steps"].append(avg_steps)
        seed_metrics["cost"].append(avg_cost)

    if not seed_metrics["sr"]:
        return None
    return seed_metrics


def cohens_d(x, y):
    """Compute Cohen's d for paired samples."""
    diff = np.array(x) - np.array(y)
    return np.mean(diff) / (np.std(diff, ddof=1) + 1e-10)


def wilcoxon_test(x, y):
    """Wilcoxon signed-rank test. Returns (statistic, p-value)."""
    x, y = np.array(x), np.array(y)
    # Remove pairs where both are equal (Wilcoxon can't handle all-zero diffs)
    diff = x - y
    if np.all(diff == 0):
        return 0.0, 1.0
    # Remove NaN pairs
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 5:
        return 0.0, 1.0
    try:
        stat, p = wilcoxon(x, y, alternative="two-sided")
        return float(stat), float(p)
    except Exception:
        return 0.0, 1.0


def sig_stars(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return "ns"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/gazebo/analysis.json")
    parser.add_argument("--gazebo-dir", default=GAZEBO_DIR)
    args = parser.parse_args()

    gazebo_dir = args.gazebo_dir

    # Load all data
    all_data = {}
    available_methods = []
    for m in METHODS:
        d = load_method_data(m, gazebo_dir)
        if d is not None:
            all_data[m] = d
            available_methods.append(m)

    if not all_data:
        print("ERROR: No Gazebo results found in", GAZEBO_DIR)
        print("Run ./scripts/run_all_gazebo.sh first.")
        sys.exit(1)

    if "daars" not in all_data:
        print("ERROR: No DAARS results found. Cannot compute comparisons.")
        sys.exit(1)

    n_seeds = len(all_data["daars"]["sr"])
    n_total = all_data["daars"]["n_episodes"]
    eps_per_seed = n_total // n_seeds if n_seeds > 0 else 0

    print("=" * 75)
    print(f"  GAZEBO EVALUATION RESULTS")
    print(f"  {n_seeds} seeds × {eps_per_seed} episodes/seed = {n_total} total per method")
    print("=" * 75)

    # ─── Summary Table ───────────────────────────────────────────────────
    print(f"\n{'Method':<15} {'SR↑':>12} {'CR↓':>12} {'Clearance↑':>14} {'PathLen':>12}")
    print("-" * 75)

    for m in available_methods:
        d = all_data[m]
        sr_m, sr_s = np.mean(d["sr"]), np.std(d["sr"], ddof=1)
        cr_m, cr_s = np.mean(d["cr"]), np.std(d["cr"], ddof=1)
        cl_vals = [c for c in d["clearance"] if np.isfinite(c)]
        cl_m = np.mean(cl_vals) if cl_vals else 0
        cl_s = np.std(cl_vals, ddof=1) if len(cl_vals) > 1 else 0
        pl_vals = [p for p in d["path_length"] if np.isfinite(p)]
        pl_m = np.mean(pl_vals) if pl_vals else 0
        pl_s = np.std(pl_vals, ddof=1) if len(pl_vals) > 1 else 0

        print(f"{LABELS[m]:<15} {sr_m:.3f}±{sr_s:.3f}  {cr_m:.3f}±{cr_s:.3f}  "
              f"{cl_m:.3f}±{cl_s:.3f}m  {pl_m:.2f}±{pl_s:.2f}m")

    # ─── Statistical Tests ───────────────────────────────────────────────
    print(f"\n{'=' * 75}")
    print("  STATISTICAL TESTS: DAARS vs baselines (Wilcoxon signed-rank, paired by seed)")
    print(f"{'=' * 75}")

    daars = all_data["daars"]
    comparisons = {}

    for other_key in available_methods:
        if other_key == "daars":
            continue
        other = all_data[other_key]
        n_paired = min(len(daars["sr"]), len(other["sr"]))
        if n_paired < 3:
            print(f"\n  DAARS vs {LABELS[other_key]}: insufficient paired data ({n_paired} seeds)")
            continue

        print(f"\n  DAARS vs {LABELS[other_key]} ({n_paired} paired seeds):")
        comp = {}

        for metric, label, lower_better in [
            ("cr", "Collision Rate", True),
            ("sr", "Success Rate", False),
            ("clearance", "Clearance", False),
        ]:
            d_vals = np.array(daars[metric][:n_paired])
            o_vals = np.array(other[metric][:n_paired])

            # Handle NaN
            mask = np.isfinite(d_vals) & np.isfinite(o_vals)
            d_v, o_v = d_vals[mask], o_vals[mask]

            if len(d_v) < 3:
                print(f"    {label:<18} insufficient valid pairs")
                continue

            _, p = wilcoxon_test(d_v, o_v)
            d_effect = cohens_d(d_v, o_v)
            stars = sig_stars(p)

            # Direction
            diff_mean = np.mean(d_v) - np.mean(o_v)
            if lower_better:
                direction = "DAARS better" if diff_mean < 0 else "baseline better"
            else:
                direction = "DAARS better" if diff_mean > 0 else "baseline better"

            print(f"    {label:<18} DAARS={np.mean(d_v):.3f} vs {np.mean(o_v):.3f}  "
                  f"p={p:.4f} {stars:>3}  d={d_effect:+.3f}  [{direction}]")

            comp[metric] = {
                "daars_mean": float(np.mean(d_v)),
                "other_mean": float(np.mean(o_v)),
                "p_value": float(p),
                "cohens_d": float(d_effect),
                "significant": p < 0.05,
                "direction": direction,
            }

        comparisons[f"daars_vs_{other_key}"] = comp

    # ─── Best Values ─────────────────────────────────────────────────────
    print(f"\n{'=' * 75}")
    print("  BEST VALUES")
    print(f"{'=' * 75}")

    for metric, label, maximize in [("sr", "Success Rate", True),
                                     ("cr", "Collision Rate", False),
                                     ("clearance", "Clearance", True)]:
        values = {}
        for m in available_methods:
            vals = [v for v in all_data[m][metric] if np.isfinite(v)]
            if vals:
                values[m] = np.mean(vals)
        if values:
            if maximize:
                best = max(values, key=values.get)
            else:
                best = min(values, key=values.get)
            print(f"  {label:<18} best = {LABELS[best]} ({values[best]:.4f})")

    # ─── LaTeX Table ─────────────────────────────────────────────────────
    print(f"\n{'=' * 75}")
    print("  LaTeX TABLE (copy into paper)")
    print(f"{'=' * 75}")
    print()
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\caption{Gazebo TurtleBot3 evaluation results (" +
          f"{n_seeds} seeds, {eps_per_seed} episodes/seed).}}")
    print(r"\label{tab:gazebo}")
    print(r"\begin{tabular}{lccc}")
    print(r"\toprule")
    print(r"Method & SR$\uparrow$ & CR$\downarrow$ & Clearance$\uparrow$ (m) \\")
    print(r"\midrule")

    # Find best values for bolding
    best_sr = max(np.mean(all_data[m]["sr"]) for m in available_methods)
    best_cr = min(np.mean(all_data[m]["cr"]) for m in available_methods)
    best_cl = max(np.nanmean(all_data[m]["clearance"]) for m in available_methods)

    for m in available_methods:
        d = all_data[m]
        sr_m, sr_s = np.mean(d["sr"]), np.std(d["sr"], ddof=1)
        cr_m, cr_s = np.mean(d["cr"]), np.std(d["cr"], ddof=1)
        cl_vals = [c for c in d["clearance"] if np.isfinite(c)]
        cl_m = np.mean(cl_vals) if cl_vals else 0
        cl_s = np.std(cl_vals, ddof=1) if len(cl_vals) > 1 else 0

        sr_str = f"{sr_m:.3f} \\pm {sr_s:.3f}"
        cr_str = f"{cr_m:.3f} \\pm {cr_s:.3f}"
        cl_str = f"{cl_m:.3f} \\pm {cl_s:.3f}"

        if abs(sr_m - best_sr) < 0.001:
            sr_str = r"\mathbf{" + sr_str + "}"
        if abs(cr_m - best_cr) < 0.001:
            cr_str = r"\mathbf{" + cr_str + "}"
        if abs(cl_m - best_cl) < 0.001:
            cl_str = r"\mathbf{" + cl_str + "}"

        label = LABELS[m].replace("$", "").replace(r"\alpha", r"$\alpha$").replace(r"\beta", r"$\beta$")
        print(f"{label} & ${sr_str}$ & ${cr_str}$ & ${cl_str}$ \\\\")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")

    # ─── Save JSON ───────────────────────────────────────────────────────
    output = {
        "metadata": {
            "n_seeds": n_seeds,
            "episodes_per_seed": eps_per_seed,
            "methods": available_methods,
        },
        "summaries": {},
        "comparisons": comparisons,
    }
    for m in available_methods:
        d = all_data[m]
        cl_vals = [c for c in d["clearance"] if np.isfinite(c)]
        output["summaries"][m] = {
            "success_rate": {"mean": float(np.mean(d["sr"])), "std": float(np.std(d["sr"], ddof=1))},
            "collision_rate": {"mean": float(np.mean(d["cr"])), "std": float(np.std(d["cr"], ddof=1))},
            "clearance": {"mean": float(np.mean(cl_vals)) if cl_vals else 0,
                          "std": float(np.std(cl_vals, ddof=1)) if len(cl_vals) > 1 else 0},
        }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Analysis saved → {args.output}")


if __name__ == "__main__":
    main()
 