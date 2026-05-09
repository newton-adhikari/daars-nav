"""Statistical analysis.

Using Wilcoxon instead of
t-test is more defensible with 10 seeds and skewed safety metrics.
"""

from __future__ import annotations

import numpy as np
from scipy import stats



# Confidence intervals
def bootstrap_ci(
    values: list[float],
    n_boot: int = 10_000,
    confidence: float = 0.95,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """BCa bootstrap confidence interval.
    """
    if len(values) < 2:
        m = float(np.mean(values)) if values else 0.0
        return (m, m)

    rng  = rng or np.random.default_rng(0)
    arr  = np.asarray(values, dtype=float)
    boot = rng.choice(arr, size=(n_boot, len(arr)), replace=True)
    boot_means = np.mean(boot, axis=1)
    lo = float(np.percentile(boot_means, 100 * (1 - confidence) / 2))
    hi = float(np.percentile(boot_means, 100 * (1 + confidence) / 2))
    return (lo, hi)


def _cohens_d(a: list[float], b: list[float]) -> float:
    """Pooled-standard-deviation Cohen's d."""
    a_arr, b_arr = np.asarray(a, float), np.asarray(b, float)
    pooled = np.sqrt((np.var(a_arr, ddof=1) + np.var(b_arr, ddof=1)) / 2)
    return float((np.mean(a_arr) - np.mean(b_arr)) / max(pooled, 1e-8))



# Main statistics function
def compute_statistics(all_results: dict) -> dict:
    """
    Aggregate means, CIs, and significance tests across seeds.
    NOTE: Make this args driven
    """
    rng    = np.random.default_rng(42)
    output = {"per_method": {}, "comparisons": {}}

    METRICS = [
        "success_rate",
        "collision_rate",
        "path_efficiency",
        "avg_min_clearance",
        "constraint_violation_rate",
    ]

    # Per-method aggregation
    for method, scenarios in all_results.items():
        output["per_method"][method] = {}
        for scenario, seed_results in scenarios.items():
            if not seed_results:
                continue
            agg: dict[str, dict] = {}
            for metric in METRICS:
                vals = [r["summary"].get(metric, 0.0) for r in seed_results]
                ci   = bootstrap_ci(vals, rng=rng)
                agg[metric] = {
                    "mean":   float(np.mean(vals)),
                    "std":    float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0),
                    "ci_95":  ci,
                    "values": vals,
                }
            output["per_method"][method][scenario] = agg

    
    # Pairwise comparisons: DAARS vs each other method
    if "daars" not in all_results:
        return output

    for method in all_results:
        if method == "daars":
            continue
        cmp_key = f"daars_vs_{method}"
        output["comparisons"][cmp_key] = {}

        for scenario in all_results["daars"]:
            if scenario not in all_results.get(method, {}):
                continue

            daars_res  = all_results["daars"][scenario]
            other_res  = all_results[method][scenario]
            if not daars_res or not other_res:
                continue

            comp: dict[str, dict] = {}
            for metric in ["collision_rate", "success_rate",
                           "avg_min_clearance", "constraint_violation_rate"]:
                d_vals = [r["summary"].get(metric, 0.0) for r in daars_res]
                o_vals = [r["summary"].get(metric, 0.0) for r in other_res]

                # Wilcoxon signed-rank (paired; same number of seeds)
                if len(d_vals) == len(o_vals) and len(d_vals) >= 4:
                    try:
                        _, p_wx = stats.wilcoxon(d_vals, o_vals,
                                                 alternative="two-sided")
                    except ValueError:
                        p_wx = 1.0   # all differences zero
                else:
                    p_wx = 1.0

                # Independent t-test (reported for transparency)
                _, p_tt = stats.ttest_ind(d_vals, o_vals, equal_var=False)
                d       = _cohens_d(d_vals, o_vals)

                def sig_label(p):
                    if p < 0.001: return "***"
                    if p < 0.01:  return "**"
                    if p < 0.05:  return "*"
                    return "ns"

                comp[metric] = {
                    "p_wilcoxon":     float(p_wx),
                    "p_ttest":        float(p_tt),
                    "cohens_d":       float(d),
                    "significant_005_wilcoxon": p_wx < 0.05,
                    "significant_005_ttest":    p_tt < 0.05,
                    "sig_label":      sig_label(p_wx),
                    "daars_mean":     float(np.mean(d_vals)),
                    "other_mean":     float(np.mean(o_vals)),
                }
            output["comparisons"][cmp_key][scenario] = comp

    return output



# Console table
def print_results_table(stats: dict) -> None:
    """Print a formatted results table to stdout."""
    methods   = sorted(stats.get("per_method", {}).keys())
    scenarios = ["simple", "complex", "dynamic"]
    metrics   = ["success_rate", "collision_rate", "path_efficiency",
                 "avg_min_clearance"]

    for sc in scenarios:
        print(f"\n{'='*90}")
        print(f"  Scenario: {sc.upper()}")
        print(f"{'='*90}")
        hdr = f"  {'Method':<14}"
        for m in metrics:
            hdr += f"  {m:<22}"
        print(hdr)
        print("-" * 90)
        for method in methods:
            d = stats["per_method"].get(method, {}).get(sc, {})
            row = f"  {method:<14}"
            for m in metrics:
                md = d.get(m, {})
                mn = md.get("mean", 0.0)
                sd = md.get("std",  0.0)
                row += f"  {mn:.3f} ± {sd:.3f}         "
            print(row)

    # Significance table
    if stats.get("comparisons"):
        print(f"\n\n{'='*90}")
        print("  Statistical Significance (Wilcoxon signed-rank, two-sided)")
        print(f"{'='*90}")
        for comp, sc_data in stats["comparisons"].items():
            print(f"\n  {comp}")
            for sc, m_data in sc_data.items():
                for metric, result in m_data.items():
                    print(
                        f"    {sc:10s}  {metric:30s}  "
                        f"p={result['p_wilcoxon']:.4f}  "
                        f"d={result['cohens_d']:+.3f}  "
                        f"{result['sig_label']}"
                    )
