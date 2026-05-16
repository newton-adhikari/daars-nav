#!/usr/bin/env python3
"""Analyze Gazebo results and produce table + Wilcoxon tests.
"""

import json, glob, sys
import numpy as np
from scipy.stats import wilcoxon

METHODS = ["daars", "static", "alpha_only", "beta_only", "ppo_lag"]
LABELS  = {"daars": "DAARS", "static": "Static", "alpha_only": "$\\alpha$-only",
           "beta_only": "$\\beta$-only", "ppo_lag": "PPO-Lag"}
GAZEBO_DIR = "results/gazebo"


def load_method(method):
    files = sorted(glob.glob(f"{GAZEBO_DIR}/{method}_seed*.json"))
    if not files:
        return None
    seed_data = {"sr": [], "cr": [], "cl": [], "n_succ": 0, "n_total": 0}
    for f in files:
        with open(f) as fh:
            data = json.load(fh)
        n = len(data)
        seed_data["n_total"] += n
        sr = sum(1 for e in data if e["success"]) / n
        cr = sum(1 for e in data if e["collision"]) / n
        succ = [e for e in data if e["success"]]
        seed_data["n_succ"] += len(succ)
        cl = np.mean([e["min_clearance"] for e in succ]) if succ else float("nan")
        seed_data["sr"].append(sr)
        seed_data["cr"].append(cr)
        seed_data["cl"].append(cl)
    return seed_data


def main():
    all_data = {}
    for m in METHODS:
        d = load_method(m)
        if d is None:
            print(f"WARNING: No data for {m}")
            sys.exit(1)
        all_data[m] = d

    n_seeds = len(all_data["daars"]["sr"])
    n_total = all_data["daars"]["n_total"]
    eps_per_seed = n_total // n_seeds

    print(f"Gazebo results: {n_seeds} seeds × {eps_per_seed} episodes = {n_total} per method\n")

    # Summary table
    print("=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    for m in METHODS:
        d = all_data[m]
        sr_m, sr_s = np.mean(d["sr"]), np.std(d["sr"], ddof=1)
        cr_m, cr_s = np.mean(d["cr"]), np.std(d["cr"], ddof=1)
        valid_cl = [c for c in d["cl"] if not np.isnan(c)]
        cl_m = np.mean(valid_cl) if valid_cl else 0
        cl_s = np.std(valid_cl, ddof=1) if len(valid_cl) > 1 else 0
        print(f"  {LABELS[m]:15s}  SR={sr_m:.3f}±{sr_s:.3f}  CR={cr_m:.3f}±{cr_s:.3f}  "
              f"CL={cl_m:.3f}±{cl_s:.3f}m  ({d['n_succ']}/{d['n_total']} succ)")

    # Wilcoxon tests: DAARS vs each other
    print(f"\n{'=' * 70}")
    print("  WILCOXON SIGNED-RANK TESTS (DAARS vs others, paired by seed)")
    print(f"{'=' * 70}")
    daars = all_data["daars"]
    for other_key in ["static", "alpha_only", "beta_only", "ppo_lag"]:
        other = all_data[other_key]
        print(f"\n  DAARS vs {LABELS[other_key]}:")
        for metric, label, lower_better in [("cr", "CR", True), ("sr", "SR", False), ("cl", "CL", False)]:
            d_vals = np.array(daars[metric])
            o_vals = np.array(other[metric])
            # Truncate to equal length (paired by seed index)
            n_paired = min(len(d_vals), len(o_vals))
            d_vals = d_vals[:n_paired]
            o_vals = o_vals[:n_paired]
            # Remove NaN pairs for CL
            mask = np.isfinite(d_vals) & np.isfinite(o_vals)
            d_v, o_v = d_vals[mask], o_vals[mask]
            diff = d_v - o_v
            if np.all(diff == 0):
                print(f"    {label}: identical (all diffs = 0)")
                continue
            try:
                stat, p = wilcoxon(d_v, o_v, alternative="two-sided")
                direction = "DAARS better" if (np.mean(diff) < 0) == lower_better else "other better"
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
                print(f"    {label}: DAARS {np.mean(d_v):.3f} vs {np.mean(o_v):.3f}  "
                      f"p={p:.4f} ({sig})  [{direction}]")
            except Exception as e:
                print(f"    {label}: test failed ({e})")

    # data for table
    print(f"\n{'=' * 70}")
    print("  LaTeX TABLE (copy into paper)")
    print(f"{'=' * 70}")

    # Find best values
    best_sr = max(np.mean(all_data[m]["sr"]) for m in METHODS)
    best_cr = min(np.mean(all_data[m]["cr"]) for m in METHODS)
    best_cl = max(np.nanmean(all_data[m]["cl"]) for m in METHODS)

    # Compute significance markers vs DAARS
    sig_markers = {}
    for other_key in ["static", "alpha_only", "beta_only", "ppo_lag"]:
        sig_markers[other_key] = {}
        for metric in ["cr", "cl"]:
            d_v = np.array(daars[metric])
            o_v = np.array(all_data[other_key][metric])
            n_paired = min(len(d_v), len(o_v))
            d_v, o_v = d_v[:n_paired], o_v[:n_paired]
            mask = np.isfinite(d_v) & np.isfinite(o_v)
            try:
                _, p = wilcoxon(d_v[mask], o_v[mask], alternative="two-sided")
                if p < 0.001:
                    sig_markers[other_key][metric] = "^\\S"
                elif p < 0.01:
                    sig_markers[other_key][metric] = "^\\ddagger"
                elif p < 0.05:
                    sig_markers[other_key][metric] = "^\\dagger"
                else:
                    sig_markers[other_key][metric] = ""
            except:
                sig_markers[other_key][metric] = ""

    for m in METHODS:
        d = all_data[m]
        sr_m, sr_s = np.mean(d["sr"]), np.std(d["sr"], ddof=1)
        cr_m, cr_s = np.mean(d["cr"]), np.std(d["cr"], ddof=1)
        valid_cl = [c for c in d["cl"] if not np.isnan(c)]
        cl_m = np.mean(valid_cl) if valid_cl else 0
        cl_s = np.std(valid_cl, ddof=1) if len(valid_cl) > 1 else 0

        sr_bold = "\\mathbf{" if abs(sr_m - best_sr) < 0.001 else ""
        sr_end = "}" if sr_bold else ""
        cr_bold = "\\mathbf{" if abs(cr_m - best_cr) < 0.001 else ""
        cr_end = "}" if cr_bold else ""
        cl_bold = "\\mathbf{" if abs(cl_m - best_cl) < 0.001 else ""
        cl_end = "}" if cl_bold else ""

        cr_sig = sig_markers.get(m, {}).get("cr", "")
        cl_sig = sig_markers.get(m, {}).get("cl", "")

        print(f"        {LABELS[m]:20s} & ${sr_bold}{sr_m:.3f} \\pm {sr_s:.3f}{sr_end}$ "
              f"& ${cr_bold}{cr_m:.3f} \\pm {cr_s:.3f}{cr_end}{cr_sig}$ "
              f"& ${cl_bold}{cl_m:.3f} \\pm {cl_s:.3f}{cl_end}{cl_sig}$ \\\\")


if __name__ == "__main__":
    main()
 