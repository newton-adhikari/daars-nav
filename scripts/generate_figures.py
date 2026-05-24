#!/usr/bin/env python3
"""
Generate quality figures
"""

import os
import sys
import json
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "axes.titleweight": "bold",
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "legend.handlelength": 1.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
    "text.usetex": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "lines.linewidth": 1.2,
    "patch.linewidth": 0.5,
})

SINGLE_COL = 3.5   # inches (IEEE single column)
DOUBLE_COL = 7.16  # inches (IEEE double column)

# ─── Method styling (DAARS is primary, warm color) ────────────────────────
METHODS = ["daars", "static", "alpha_only", "beta_only", "ppo_lag", "focops"]

COLORS = {
    "daars":      "#D62728",  # red — our method
    "static":     "#1F77B4",  # blue
    "alpha_only": "#FF7F0E",  # orange
    "beta_only":  "#9467BD",  # purple
    "ppo_lag":    "#2CA02C",  # green
    "focops":     "#8C564B",  # brown
}

LABELS = {
    "daars":      "DAARS (ours)",
    "static":     "Static",
    "alpha_only": r"$\alpha$-only",
    "beta_only":  r"$\beta$-only",
    "ppo_lag":    "PPO-Lag",
    "focops":     "FOCOPS",
}

MARKERS = {
    "daars": "D", "static": "o", "alpha_only": "^",
    "beta_only": "s", "ppo_lag": "v", "focops": "P",
}

SCENARIOS = ["simple", "complex", "dynamic"]
SC_LABELS = {"simple": "Simple", "complex": "Complex", "dynamic": "Dynamic"}

OUT_DIR = "paper/figures"


def _save(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(os.path.join(OUT_DIR, f"{name}.pdf"))
    fig.savefig(os.path.join(OUT_DIR, f"{name}.png"))
    plt.close(fig)
    print(f"  ✓ {name}")


def load_json(path):
    with open(path) as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════════════════
# Fig 1: Main comparison bar chart (SR, CR, Clearance) — double column
# ═══════════════════════════════════════════════════════════════════════════

def fig1_main_comparison(stats):
    metrics = [
        ("success_rate", "Success Rate (SR)", (0.5, 1.05)),
        ("collision_rate", "Collision Rate (CR)", (-0.02, 0.40)),
        ("avg_min_clearance", "Clearance (m)", (0.25, 0.95)),
    ]
    methods = [m for m in METHODS if m in stats["per_method"]]
    n = len(methods)

    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, 2.5))
    x = np.arange(len(SCENARIOS))
    w = 0.78 / n

    for ax, (metric, ylabel, ylim) in zip(axes, metrics):
        for i, method in enumerate(methods):
            offset = (i - (n - 1) / 2) * w
            means, stds = [], []
            for sc in SCENARIOS:
                d = stats["per_method"].get(method, {}).get(sc, {}).get(metric, {})
                means.append(d.get("mean", 0))
                stds.append(d.get("std", 0))
            ax.bar(
                x + offset, means, w * 0.9, yerr=stds,
                color=COLORS[method], edgecolor="white", linewidth=0.3,
                capsize=1.2, error_kw=dict(elinewidth=0.5, capthick=0.4),
                label=LABELS[method], zorder=3,
            )
        ax.set_xticks(x)
        ax.set_xticklabels([SC_LABELS[s] for s in SCENARIOS])
        ax.set_ylabel(ylabel)
        ax.set_ylim(ylim)
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))
        ax.grid(axis="y", linewidth=0.3, alpha=0.4, zorder=0)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=n,
               frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, -0.04))
    fig.subplots_adjust(bottom=0.26, wspace=0.42)
    _save(fig, "fig1_main_comparison")


# ═══════════════════════════════════════════════════════════════════════════
# Fig 2: Training curves with confidence bands — double column
# ═══════════════════════════════════════════════════════════════════════════

def fig2_training_curves(curves):
    curve_keys = ["rewards", "successes", "collisions"]
    ylabels = ["Episode Reward", "Success Rate", "Collision Rate"]
    methods = [m for m in METHODS if m in curves]
    window = 80

    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, 2.2))

    for ax, key, ylabel in zip(axes, curve_keys, ylabels):
        for method in methods:
            seeds_data = curves.get(method, [])
            smoothed = []
            for sd in seeds_data:
                if sd is None:
                    continue
                v = np.array(sd.get(key, []), dtype=float)
                if len(v) < window:
                    continue
                sm = np.convolve(v, np.ones(window) / window, mode="valid")
                smoothed.append(sm)
            if not smoothed:
                continue
            L = min(len(s) for s in smoothed)
            mat = np.array([s[:L] for s in smoothed])
            mu = np.mean(mat, axis=0)
            lo = np.percentile(mat, 10, axis=0)
            hi = np.percentile(mat, 90, axis=0)
            ep = np.arange(L)
            c = COLORS.get(method, "#aaa")
            ax.plot(ep, mu, color=c, lw=1.0, label=LABELS.get(method, method))
            ax.fill_between(ep, lo, hi, color=c, alpha=0.12)

        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        ax.grid(linewidth=0.3, alpha=0.3)
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(methods), 6),
               frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, -0.04))
    fig.subplots_adjust(bottom=0.24, wspace=0.38)
    _save(fig, "fig2_training_curves")


# ═══════════════════════════════════════════════════════════════════════════
# Fig 3: α and β scaling functions — single column
# ═══════════════════════════════════════════════════════════════════════════

def fig3_scaling_functions():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE_COL * 0.75, 2.2))
    d = np.linspace(0.01, 2.5, 500)

    # α(d)
    ALPHA_MIN = 0.3
    for ds, ls, c in [(0.5, "-", "#1f77b4"), (0.8, "--", "#ff7f0e"),
                       (1.0, "-.", "#d62728"), (1.5, ":", "#2ca02c")]:
        a = np.clip(d / ds, ALPHA_MIN, 1.0)
        ax1.plot(d, a, ls=ls, lw=1.2, color=c, label=f"$d_{{\\mathrm{{safe}}}}={ds}$")
    ax1.axhline(ALPHA_MIN, color="gray", ls=":", lw=0.6, alpha=0.6)
    ax1.set_xlabel(r"$d_{\mathrm{obs}}$ (m)")
    ax1.set_ylabel(r"$\alpha(d_{\mathrm{obs}})$")
    ax1.set_ylim(-0.02, 1.1)
    ax1.legend(frameon=False, fontsize=6.5, loc="lower right")
    ax1.grid(linewidth=0.3, alpha=0.3)
    ax1.set_title(r"(a) Goal modulator $\alpha$")

    # β(d)
    BETA_MAX = 5.0
    eps = 0.1
    for ks, ls, c in [(0.3, "-", "#1f77b4"), (0.5, "--", "#ff7f0e"),
                       (0.8, "-.", "#d62728"), (1.2, ":", "#2ca02c")]:
        b = np.exp(ks / (d + eps)) - np.exp(ks / (d + eps + 2.0))
        b = np.minimum(b, BETA_MAX)
        ax2.plot(d, b, ls=ls, lw=1.2, color=c, label=f"$k_s={ks}$")
    ax2.axhline(BETA_MAX, color="gray", ls=":", lw=0.6, alpha=0.6)
    ax2.set_xlabel(r"$d_{\mathrm{obs}}$ (m)")
    ax2.set_ylabel(r"$\beta(d_{\mathrm{obs}})$")
    ax2.set_ylim(-0.2, 5.8)
    ax2.legend(frameon=False, fontsize=6.5, loc="upper right")
    ax2.grid(linewidth=0.3, alpha=0.3)
    ax2.set_title(r"(b) Safety amplifier $\beta$")

    fig.tight_layout(pad=0.8)
    _save(fig, "fig3_scaling_functions")


# ═══════════════════════════════════════════════════════════════════════════
# Fig 4: Ablation heatmaps (k_s × d_safe) — double column
# ═══════════════════════════════════════════════════════════════════════════

def fig4_ablation_heatmaps(ablation):
    if not ablation:
        return
    ks_vals = ablation["ks_values"]
    ds_vals = ablation["dsafe_values"]

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL * 0.78, 2.8))
    pairs = [
        ("collision_rate_grid", "Collision Rate (CR)", "YlOrRd"),
        ("success_rate_grid", "Success Rate (SR)", "YlGn"),
    ]

    for ax, (key, title, cmap) in zip(axes, pairs):
        grid = np.array([[v if v is not None else np.nan for v in row]
                         for row in ablation[key]])
        im = ax.imshow(grid, cmap=cmap, aspect="auto", origin="lower",
                       vmin=0, vmax=1)
        ax.set_xticks(range(len(ds_vals)))
        ax.set_xticklabels([f"{v}" for v in ds_vals])
        ax.set_yticks(range(len(ks_vals)))
        ax.set_yticklabels([f"{v}" for v in ks_vals])
        ax.set_xlabel(r"$d_{\mathrm{safe}}$ (m)")
        ax.set_ylabel(r"$k_s$")
        ax.set_title(title)
        cb = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.03)
        cb.ax.tick_params(labelsize=6)
        # Annotate cells
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if not np.isnan(grid[i, j]):
                    tc = "white" if grid[i, j] > 0.55 else "black"
                    ax.text(j, i, f"{grid[i,j]:.2f}", ha="center", va="center",
                            fontsize=5.5, color=tc, fontweight="bold")
        # Mark default params
        di = ks_vals.index(0.5) if 0.5 in ks_vals else -1
        dj = ds_vals.index(0.8) if 0.8 in ds_vals else -1
        if di >= 0 and dj >= 0:
            ax.plot(dj, di, "k*", ms=10, markeredgewidth=0.5)

    fig.tight_layout(pad=0.8)
    _save(fig, "fig4_ablation_heatmaps")


# ═══════════════════════════════════════════════════════════════════════════
# Fig 5: Safety-performance Pareto front — single column
# ═══════════════════════════════════════════════════════════════════════════

def fig5_pareto(stats):
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, 2.2), sharey=True)
    methods = [m for m in METHODS if m in stats["per_method"]]

    for ax, sc in zip(axes, SCENARIOS):
        for method in methods:
            d = stats["per_method"].get(method, {}).get(sc, {})
            cr = d.get("collision_rate", {}).get("mean")
            sr = d.get("success_rate", {}).get("mean")
            if cr is None or sr is None:
                continue
            cr_std = d.get("collision_rate", {}).get("std", 0)
            sr_std = d.get("success_rate", {}).get("std", 0)
            ax.errorbar(
                cr, sr, xerr=cr_std, yerr=sr_std,
                fmt=MARKERS[method], ms=6, color=COLORS[method],
                elinewidth=0.5, capsize=1.5, capthick=0.4,
                label=LABELS[method], zorder=3,
                markeredgecolor="white", markeredgewidth=0.3,
            )
        ax.set_xlabel("Collision Rate ↓")
        if ax == axes[0]:
            ax.set_ylabel("Success Rate ↑")
        ax.set_title(SC_LABELS[sc])
        ax.grid(linewidth=0.3, alpha=0.3)
        # Ideal corner shading
        ax.axvspan(ax.get_xlim()[0], 0.05, alpha=0.03, color="green", zorder=0)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(methods),
               frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, -0.05))
    fig.subplots_adjust(bottom=0.26, wspace=0.08)
    _save(fig, "fig5_pareto")


# ═══════════════════════════════════════════════════════════════════════════
# Fig 6: Gazebo results bar chart — single column
# ═══════════════════════════════════════════════════════════════════════════

def fig6_gazebo(gazebo_analysis):
    summaries = gazebo_analysis.get("summaries", {})
    if not summaries:
        print("  ⚠ No Gazebo analysis data")
        return

    methods = [m for m in METHODS if m in summaries]
    metrics = [
        ("success_rate", "SR", "↑"),
        ("collision_rate", "CR", "↓"),
        ("clearance", "Clearance (m)", "↑"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL * 0.85, 2.2))
    x = np.arange(len(methods))

    for ax, (metric, ylabel, arrow) in zip(axes, metrics):
        means = [summaries[m][metric]["mean"] for m in methods]
        stds = [summaries[m][metric]["std"] for m in methods]
        colors = [COLORS[m] for m in methods]

        bars = ax.bar(x, means, 0.7, yerr=stds, color=colors,
                      edgecolor="white", linewidth=0.3,
                      capsize=2, error_kw=dict(elinewidth=0.5, capthick=0.4),
                      zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels([LABELS[m].replace(" (ours)", "") for m in methods],
                           rotation=35, ha="right", fontsize=6.5)
        ax.set_ylabel(f"{ylabel} {arrow}")
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))
        ax.grid(axis="y", linewidth=0.3, alpha=0.4, zorder=0)

        # Highlight best
        if arrow == "↑":
            best_idx = np.argmax(means)
        else:
            best_idx = np.argmin(means)
        bars[best_idx].set_edgecolor("black")
        bars[best_idx].set_linewidth(1.2)

    fig.suptitle("Gazebo TurtleBot3 Validation (5 seeds × 20 eps)", fontsize=8, y=0.98)
    fig.tight_layout(pad=0.8)
    fig.subplots_adjust(top=0.88)
    _save(fig, "fig6_gazebo")


# ═══════════════════════════════════════════════════════════════════════════
# Fig 7: Clearance CDF — single column
# ═══════════════════════════════════════════════════════════════════════════

def fig7_clearance_cdf(stats, scenario="complex"):
    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.4))
    methods = [m for m in METHODS if m in stats["per_method"]]

    for method in methods:
        vals = (stats["per_method"].get(method, {})
                .get(scenario, {})
                .get("avg_min_clearance", {})
                .get("values", []))
        if not vals:
            continue
        sv = np.sort(vals)
        cdf = np.arange(1, len(sv) + 1) / len(sv)
        ax.step(sv, cdf, where="post", color=COLORS[method],
                lw=1.2, label=LABELS[method])

    ax.axvline(0.18, color="gray", ls="--", lw=0.7, alpha=0.7)
    ax.annotate("Robot radius", xy=(0.20, 0.3), fontsize=6, color="gray")
    ax.set_xlabel("Minimum Clearance (m)")
    ax.set_ylabel("CDF")
    ax.set_title(f"Clearance Distribution — {SC_LABELS[scenario]}")
    ax.set_xlim(0, 1.2)
    ax.legend(frameon=False, fontsize=6.5, loc="lower right")
    ax.grid(linewidth=0.3, alpha=0.3)
    fig.tight_layout(pad=0.5)
    _save(fig, "fig7_clearance_cdf")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("Generating figures...")
    print(f"Output: {OUT_DIR}/\n")

    # Load data
    stats = load_json("results/statistics.json")
    curves = load_json("results/training_curves.json")
    ablation = load_json("results/ablation_results.json")

    gazebo_path = "results/gazebo/analysis.json"
    gazebo = load_json(gazebo_path) if os.path.exists(gazebo_path) else {}

    # Generate all figures
    fig1_main_comparison(stats)
    fig2_training_curves(curves)
    fig3_scaling_functions()
    fig4_ablation_heatmaps(ablation)
    fig5_pareto(stats)
    fig6_gazebo(gazebo)
    fig7_clearance_cdf(stats)

    n_files = len([f for f in os.listdir(OUT_DIR) if f.endswith(('.pdf', '.png'))])
    print(f"\nDone! Generated {n_files} files in {OUT_DIR}/")


if __name__ == "__main__":
    main()
 
