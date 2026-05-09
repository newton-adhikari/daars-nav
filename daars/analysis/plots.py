"""
Figures from the daars evaluation data.
"""

from __future__ import annotations

import os
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times", "Times New Roman", "DejaVu Serif"],
    "font.size":          8,
    "axes.labelsize":     9,
    "axes.titlesize":     9,
    "axes.titleweight":   "bold",
    "xtick.labelsize":    7,
    "ytick.labelsize":    7,
    "legend.fontsize":    7,
    "legend.handlelength": 1.5,
    "figure.dpi":         300,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.05,
    "text.usetex":        False,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          False,
    "axes.linewidth":     0.6,
    "xtick.major.width":  0.5,
    "ytick.major.width":  0.5,
    "lines.linewidth":    1.2,
})

SINGLE_COL = 3.5    # inches
DOUBLE_COL = 7.16   # inches

COLORS = {
    "alpha_only": "#D62728",   # red — our main method
    "daars":      "#E377C2",   # pink — dual modulation variant
    "static":     "#1F77B4",   # blue
    "beta_only":  "#8C564B",   # brown
    "ppo_lag":    "#2CA02C",   # green
}

LABELS = {
    "alpha_only": "PARS (ours)",
    "daars":      "DAARS (α+β)",
    "static":     "Static",
    "beta_only":  r"$\beta$-only",
    "ppo_lag":    "PPO-Lag",
}

HATCHES = {
    "alpha_only": "",
    "daars":      "//",
    "static":     "\\\\",
    "beta_only":  "xx",
    "ppo_lag":    "..",
}

METHOD_ORDER = ["alpha_only", "daars", "static", "beta_only", "ppo_lag"]
ABLATION_ORDER = ["alpha_only", "daars", "static", "beta_only", "ppo_lag"]
SCENARIOS = ["simple", "complex", "dynamic"]
SCENARIO_LABELS = {"simple": "Simple", "complex": "Complex", "dynamic": "Dynamic"}


def _save(fig, name: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"))
    plt.close(fig)



# Fig 1 — Main comparison: 4 subplots (one per metric), grouped bars

def plot_main_comparison(stats: dict, out_dir: str):
    metrics = [
        ("success_rate",        "Success Rate",       (0.55, 1.02)),
        ("collision_rate",      "Collision Rate",     (-0.02, 0.30)),
        ("avg_min_clearance",   "Min. Clearance (m)", (0.5, 1.7)),
        ("constraint_violation_rate", "CVR",          (-0.02, 0.60)),
    ]
    methods = [m for m in METHOD_ORDER if m in stats["per_method"]]
    n_methods = len(methods)

    fig, axes = plt.subplots(1, 4, figsize=(DOUBLE_COL, 2.8))
    x = np.arange(len(SCENARIOS))
    w = 0.75 / n_methods

    for ax, (metric, ylabel, ylim) in zip(axes, metrics):
        for i, method in enumerate(methods):
            offset = (i - (n_methods - 1) / 2) * w
            means, stds = [], []
            for sc in SCENARIOS:
                d = stats["per_method"].get(method, {}).get(sc, {}).get(metric, {})
                means.append(d.get("mean", 0))
                stds.append(d.get("std", 0))
            bars = ax.bar(
                x + offset, means, w * 0.88, yerr=stds,
                color=COLORS.get(method, "#aaa"),
                hatch=HATCHES.get(method, ""),
                edgecolor="white", linewidth=0.4,
                capsize=1.5, error_kw=dict(elinewidth=0.6, capthick=0.5),
                label=LABELS.get(method, method),
            )
        ax.set_xticks(x)
        ax.set_xticklabels([SCENARIO_LABELS[s] for s in SCENARIOS])
        ax.set_ylabel(ylabel)
        ax.set_ylim(ylim)
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))
        ax.grid(axis="y", linewidth=0.3, alpha=0.4)

    # Single legend below the figure
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=n_methods,
               frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, -0.02))
    fig.subplots_adjust(bottom=0.25, wspace=0.45)
    _save(fig, "fig1_main_comparison", out_dir)



# Fig 2 — Training curves with CI shading

def plot_training_curves(training_curves: dict, out_dir: str):
    curve_keys = ["rewards",    "successes",     "collisions"]
    ylabels    = ["Episode Reward", "Success Rate", "Collision Rate"]
    methods = [m for m in METHOD_ORDER if m in training_curves]
    window = 80

    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, 2.2))

    for ax, key, ylabel in zip(axes, curve_keys, ylabels):
        for method in methods:
            seeds_data = training_curves.get(method, [])
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
               frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, -0.02))
    fig.subplots_adjust(bottom=0.22, wspace=0.38)
    _save(fig, "fig2_training_curves", out_dir)



# Fig 3 — α and β scaling functions (matches actual code values)

def plot_scaling_functions(config: dict | None, out_dir: str):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE_COL * 0.75, 2.4))
    d = np.linspace(0.01, 2.5, 500)

    # α(d) with floor = 0.3
    ALPHA_MIN = 0.3
    for ds, ls, c in zip(
        [0.5, 0.8, 1.0, 1.5],
        ["-", "--", "-.", ":"],
        ["#1f77b4", "#ff7f0e", "#d62728", "#2ca02c"],
    ):
        a = np.clip(d / ds, ALPHA_MIN, 1.0)
        ax1.plot(d, a, ls=ls, lw=1.3, color=c, label=f"$d_{{safe}}={ds}$")
    ax1.axhline(ALPHA_MIN, color="gray", ls=":", lw=0.7, alpha=0.6)
    ax1.annotate(r"$\alpha_{\min}=0.3$", xy=(2.0, ALPHA_MIN),
                 fontsize=6.5, color="gray", va="bottom")
    ax1.set_xlabel(r"$d_{\mathrm{obs}}$ (m)")
    ax1.set_ylabel(r"$\alpha(d_{\mathrm{obs}})$")
    ax1.set_ylim(-0.02, 1.12)
    ax1.legend(frameon=False, fontsize=6.5, loc="lower right")
    ax1.grid(linewidth=0.3, alpha=0.3)

    # β(d) with cap = 5.0
    BETA_MAX = 5.0
    eps = 0.1
    for ks, ls, c in zip(
        [0.3, 0.5, 0.8, 1.2],
        ["-", "--", "-.", ":"],
        ["#1f77b4", "#ff7f0e", "#d62728", "#2ca02c"],
    ):
        b = np.exp(ks / (d + eps)) - np.exp(ks / (d + eps + 2.0))
        b = np.minimum(b, BETA_MAX)
        ax2.plot(d, b, ls=ls, lw=1.3, color=c, label=f"$k_s={ks}$")
    ax2.axhline(BETA_MAX, color="gray", ls=":", lw=0.7, alpha=0.6)
    ax2.annotate(r"$\beta_{\max}=5.0$", xy=(1.8, BETA_MAX),
                 fontsize=6.5, color="gray", va="bottom")
    ax2.set_xlabel(r"$d_{\mathrm{obs}}$ (m)")
    ax2.set_ylabel(r"$\beta(d_{\mathrm{obs}})$")
    ax2.set_ylim(-0.2, 6.0)
    ax2.legend(frameon=False, fontsize=6.5, loc="upper right")
    ax2.grid(linewidth=0.3, alpha=0.3)

    fig.tight_layout(pad=0.8)
    _save(fig, "fig3_scaling_functions", out_dir)




# Fig 4 — Ablation heatmaps (k_s × d_safe)

def plot_ablation_heatmaps(ablation: dict, out_dir: str):
    if not ablation:
        return
    ks_vals = ablation.get("ks_values", [])
    dsafe_vals = ablation.get("dsafe_values", [])
    if not ks_vals or not dsafe_vals:
        return

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL * 0.78, 3.2))
    pairs = [
        ("collision_rate_grid", "Collision Rate", "YlOrRd"),
        ("success_rate_grid",   "Success Rate",   "YlGn"),
    ]

    for ax, (key, title, cmap) in zip(axes, pairs):
        grid = ablation.get(key)
        if grid is None:
            continue
        g = np.array(
            [[v if v is not None else np.nan for v in row] for row in grid]
        )
        im = ax.imshow(g, cmap=cmap, aspect="auto", origin="lower",
                        vmin=0, vmax=1)
        ax.set_xticks(range(len(dsafe_vals)))
        ax.set_xticklabels([f"{v}" for v in dsafe_vals])
        ax.set_yticks(range(len(ks_vals)))
        ax.set_yticklabels([f"{v}" for v in ks_vals])
        ax.set_xlabel(r"$d_{\mathrm{safe}}$ (m)")
        ax.set_ylabel(r"$k_s$")
        ax.set_title(title)
        cb = plt.colorbar(im, ax=ax, shrink=0.82, pad=0.04)
        cb.ax.tick_params(labelsize=6)
        # Cell annotations
        for i in range(g.shape[0]):
            for j in range(g.shape[1]):
                if not np.isnan(g[i, j]):
                    txt_color = "white" if g[i, j] > 0.6 else "black"
                    ax.text(j, i, f"{g[i, j]:.2f}", ha="center", va="center",
                            fontsize=6, fontweight="bold", color=txt_color)

    fig.tight_layout(pad=0.8)
    _save(fig, "fig4_ablation_heatmaps", out_dir)



# Fig 5 — Safety–performance trade-off scatter

def plot_safety_performance_tradeoff(stats: dict, out_dir: str):
    markers = {"simple": "o", "complex": "s", "dynamic": "^"}
    methods = [m for m in METHOD_ORDER if m in stats["per_method"]]

    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, 2.4),
                              sharey=True, sharex=True)

    for ax, sc in zip(axes, SCENARIOS):
        for method in methods:
            d = stats["per_method"].get(method, {}).get(sc, {})
            cr = d.get("collision_rate", {}).get("mean", None)
            sr = d.get("success_rate", {}).get("mean", None)
            cr_std = d.get("collision_rate", {}).get("std", 0)
            sr_std = d.get("success_rate", {}).get("std", 0)
            if cr is None or sr is None:
                continue
            ax.errorbar(
                cr, sr, xerr=cr_std, yerr=sr_std,
                fmt=markers[sc], ms=6, color=COLORS.get(method, "#aaa"),
                elinewidth=0.6, capsize=2, capthick=0.5,
                label=LABELS.get(method, method), zorder=3,
            )
        ax.set_xlabel("Collision Rate ↓")
        if ax == axes[0]:
            ax.set_ylabel("Success Rate ↑")
        ax.set_title(SCENARIO_LABELS[sc])
        ax.set_xlim(-0.03, 0.35)
        ax.set_ylim(0.55, 1.03)
        ax.grid(linewidth=0.3, alpha=0.3)
        # Ideal region shading
        ax.axvspan(-0.03, 0.05, alpha=0.04, color="green")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(methods),
               frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, -0.03))
    fig.subplots_adjust(bottom=0.24, wspace=0.08)
    _save(fig, "fig5_safety_performance_tradeoff", out_dir)



# Fig 6 — Component ablation (α-only, β-only, static, DAARS)

def plot_ablation_components(stats: dict, out_dir: str):
    metrics = [
        ("collision_rate",            "Collision Rate",     (-0.01, 0.22)),
        ("avg_min_clearance",         "Min. Clearance (m)", (0.55, 1.6)),
        ("constraint_violation_rate", "CVR",                (0.0, 0.50)),
    ]
    methods = [m for m in ABLATION_ORDER if m in stats["per_method"]]
    n_methods = len(methods)

    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, 2.4))
    x = np.arange(len(SCENARIOS))
    w = 0.75 / n_methods

    for ax, (metric, ylabel, ylim) in zip(axes, metrics):
        for i, method in enumerate(methods):
            offset = (i - (n_methods - 1) / 2) * w
            means, stds = [], []
            for sc in SCENARIOS:
                d = stats["per_method"].get(method, {}).get(sc, {}).get(metric, {})
                means.append(d.get("mean", 0))
                stds.append(d.get("std", 0))
            ax.bar(
                x + offset, means, w * 0.88, yerr=stds,
                color=COLORS.get(method, "#aaa"),
                hatch=HATCHES.get(method, ""),
                edgecolor="white", linewidth=0.4,
                capsize=1.5, error_kw=dict(elinewidth=0.6, capthick=0.5),
                label=LABELS.get(method, method),
            )
        ax.set_xticks(x)
        ax.set_xticklabels([SCENARIO_LABELS[s] for s in SCENARIOS])
        ax.set_ylabel(ylabel)
        ax.set_ylim(ylim)
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))
        ax.grid(axis="y", linewidth=0.3, alpha=0.4)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=n_methods,
               frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, -0.02))
    fig.subplots_adjust(bottom=0.22, wspace=0.40)
    _save(fig, "fig6_ablation_components", out_dir)



# Fig 7 — Min-clearance CDF

def plot_clearance_cdf(stats: dict, out_dir: str, scenario="complex"):
    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.4))
    methods = [m for m in METHOD_ORDER if m in stats["per_method"]]

    for method in methods:
        vals = (stats["per_method"].get(method, {})
                .get(scenario, {})
                .get("avg_min_clearance", {})
                .get("values", []))
        if not vals:
            continue
        sv = np.sort(vals)
        cdf = np.arange(1, len(sv) + 1) / len(sv)
        ax.step(sv, cdf, where="post", color=COLORS.get(method, "#aaa"),
                lw=1.3, label=LABELS.get(method, method))

    ax.axvline(0.18, color="red", ls="--", lw=0.8, alpha=0.7)
    ax.annotate("Robot\nradius", xy=(0.18, 0.5), xytext=(0.35, 0.45),
                fontsize=6, color="red", alpha=0.8,
                arrowprops=dict(arrowstyle="->", color="red", lw=0.6))
    ax.set_xlabel("Min. Clearance (m)")
    ax.set_ylabel("CDF")
    ax.set_title(f"Clearance Distribution ({scenario.capitalize()})")
    ax.set_xlim(0, 1.8)
    ax.legend(frameon=False, fontsize=6.5, loc="lower right")
    ax.grid(linewidth=0.3, alpha=0.3)
    fig.tight_layout(pad=0.5)
    _save(fig, f"fig7_clearance_cdf_{scenario}", out_dir)



# The main entry function

def generate_all_figures(
    stats: dict,
    training_curves: dict,
    ablation_results: dict,
    out_dir: str = "figures",
    config: dict | None = None,
):
    print(f"\n>> Generating figures → {out_dir}/")
    plot_main_comparison(stats, out_dir)
    plot_training_curves(training_curves, out_dir)
    plot_scaling_functions(config, out_dir)
    plot_ablation_heatmaps(ablation_results, out_dir)
    plot_safety_performance_tradeoff(stats, out_dir)
    plot_ablation_components(stats, out_dir)
    plot_clearance_cdf(stats, out_dir, scenario="complex")
    print(f"  Saved {len(os.listdir(out_dir))} files.")
 