"""Figures for the E4 report section, drawn from the sweep CSVs in results/.

    python report/e4/make_figures.py

Writes vector PDFs to report/e4/figures/. Sized for IEEE columns (3.45 in single, 7.1 in double).
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, LogNorm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments/e4_workload_selector"))
from harness import configs as cfgmod  # noqa: E402
from selector import load_latency  # noqa: E402

OUT = Path(__file__).resolve().parent / "figures"
COL, WIDE = 3.45, 7.1
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#dddcd6"
SEQ = LinearSegmentedColormap.from_list("blue", ["#eef5fd", "#b7d3f6", "#5598e7", "#1c5cab", "#0d366b"])
SCEN = {"natural": "Unmodified model", "break_gelu": "GELU in PyTorch", "break_layernorm": "LayerNorm in PyTorch"}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
    "lines.linewidth": 1.4, "lines.markersize": 4.5, "legend.frameon": False,
    "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name)
    plt.close(fig)
    print("wrote", OUT / name)


def gpu_bound(config_id):
    """One engine, or everything captured in a CUDA graph: no Python between GPU kernels."""
    return config_id.startswith("trt_") or config_id.endswith("cudagraphs")


def fig_warmup():
    """Each timing repeat relative to the fastest repeat of the same run, before and after the fix."""
    fig, axes = plt.subplots(1, 2, figsize=(COL, 2.05), sharey=True, gridspec_kw={"width_ratios": [3, 5]})
    rng = np.random.default_rng(0)
    panels = [("results/sweep/sweep_v1_short_warmup.csv", "(a) Sweep 1"),
              ("results/sweep/sweep.csv", "(b) Sweep 2")]
    for ax, (path, title) in zip(axes, panels):
        d = pd.read_csv(ROOT / path)
        d = d[(d["status"] == "ok") & (d["split"] == "grid") & (d["batch"] <= 4)]
        d = d[~((d["batch"] == 4) & (d["seq_len"] == 512))]  # the shapes both sweeps cover
        d["rel"] = d["median_ms"] / d.groupby(["config_id", "batch", "seq_len"])["median_ms"].transform("min")
        for flag, color, marker, label in [(False, ORANGE, "^", "CPU-bound"), (True, BLUE, "o", "GPU-bound")]:
            g = d[d["config_id"].map(gpu_bound) == flag]
            x = g["repeat"] + 1 + rng.uniform(-0.22, 0.22, len(g)) + (0.0 if flag else 0.0)
            ax.scatter(x, g["rel"], s=9, color=color, marker=marker, alpha=0.75, linewidths=0, label=label)
        ax.set_title(title, loc="left")
        ax.set_xticks(range(1, d["repeat"].max() + 2))
        ax.set_xlabel("Timing repeat")
        ax.set_ylim(0.98, 1.30)
    axes[0].set_ylabel("Latency / fastest repeat")
    axes[1].legend(loc="upper right", handletextpad=0.2, borderaxespad=0.1)
    fig.subplots_adjust(wspace=0.08)
    save(fig, "fig_warmup.pdf")


def fig_latency(lat):
    """Latency against batch size at 512 tokens, one panel per scenario."""
    series = {
        "natural": [("trt_default", "Default", BLUE, "o"), ("trt_opt1", "Opt. level 1", AQUA, "s"),
                    ("trt_cudagraphs", "CUDA Graphs", ORANGE, "^")],
        "break_gelu": [("gelu_mbs5", "Default", BLUE, "o"), ("gelu_mbs1", "min_block_size 1", AQUA, "s"),
                       ("gelu_cudagraphs", "CUDA Graphs", ORANGE, "^")],
        "break_layernorm": [("ln_mbs5", "Default", BLUE, "o"), ("ln_mbs1", "min_block_size 1", AQUA, "s"),
                            ("ln_cudagraphs", "CUDA Graphs", ORANGE, "^")],
    }
    fig, axes = plt.subplots(1, 3, figsize=(WIDE, 2.15), sharey=True)
    d = lat[(lat["split"] == "grid") & (lat["seq_len"] == 512)]
    for ax, (scen, items) in zip(axes, series.items()):
        e = d[d["config_id"] == "eager"].sort_values("batch")
        ax.plot(e["batch"], e["latency_ms"], color=MUTED, linestyle=(0, (4, 2)), marker="D", markersize=3.5,
                label="PyTorch eager")
        for config_id, label, color, marker in items:
            g = d[d["config_id"] == config_id].sort_values("batch")
            ax.plot(g["batch"], g["latency_ms"], color=color, marker=marker, markeredgecolor="white",
                    markeredgewidth=0.6, label=label)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks([1, 4, 8, 16], ["1", "4", "8", "16"])
        ax.set_yticks([0.5, 1, 2, 4, 8], ["0.5", "1", "2", "4", "8"])
        ax.minorticks_off()
        ax.set_title(SCEN[scen], loc="left")
        ax.set_xlabel("Batch size (512 tokens per sequence)")
        ax.legend(loc="lower right", handlelength=1.8, borderaxespad=0.2, labelspacing=0.25)
    axes[0].set_ylabel("Median latency (ms)")
    fig.subplots_adjust(wspace=0.06)
    save(fig, "fig_latency.pdf")


def fig_gain(lat, spec):
    """Speedup of the best tested config over the scenario's default, on the training grid."""
    fig, axes = plt.subplots(1, 3, figsize=(WIDE, 1.75), sharey=True)
    norm = LogNorm(1.0, 5.2)
    for ax, scen in zip(axes, SCEN):
        d = lat[(lat["split"] == "grid") & lat["config_id"].isin(cfgmod.scenario_configs(spec, scen))]
        wide = d.pivot_table(index="batch", columns="seq_len", values="latency_ms", aggfunc="min")
        base = d[d["config_id"] == cfgmod.default_config(spec, scen)].pivot(
            index="batch", columns="seq_len", values="latency_ms")
        gain = (base / wide).sort_index(ascending=False)
        im = ax.imshow(gain.to_numpy(), cmap=SEQ, norm=norm, aspect="auto")
        for i in range(gain.shape[0]):
            for j in range(gain.shape[1]):
                v = gain.iat[i, j]
                ax.text(j, i, f"{v:.2f}$\\times$", ha="center", va="center", fontsize=7,
                        color="white" if v > 2.0 else INK)
        ax.set_xticks(range(gain.shape[1]), gain.columns)
        ax.set_yticks(range(gain.shape[0]), gain.index)
        ax.set_xticks(np.arange(-0.5, gain.shape[1]), minor=True)
        ax.set_yticks(np.arange(-0.5, gain.shape[0]), minor=True)
        ax.grid(False)
        ax.grid(which="minor", color="white", linewidth=1.5)
        ax.tick_params(which="both", length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(SCEN[scen], loc="left")
        ax.set_xlabel("Sequence length (tokens)")
    axes[0].set_ylabel("Batch size")
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.015, ticks=[1, 2, 3, 5])
    cb.ax.set_yticklabels(["1", "2", "3", "5"])
    cb.ax.minorticks_off()
    cb.outline.set_visible(False)
    cb.set_label("Best / default", fontsize=7)
    save(fig, "fig_gain.pdf")


METHODS = [("fixed_tuned", "One fixed tuned config", BLUE, "o"), ("lookup", "Lookup selector", ORANGE, "^"),
           ("regression", "Regression selector", AQUA, "s")]


def fig_regret(summary, noise):
    """Held-out regret of the three data-driven methods."""
    fig, ax = plt.subplots(figsize=(COL, 2.0))
    h = summary[summary["eval"] == "heldout"]
    width = 0.24
    for k, (method, label, color, _) in enumerate(METHODS):
        vals = [float(h[(h["scenario"] == s) & (h["method"] == method)]["regret_pct"].iloc[0]) for s in SCEN]
        x = np.arange(len(SCEN)) + (k - 1) * (width + 0.02)
        ax.bar(x, vals, width, color=color, label=label)
        for xi, v in zip(x, vals):
            ax.text(xi, v + 0.08, f"{v:.1f}", ha="center", va="bottom", fontsize=7, color=INK)
    floor = float(noise["repeat_pct"].iloc[0])
    ax.axhline(floor, color=MUTED, linewidth=0.8, linestyle=(0, (4, 2)))
    ax.text(-0.46, floor + 0.05, "timing noise", ha="left", va="bottom", fontsize=7, color=MUTED)
    ax.set_xticks(range(len(SCEN)), ["Unmodified\nmodel", "GELU in\nPyTorch", "LayerNorm in\nPyTorch"])
    ax.set_ylabel("Extra latency over\nbest tested config (%)")
    ax.set_ylim(0, 4.2)
    ax.grid(axis="x", visible=False)
    ax.set_xlim(-0.5, 2.5)
    ax.legend(loc="upper right", borderaxespad=0.1, labelspacing=0.25, handlelength=1.2)
    save(fig, "fig_regret.pdf")


def fig_budget(budget):
    """Regret on the held-out shapes when only k grid shapes were measured."""
    fig, axes = plt.subplots(1, 3, figsize=(WIDE, 2.0), sharey=True)
    for ax, scen in zip(axes, SCEN):
        d = budget[budget["scenario"] == scen]
        for method, label, color, marker in METHODS:
            g = d[d["method"] == method].sort_values("k")
            ax.plot(g["k"], g["regret_pct"], color=color, marker=marker, markeredgecolor="white",
                    markeredgewidth=0.6, label=label)
        ax.set_title(SCEN[scen], loc="left")
        ax.set_xticks([1, 2, 4, 6, 8, 10, 12])
        ax.set_xlabel("Training shapes measured")
        ax.set_ylim(-0.5, 16.5)
        ax.set_yticks([0, 5, 10, 15])
    axes[0].set_ylabel("Extra latency over\nbest tested config (%)")
    axes[2].legend(loc="upper right", borderaxespad=0.1, labelspacing=0.25)
    fig.subplots_adjust(wspace=0.06)
    save(fig, "fig_budget.pdf")


def fig_choice(per):
    """Which config is fastest at each shape, and where the lookup selector misses."""
    fig, axes = plt.subplots(1, 2, figsize=(COL, 2.25), sharey=True)
    for ax, scen in zip(axes, ["break_gelu", "break_layernorm"]):
        d = per[(per["scenario"] == scen) & (per["method"] == "lookup")]
        for is_eager, color, marker, label in [(False, ORANGE, "o", "CUDA Graphs fastest"),
                                               (True, MUTED, "s", "PyTorch eager fastest")]:
            g = d[(d["oracle_config"] == "eager") == is_eager]
            held = g["eval"] == "heldout"
            ax.scatter(g[~held]["seq_len"], g[~held]["batch"], s=26, color=color, marker=marker,
                       linewidths=0, label=label)
            ax.scatter(g[held]["seq_len"], g[held]["batch"], s=26, color=color, marker=marker,
                       edgecolors=INK, linewidths=0.9)
        miss = d[(d["eval"] == "heldout") & (d["config_id"] != d["oracle_config"])]
        ax.scatter(miss["seq_len"], miss["batch"], s=120, facecolors="none", edgecolors=BLUE, linewidths=1.1,
                   label="Lookup selector misses")
        for _, r in miss.iterrows():
            ax.annotate(f"+{(r['latency_ms'] / r['oracle_ms'] - 1) * 100:.0f}%", (r["seq_len"], r["batch"]),
                        xytext=(-9, 5 if r["batch"] == 16 else -4), textcoords="offset points", ha="right",
                        va="center", fontsize=7, color=INK)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=2)
        ax.set_xticks([128, 256, 512], ["128", "256", "512"])
        ax.set_yticks([1, 2, 4, 8, 16], ["1", "2", "4", "8", "16"])
        ax.minorticks_off()
        ax.set_xlim(105, 640)
        ax.set_ylim(0.75, 22)
        ax.set_title(SCEN[scen], loc="left")
        ax.set_xlabel("Sequence length (tokens)")
    axes[0].set_ylabel("Batch size")
    handles, labels = axes[1].get_legend_handles_labels()
    handles.insert(2, plt.Line2D([], [], marker="o", linestyle="", markersize=5, markerfacecolor="white",
                                 markeredgecolor=INK, markeredgewidth=0.9))
    labels.insert(2, "Dark outline: held-out shape")
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.06), labelspacing=0.3,
               handletextpad=0.3, columnspacing=1.0, markerscale=0.7)
    fig.subplots_adjust(wspace=0.08)
    save(fig, "fig_choice.pdf")


def main():
    spec = cfgmod.load_spec()
    lat = load_latency(ROOT / "results/sweep/sweep.csv")
    res = ROOT / "results/e4_workload_selector"
    fig_warmup()
    fig_latency(lat)
    fig_gain(lat, spec)
    fig_regret(pd.read_csv(res / "summary.csv"), pd.read_csv(res / "noise.csv"))
    fig_budget(pd.read_csv(res / "budget.csv"))
    fig_choice(pd.read_csv(res / "per_workload.csv"))


if __name__ == "__main__":
    main()
