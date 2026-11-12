"""Report figures for E4. Every figure has a table twin in summary.csv / budget.csv / per_workload.csv."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SURFACE, INK, INK_2, RULE = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE = "#2a78d6"
SERIES = {"fixed_tuned": "#2a78d6", "lookup": "#eb6834", "regression": "#1baf7a"}  # fixed order
LABEL = {"eager": "PyTorch eager", "default": "Default Torch-TensorRT", "fixed_tuned": "One fixed tuned config",
         "lookup": "Lookup selector", "regression": "Regression selector", "oracle": "Best tested config"}
REGRET_RAMP = LinearSegmentedColormap.from_list("regret", [SURFACE, "#cde2fb", "#86b6ef", "#5598e7"])
EVAL_NAME = {"heldout": "held-out shapes", "loo": "leave-one-out over the grid"}


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(RULE)
    ax.tick_params(colors=INK_2, length=0, labelsize=9)
    ax.xaxis.label.set_color(INK_2)
    ax.yaxis.label.set_color(INK_2)


def _save(fig, path):
    fig.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return path


def baselines(summary, scenario, eval_name, path):
    """Extra latency over the best tested config, per method. Bars start at zero (= oracle)."""
    s = summary[(summary["scenario"] == scenario) & (summary["eval"] == eval_name)].set_index("method")
    methods = [m for m in LABEL if m in s.index][::-1]
    values = s.loc[methods, "regret_pct"].to_numpy()
    fig, ax = plt.subplots(figsize=(7, 0.5 * len(methods) + 1.3))
    fig.patch.set_facecolor(SURFACE)
    ax.barh([LABEL[m] for m in methods], values, height=0.55, color=BLUE)
    span = max(values.max(), 1.0)
    for y, v in enumerate(values):
        ax.text(v + span * 0.015, y, f"+{v:.1f}%", va="center", ha="left", color=INK, fontsize=9)
    ax.set_xlim(0, span * 1.15)
    ax.set_xlabel("Extra latency over the best tested config (geometric mean, %)")
    ax.set_title(f"How far each method is from the best tested config\n{scenario}, {EVAL_NAME[eval_name]}",
                 loc="left", color=INK, fontsize=11)
    ax.xaxis.grid(True, color=RULE, linewidth=0.8)
    ax.set_axisbelow(True)
    _style(ax)
    return _save(fig, path)


def budget(budget_df, scenario, path):
    """Selector regret against what it cost to collect the tuning measurements."""
    b = budget_df[budget_df["scenario"] == scenario]
    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor(SURFACE)
    ends = []
    for method, color in SERIES.items():
        m = b[b["method"] == method].sort_values("k")
        x = m["tuning_cost_s"] / 60
        ax.plot(x, m["regret_pct"], color=color, linewidth=2, marker="o", markersize=5,
                markeredgecolor=SURFACE, markeredgewidth=1, label=LABEL[method])
        ends.append((float(m["regret_pct"].iloc[-1]), float(x.iloc[-1]), method))
    # Direct labels at the line ends, pushed apart so they never overlap.
    top = max(ax.get_ylim()[1], 1.0)
    gap, last = top * 0.06, -np.inf
    for y, x, method in sorted(ends):
        y_text = max(y, last + gap)
        last = y_text
        ax.annotate(LABEL[method], (x, y), xytext=(8, 0), textcoords="offset points",
                    va="center", color=INK, fontsize=9) if y_text == y else \
            ax.annotate(LABEL[method], (x, y), xytext=(x * 1.02, y_text), textcoords="data",
                        va="center", color=INK, fontsize=9, arrowprops={"arrowstyle": "-", "color": RULE})
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0, right=ax.get_xlim()[1] * 1.3)
    ax.set_xlabel("Tuning cost: GPU minutes spent measuring configs")
    ax.set_ylabel("Extra latency over best tested config (%)")
    ax.set_title(f"Selector quality against tuning budget\n{scenario}; each point adds one measured workload",
                 loc="left", color=INK, fontsize=11)
    ax.yaxis.grid(True, color=RULE, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="upper right")
    _style(ax)
    return _save(fig, path)


def choice_map(per, scenario, method, path):
    """What the selector chose for every workload and what it cost, on a batch × length grid."""
    p = per[(per["scenario"] == scenario) & (per["method"] == method)]
    batches, seqs = sorted(p["batch"].unique()), sorted(p["seq_len"].unique())
    regret = (p["latency_ms"] / p["oracle_ms"] - 1) * 100
    vmax = max(5.0, float(np.nanmax(regret)))
    fig, ax = plt.subplots(figsize=(1.55 * len(seqs) + 1.2, 0.75 * len(batches) + 1.6))
    fig.patch.set_facecolor(SURFACE)
    for (_, row), r in zip(p.iterrows(), regret):
        x, y = seqs.index(row["seq_len"]), batches.index(row["batch"])
        held = row["eval"] == "heldout"
        fill = REGRET_RAMP(0 if np.isnan(r) else r / vmax)
        ax.add_patch(plt.Rectangle((x - 0.47, y - 0.44), 0.94, 0.88, facecolor=fill,
                                   edgecolor=INK if held else RULE, linewidth=1.6 if held else 0.8))
        lines = [row["config_id"], "failed here" if np.isnan(r) else f"+{r:.1f}%"]
        if row["config_id"] != row["oracle_config"]:
            lines.append(f"best: {row['oracle_config']}")
        ax.text(x, y, "\n".join(lines), ha="center", va="center", color=INK, fontsize=7, linespacing=1.25)
    ax.set_xticks(range(len(seqs)), seqs)
    ax.set_yticks(range(len(batches)), batches)
    ax.set_xlim(-0.6, len(seqs) - 0.4)
    ax.set_ylim(-0.6, len(batches) - 0.4)
    ax.set_xlabel("Sequence length (tokens)")
    ax.set_ylabel("Batch size")
    ax.set_title(f"{LABEL[method]}: chosen config and extra latency over the best tested config\n"
                 f"{scenario}; dark outline = held-out shape, others = leave-one-out; darker = larger gap",
                 loc="left", color=INK, fontsize=10)
    _style(ax)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    return _save(fig, path)


def make_all(per, summary, budget_df, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for scenario in summary["scenario"].unique():
        evals = summary.loc[summary["scenario"] == scenario, "eval"].unique()
        for eval_name in evals:
            paths.append(baselines(summary, scenario, eval_name, out_dir / f"baselines_{scenario}_{eval_name}.png"))
        paths.append(budget(budget_df, scenario, out_dir / f"budget_{scenario}.png"))
        for method in ("lookup", "regression"):
            paths.append(choice_map(per, scenario, method, out_dir / f"choices_{scenario}_{method}.png"))
    return paths
