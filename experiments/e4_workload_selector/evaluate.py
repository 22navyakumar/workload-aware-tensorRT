"""Evaluate the E4 selectors against the baselines from a sweep CSV.

    python experiments/e4_workload_selector/evaluate.py --csv results/sweep/sweep.csv

For every scenario in configs/workloads.yaml it compares eager, the default config, one fixed
tuned config, the lookup and regression selectors, and the oracle (best tested config), two ways:
  heldout  fit on the training grid, test on the off-grid held-out shapes
  loo      leave one grid workload out, fit on the rest, test on the one left out
and traces selector quality against tuning budget (how many workloads were measured).
Writes per_workload.csv, summary.csv, budget.csv and figures/ under --out.
"""
import argparse
import itertools
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import configs as cfgmod  # noqa: E402
from selector import KEY, LEARNED, load_latency, make_selectors  # noqa: E402

METHODS = ["eager", "default", "fixed_tuned", "lookup", "regression", "oracle"]


def score(selectors, test, scenario, eval_name):
    """One row per (test workload, method): which config was run and how it compares."""
    rows = []
    for (batch, seq_len), g in test.groupby(KEY):
        table = dict(zip(g["config_id"], g["latency_ms"]))
        oracle_id = min(table, key=table.get)
        chosen = {name: sel.choose(batch, seq_len) for name, sel in selectors.items()}
        chosen["oracle"] = oracle_id
        for method in METHODS:
            rows.append({
                "scenario": scenario, "eval": eval_name, "batch": batch, "seq_len": seq_len,
                "method": method, "config_id": chosen[method],
                "latency_ms": table.get(chosen[method], np.nan),  # NaN: that config failed here
                "oracle_ms": table[oracle_id], "oracle_config": oracle_id,
                "eager_ms": table.get(chosen["eager"], np.nan),
                "default_ms": table.get(chosen["default"], np.nan),
                "fixed_ms": table.get(chosen["fixed_tuned"], np.nan),
            })
    return rows


def fit_all(train, default_id):
    return {name: sel.fit(train) for name, sel in make_selectors(default_id).items()}


def geomean(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.exp(np.log(x).mean())) if len(x) else np.nan


def summarise(per, tol):
    out = []
    for (scenario, eval_name, method), g in per.groupby(["scenario", "eval", "method"], sort=False):
        slowdown = g["latency_ms"] / g["oracle_ms"]
        out.append({
            "scenario": scenario, "eval": eval_name, "method": method,
            "n_workloads": len(g), "n_failed": int(g["latency_ms"].isna().sum()),
            "regret_pct": (geomean(slowdown) - 1) * 100,
            "max_regret_pct": (slowdown.max() - 1) * 100,
            "near_best_pct": float((slowdown <= 1 + tol).mean() * 100),
            "speedup_vs_eager": geomean(g["eager_ms"] / g["latency_ms"]),
            "speedup_vs_default": geomean(g["default_ms"] / g["latency_ms"]),
            "speedup_vs_fixed": geomean(g["fixed_ms"] / g["latency_ms"]),
            "ms_saved_vs_default": float((g["default_ms"] - g["latency_ms"]).mean()),
        })
    return pd.DataFrame(out)


def budget_curve(grid, test, scenario, default_id, max_subsets, rng):
    """Selector regret when only k of the grid workloads were measured, for every k."""
    points = sorted(set(map(tuple, grid[KEY].to_numpy())))
    rows = []
    for k in range(1, len(points) + 1):
        if math.comb(len(points), k) <= max_subsets:
            subsets = list(itertools.combinations(range(len(points)), k))
        else:
            subsets = {tuple(sorted(rng.choice(len(points), k, replace=False))) for _ in range(max_subsets)}
        for subset in subsets:
            chosen = {points[i] for i in subset}
            mask = [tuple(w) in chosen for w in grid[KEY].to_numpy()]
            train = grid[mask]
            target = test if test is not None else grid[[not m for m in mask]]
            if target.empty:
                continue
            per = pd.DataFrame(score(fit_all(train, default_id), target, scenario, "budget"))
            for method in LEARNED:
                m = per[per["method"] == method]
                rows.append({"scenario": scenario, "k": k, "method": method,
                             "regret_pct": (geomean(m["latency_ms"] / m["oracle_ms"]) - 1) * 100,
                             "tuning_cost_s": train["measure_s"].sum()})
    df = pd.DataFrame(rows)
    return df.groupby(["scenario", "k", "method"], as_index=False).agg(
        regret_pct=("regret_pct", "mean"),
        regret_pct_p90=("regret_pct", lambda x: x.quantile(0.9)),
        tuning_cost_s=("tuning_cost_s", "mean"),
        n_subsets=("regret_pct", "size"),
    )


def rebuild_spread(lat, a="trt_default", b="trt_default_rebuild"):
    """Median latency gap between two separate builds of the same config, in percent (or None)."""
    wide = lat[lat["config_id"].isin([a, b])].pivot_table(index=KEY, columns="config_id", values="latency_ms")
    if a not in wide or b not in wide:
        return None
    wide = wide.dropna()
    return float(((wide[a] - wide[b]).abs() / wide[[a, b]].min(axis=1)).median() * 100)


def evaluate(csv_path, tol=0.02, max_subsets=40, seed=0):
    spec = cfgmod.load_spec()
    lat = load_latency(csv_path)
    rng = np.random.default_rng(seed)
    noise_pct = float(((lat["lat_max"] - lat["lat_min"]) / lat["latency_ms"]).median() * 100)
    rebuild_pct = rebuild_spread(lat)
    per_rows, budgets, costs = [], [], {}
    for scenario in cfgmod.scenarios(spec):
        default_id = cfgmod.default_config(spec, scenario)
        data = lat[lat["config_id"].isin(cfgmod.scenario_configs(spec, scenario))]
        grid, held = data[data["split"] == "grid"], data[data["split"] == "heldout"]
        if not {"eager", default_id} <= set(grid["config_id"]):
            print(f"skipping scenario {scenario}: eager or {default_id} missing from the grid results")
            continue
        costs[scenario] = float(grid["measure_s"].sum())
        if not held.empty:
            per_rows += score(fit_all(grid, default_id), held, scenario, "heldout")
        for w in sorted(set(map(tuple, grid[KEY].to_numpy()))):
            is_w = (grid["batch"] == w[0]) & (grid["seq_len"] == w[1])
            per_rows += score(fit_all(grid[~is_w], default_id), grid[is_w], scenario, "loo")
        budgets.append(budget_curve(grid, None if held.empty else held, scenario, default_id, max_subsets, rng))

    per = pd.DataFrame(per_rows)
    summary = summarise(per, tol)
    # Requests needed before tuning pays for itself, against simply using the default config.
    summary["tuning_cost_s"] = [costs[s] if m in LEARNED else 0.0
                                for s, m in zip(summary["scenario"], summary["method"])]
    saved = summary["ms_saved_vs_default"].where(summary["ms_saved_vs_default"] > 0)
    summary["break_even_requests"] = (summary["tuning_cost_s"] * 1000 / saved).where(summary["tuning_cost_s"] > 0)
    return per, summary, pd.concat(budgets, ignore_index=True), {"repeat_pct": noise_pct, "rebuild_pct": rebuild_pct}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=cfgmod.REPO_ROOT / "results/e4_workload_selector")
    ap.add_argument("--tie-tol", type=float, default=0.02,
                    help="a choice within this fraction of the best counts as near-best")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    per, summary, budget, noise = evaluate(args.csv, tol=args.tie_tol)
    args.out.mkdir(parents=True, exist_ok=True)
    per.to_csv(args.out / "per_workload.csv", index=False)
    summary.to_csv(args.out / "summary.csv", index=False)
    budget.to_csv(args.out / "budget.csv", index=False)

    print(f"timing noise, same engine re-timed (median spread): {noise['repeat_pct']:.2f}%")
    if noise["rebuild_pct"] is not None:
        print(f"build noise, same config built twice (median gap): {noise['rebuild_pct']:.2f}%")
    print("differences between configs smaller than these are not meaningful\n")
    pd.DataFrame([noise]).to_csv(args.out / "noise.csv", index=False)
    cols = ["scenario", "eval", "method", "n_workloads", "n_failed", "regret_pct", "max_regret_pct",
            "near_best_pct", "speedup_vs_eager", "speedup_vs_default", "speedup_vs_fixed"]
    with pd.option_context("display.width", 200, "display.float_format", "{:.2f}".format):
        print(summary[cols].to_string(index=False))
    if not args.no_figures:
        import figures

        for path in figures.make_all(per, summary, budget, args.out / "figures"):
            print("wrote", path)


if __name__ == "__main__":
    main()
