"""Workload-aware config selectors. Pure numpy/pandas/scipy: runs anywhere, no GPU needed.

A selector is fitted on measured latencies for some workloads (long table with columns
config_id, batch, seq_len, latency_ms) and then picks a config for any (batch, seq_len).
"""
import numpy as np
import pandas as pd
from scipy.optimize import nnls

KEY = ["batch", "seq_len"]


def load_latency(csv_path):
    """Collapse the sweep CSV to one row per (config, workload): median over timing repeats."""
    df = pd.read_csv(csv_path)
    ok = df[df["status"] == "ok"].copy()
    if "settle_s" not in ok:
        ok["settle_s"] = 0.0
    agg = ok.groupby(["config_id", "split"] + KEY, as_index=False).agg(
        latency_ms=("median_ms", "median"),
        lat_min=("median_ms", "min"),
        lat_max=("median_ms", "max"),
        compile_s=("compile_s", "first"),
        first_call_s=("first_call_s", "first"),
        settle_s=("settle_s", "first"),
        bench_s=("bench_s", "sum"),
        n_trt_blocks=("n_trt_blocks", "first"),
        n_torch_blocks=("n_torch_blocks", "first"),
        max_abs_diff=("max_abs_diff", "first"),
    )
    # What it costs to obtain this one measurement: build the engine, load it, warm up, time it.
    agg["measure_s"] = agg["compile_s"] + agg["first_call_s"] + agg["settle_s"].fillna(0) + agg["bench_s"]
    return agg


class Fixed:
    """Always the same config (used for the eager and default baselines)."""

    def __init__(self, config_id):
        self.config_id = config_id

    def fit(self, train):
        return self

    def choose(self, batch, seq_len):
        return self.config_id


class BestFixed:
    """One tuned config for all workloads: lowest geometric-mean slowdown on the training set."""

    def fit(self, train):
        wide = train.pivot(index=KEY, columns="config_id", values="latency_ms")
        complete = wide.dropna(axis=1)  # only configs that ran on every training workload
        slowdown = complete.div(complete.min(axis=1), axis=0)
        self.config_id = np.log(slowdown).mean().idxmin()
        return self

    def choose(self, batch, seq_len):
        return self.config_id


class Lookup:
    """Best config of the nearest measured workload, in (log2 batch, log2 seq_len) space."""

    def fit(self, train):
        best = train.loc[train.groupby(KEY)["latency_ms"].idxmin()]
        self.points = np.log2(best[KEY].to_numpy(dtype=float))
        self.config_ids = best["config_id"].to_numpy()
        return self

    def choose(self, batch, seq_len):
        d = ((self.points - np.log2([batch, seq_len])) ** 2).sum(axis=1)
        return self.config_ids[d.argmin()]


def _features(batch, seq_len):
    """Fixed overhead, per-sequence cost, per-token cost, attention (quadratic in length) cost."""
    b = np.asarray(batch, dtype=float)
    s = np.asarray(seq_len, dtype=float)
    return np.stack([np.ones_like(b), b, b * s, b * s * s], axis=1)


class Regression:
    """One non-negative latency model per config; pick the config with the lowest prediction.

    Fitted on relative error, so a 5% miss at batch 1 counts the same as a 5% miss at batch 16.
    """

    def fit(self, train):
        self.theta = {}
        for config_id, g in train.groupby("config_id"):
            X = _features(g["batch"], g["seq_len"])
            y = g["latency_ms"].to_numpy(dtype=float)
            scale = X.max(axis=0)
            coef, _ = nnls((X / scale) / y[:, None], np.ones_like(y))
            self.theta[config_id] = coef / scale
        return self

    def predict(self, batch, seq_len):
        x = _features([batch], [seq_len])[0]
        return {config_id: float(x @ theta) for config_id, theta in self.theta.items()}

    def choose(self, batch, seq_len):
        pred = self.predict(batch, seq_len)
        return min(pred, key=pred.get)


def make_selectors(default_id):
    """Every method compared in the report except the oracle, in display order."""
    return {
        "eager": Fixed("eager"),
        "default": Fixed(default_id),
        "fixed_tuned": BestFixed(),
        "lookup": Lookup(),
        "regression": Regression(),
    }


LEARNED = ["fixed_tuned", "lookup", "regression"]  # methods whose quality depends on tuning data
