"""Read configs/workloads.yaml. Needs only PyYAML, so analysis code can import it without torch."""
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_YAML = REPO_ROOT / "configs" / "workloads.yaml"

# Columns of the shared results CSV written by harness/sweep.py (one row per timing repeat).
RESULT_FIELDS = [
    "timestamp", "model", "precision", "scenario", "config_id", "split", "batch", "seq_len",
    "repeat", "status", "error", "compile_s", "first_call_s", "settle_s", "bench_s", "median_ms", "p95_ms",
    "throughput_seq_per_s", "peak_mem_mb", "gpu_used_mb", "n_trt_blocks", "n_torch_blocks",
    "max_abs_diff", "cos_sim", "settings", "host", "job_id"]


def load_spec(path=DEFAULT_YAML):
    with open(path) as f:
        return yaml.safe_load(f)


def grid_workloads(spec):
    """Training grid: every batch size × sequence length, as (batch, seq_len)."""
    return [(b, s) for b in spec["batch_sizes"] for s in spec["seq_lens"]]


def heldout_workloads(spec):
    grid = set(grid_workloads(spec))
    out = [tuple(w) for w in spec.get("heldout_workloads", [])]
    overlap = [w for w in out if w in grid]
    if overlap:
        raise ValueError(f"held-out workloads overlap the training grid: {overlap}")
    return out


def workloads(spec, split="all", max_seq_len=None):
    """Return [(split_name, batch, seq_len)], dropping shapes the model cannot take."""
    rows = []
    if split in ("grid", "all"):
        rows += [("grid", b, s) for b, s in grid_workloads(spec)]
    if split in ("heldout", "all"):
        rows += [("heldout", b, s) for b, s in heldout_workloads(spec)]
    if max_seq_len is not None:
        rows = [r for r in rows if r[2] <= max_seq_len]
    return rows


def configs(spec, only=None):
    cfgs = spec["configs"]
    ids = [c["id"] for c in cfgs]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate config ids in workloads.yaml")
    if only:
        missing = set(only) - set(ids)
        if missing:
            raise ValueError(f"unknown config ids: {sorted(missing)}")
        cfgs = [c for c in cfgs if c["id"] in only]
    return cfgs


def scenarios(spec):
    return sorted({c["scenario"] for c in spec["configs"]} - {"all", "control"})


def scenario_configs(spec, scenario):
    """Config ids that may be chosen between in one scenario (includes eager)."""
    return [c["id"] for c in spec["configs"] if c["scenario"] in (scenario, "all")]


def default_config(spec, scenario):
    hits = [c["id"] for c in spec["configs"] if c["scenario"] == scenario and c.get("default")]
    if len(hits) != 1:
        raise ValueError(f"scenario {scenario!r} needs exactly one config with default: true")
    return hits[0]
