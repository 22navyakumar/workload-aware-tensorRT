"""Write a FAKE sweep CSV from a made-up latency formula, to test the E4 pipeline without a GPU.

    python experiments/e4_workload_selector/make_synthetic.py /some/path/synthetic.csv

The numbers mean nothing. Never put this file's output in results/ or in the report.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from harness import configs as cfgmod  # noqa: E402

# config id -> (fraction of compute in TensorRT, boundaries, engine speed factor, fixed overhead ms)
SHAPE = {
    "trt_default": (1.0, 0, 0.45, 0.40), "trt_default_rebuild": (1.0, 0, 0.45, 0.40),
    "trt_opt1": (1.0, 0, 0.52, 0.40), "trt_opt5": (1.0, 0, 0.43, 0.40),
    "trt_cudagraphs": (1.0, 0, 0.45, 0.22),
    "gelu_mbs5": (0.97, 12, 0.45, 0.40), "gelu_mbs1": (0.99, 13, 0.45, 0.40),
    "gelu_mbs22": (0.18, 1, 0.45, 0.40), "gelu_global": (0.97, 12, 0.45, 0.30),
    "gelu_cudagraphs": (0.97, 4, 0.45, 0.40),
    "ln_mbs5": (0.70, 14, 0.45, 0.40), "ln_mbs1": (0.90, 26, 0.45, 0.40),
    "ln_mbs10": (0.40, 6, 0.45, 0.40), "ln_mbs20": (0.0, 0, 0.45, 0.40),
    "ln_global": (0.90, 24, 0.45, 0.40), "ln_cudagraphs": (0.70, 5, 0.45, 0.40),
}


def latency_ms(config_id, batch, seq_len):
    compute = 0.004 * batch * seq_len + 4e-6 * batch * seq_len**2
    if config_id == "eager":
        return 4.0 + compute
    frac, boundaries, speed, overhead = SHAPE[config_id]
    return overhead + 0.25 * boundaries + 4.0 * (1 - frac) + compute * (frac * speed + (1 - frac))


def main():
    out = Path(sys.argv[1])
    spec = cfgmod.load_spec()
    rng = np.random.default_rng(0)
    rows = []
    for split, batch, seq_len in cfgmod.workloads(spec, "all", max_seq_len=512):
        for cfg in cfgmod.configs(spec):
            true = latency_ms(cfg["id"], batch, seq_len)
            is_trt = cfg["backend"] == "trt"
            frac, boundaries = SHAPE.get(cfg["id"], (0, 0))[:2]
            compile_s = float(rng.uniform(40, 90)) if is_trt else 0.0
            for rep in range(spec["repeats"]):
                med = true * float(rng.lognormal(0, 0.01))
                rows.append({
                    "timestamp": "SYNTHETIC", "model": "SYNTHETIC", "precision": spec["precision"],
                    "scenario": cfg["scenario"], "config_id": cfg["id"], "split": split,
                    "batch": batch, "seq_len": seq_len, "repeat": rep, "status": "ok", "error": "",
                    "compile_s": compile_s, "first_call_s": 0.5 if is_trt else 0.05, "settle_s": 2.0,
                    "bench_s": 120 * med / 1000, "median_ms": med, "p95_ms": med * 1.05,
                    "throughput_seq_per_s": batch / (med / 1000), "peak_mem_mb": 0, "gpu_used_mb": 0,
                    "n_trt_blocks": (boundaries // 2 + 1) if is_trt and frac > 0 else 0,
                    "n_torch_blocks": boundaries // 2, "max_abs_diff": 0.0, "cos_sim": 1.0,
                    "settings": json.dumps(cfg.get("settings", {}), sort_keys=True),
                })
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cfgmod.RESULT_FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} synthetic rows to {out}")


if __name__ == "__main__":
    main()
