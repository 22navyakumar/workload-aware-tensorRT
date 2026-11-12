"""Run every config × workload and append one CSV row per timing repeat.

    python -m harness.sweep --out results/sweep/sweep.csv
    python -m harness.sweep --out results/sweep/sweep.csv --configs eager,trt_default --limit 2

Resumable: (config_id, batch, seq_len) triples already in the CSV are skipped, so a job
that hits its time limit can simply be resubmitted. Failed compiles are recorded with
status != ok and skipped on resume unless --retry-errors is given.
"""
import argparse
import csv
import gc
import json
import logging
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from harness import configs as cfgmod
from harness.bench import benchmark
from harness.model import load_model, make_inputs
from harness.trt import EncoderWrapper, compile_config, first_tensor, partition_summary, runnable

FIELDS = cfgmod.RESULT_FIELDS


def read_done(path, retry_errors):
    done = set()
    if not path.exists():
        return done
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if retry_errors and row["status"] != "ok":
                continue
            done.add((row["config_id"], int(row["batch"]), int(row["seq_len"])))
    return done


def append_rows(path, rows):
    new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerows(rows)


def gpu_used_mb():
    free, total = torch.cuda.mem_get_info()
    return (total - free) / 2**20


def agreement(out, ref):
    a, b = out.float().flatten(), ref.float().flatten()
    return (a - b).abs().max().item(), torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def free_gpu():
    gc.collect()
    torch.cuda.empty_cache()


def settle(fn, inputs, seconds):
    """Keep the GPU busy for `seconds` before timing.

    After an idle spell (a compile runs mostly on the CPU) the A100 takes a few hundred
    milliseconds of sustained load to reach full speed; until then latencies read about
    20% high. The shared 20-iteration warm-up is too short to get past that.
    """
    t0 = time.perf_counter()
    with torch.inference_mode():
        while time.perf_counter() - t0 < seconds:
            for _ in range(10):
                fn(*inputs)
            torch.cuda.synchronize()
    return time.perf_counter() - t0


def run_one(cfg, eager, inputs, ref, spec, repeats):
    """Compile (if needed) and time one config on one workload. Returns per-repeat dicts."""
    base = {"status": "ok", "error": "", "compile_s": 0.0, "n_trt_blocks": 0, "n_torch_blocks": 0}
    module = eager
    if cfg["backend"] == "trt":
        t0 = time.perf_counter()
        module = compile_config(eager, inputs, cfg.get("settings"))
        base["compile_s"] = time.perf_counter() - t0
        part = partition_summary(module)
        base["n_trt_blocks"], base["n_torch_blocks"] = part["n_trt_blocks"], part["n_torch_blocks"]

    rows = []
    with runnable(module, cfg.get("cudagraphs", False)) as fn:
        with torch.inference_mode():
            t0 = time.perf_counter()
            out = first_tensor(fn(*inputs))
            torch.cuda.synchronize()
            base["first_call_s"] = time.perf_counter() - t0
        base["max_abs_diff"], base["cos_sim"] = agreement(out, ref)
        base["settle_s"] = settle(fn, inputs, spec.get("settle_s", 0.0))
        for rep in range(repeats):
            t0 = time.perf_counter()
            m = benchmark(fn, inputs, warmup=spec["warmup_iters"], iters=spec["timed_iters"])
            rows.append({**base, **m, "repeat": rep, "bench_s": time.perf_counter() - t0,
                         "gpu_used_mb": gpu_used_mb()})
    del module, out
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--spec", type=Path, default=cfgmod.DEFAULT_YAML)
    ap.add_argument("--configs", help="comma-separated config ids (default: all)")
    ap.add_argument("--split", choices=["grid", "heldout", "all"], default="all")
    ap.add_argument("--limit", type=int, help="only the first N workloads (smoke test)")
    ap.add_argument("--repeats", type=int)
    ap.add_argument("--retry-errors", action="store_true")
    args = ap.parse_args()

    logging.getLogger("torch_tensorrt").setLevel(logging.ERROR)  # its INFO output is ~5 kB per compile
    spec = cfgmod.load_spec(args.spec)
    repeats = args.repeats or spec.get("repeats", 3)
    cfgs = cfgmod.configs(spec, args.configs.split(",") if args.configs else None)

    eager = EncoderWrapper(load_model(spec["model"])).eval()
    hf_cfg = eager.model.config
    work = cfgmod.workloads(spec, args.split, max_seq_len=hf_cfg.max_position_embeddings)
    dropped = len(cfgmod.workloads(spec, args.split)) - len(work)
    if dropped:
        print(f"dropped {dropped} workloads longer than {hf_cfg.max_position_embeddings} tokens")
    if args.limit:
        work = work[: args.limit]

    done = read_done(args.out, args.retry_errors)
    todo = [(c, w) for w in work for c in cfgs if (c["id"], w[1], w[2]) not in done]
    print(f"{len(todo)} runs to do, {len(done)} already in {args.out}")

    for i, (cfg, (split, batch, seq_len)) in enumerate(todo, 1):
        inputs = make_inputs(batch, seq_len, vocab_size=hf_cfg.vocab_size)
        common = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "model": spec["model"], "precision": spec["precision"], "scenario": cfg["scenario"],
            "config_id": cfg["id"], "split": split, "batch": batch, "seq_len": seq_len,
            "host": platform.node(), "job_id": os.environ.get("SLURM_JOB_ID", ""),
            "settings": json.dumps({**cfg.get("settings", {}), "cudagraphs": cfg.get("cudagraphs", False)},
                                   sort_keys=True),
        }
        try:
            with torch.inference_mode():
                ref = eager(*inputs)
            rows = run_one(cfg, eager, inputs, ref, spec, repeats)
        except torch.cuda.OutOfMemoryError as e:
            rows = [{"status": "oom", "error": str(e)[:300], "repeat": 0}]
        except Exception as e:  # keep sweeping; the failure itself is a data point
            rows = [{"status": "error", "error": f"{type(e).__name__}: {e}"[:300], "repeat": 0}]
        append_rows(args.out, [{**common, **r} for r in rows])
        r = rows[0]
        msg = (f"{r['median_ms']:.3f} ms, compile {r['compile_s']:.1f}s, "
               f"{r['n_trt_blocks']} trt / {r['n_torch_blocks']} torch blocks"
               if r["status"] == "ok" else f"{r['status']}: {r['error']}")
        print(f"[{i}/{len(todo)}] {cfg['id']} b={batch} s={seq_len}: {msg}", flush=True)
        free_gpu()


if __name__ == "__main__":
    main()
