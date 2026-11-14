# Sweep results

- `sweep.csv`: the sweep to use. 17 configs × 20 shapes, FP16, written by `harness/sweep.py` in job 29729 on one A100 node (2026-11-12). 2 s warm-up, then 5 timing repeats of 100 iterations; one row per repeat. 320 runs succeeded; the 20 `ln_global` runs failed to build and have `status=error`.
- `sweep_v1_short_warmup.csv`: the first, partial sweep (83 of 340 runs). **Do not use its latencies.** GPU-bound configs read about 20% high in some repeats because the warm-up was too short. Kept only as the evidence for that finding; see `experiments/e4_workload_selector/LOG.md`.

Analysis of `sweep.csv`: `results/e4_workload_selector/README.md`.
