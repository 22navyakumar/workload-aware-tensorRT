# Sweep results

- `sweep.csv`: the sweep to use (written by `harness/sweep.py`, 2 s warm-up, 5 timing repeats). Not here yet; the run is in progress.
- `sweep_v1_short_warmup.csv`: the first, partial sweep (83 of 340 runs). **Do not use its latencies.** GPU-bound configs read about 20% high in some repeats because the warm-up was too short. Kept only as the evidence for that finding; see `experiments/e4_workload_selector/LOG.md`.
