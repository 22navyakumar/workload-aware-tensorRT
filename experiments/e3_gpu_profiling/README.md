# E3 — GPU bottleneck analysis (Demetri)

**Plan:** Profile representative runs with PyTorch Profiler and Nsight Systems. Contrast naturally occurring fallback regions with deliberately controlled graph breaks; keep conclusions separate.

**Deliverable:** Nsight traces explaining launch/dispatch overhead.

Use the shared harness (`harness/`) and `configs/workloads.yaml`. Write outputs to `results/e3_gpu_profiling/`.
