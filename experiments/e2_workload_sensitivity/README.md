# E2 — Workload sensitivity (Fahad)

**Plan:** Compare shared configurations for batch sizes 1, 4, 8, 16 and sequence lengths 128–1024 (as GPU memory permits), holding weights and precision constant.

**Deliverable:** Heatmap of best settings by batch and sequence length.

Use the shared harness (`harness/`) and `configs/workloads.yaml`. Write outputs to `results/e2_workload_sensitivity/`.
