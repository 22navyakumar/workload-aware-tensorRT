# When Does Graph Compilation Help?
### Workload-Aware Optimization of Transformer Inference

Efficient AI and Hardware Accelerators — course project.
Navya Kumar · Fahad Tajiki · Demetri López · Tanish Vardhineni

## Research question
When do boundaries between TensorRT and PyTorch regions reduce inference speed, and can we
choose a better Torch-TensorRT compilation configuration for each workload (batch size × sequence length)?

We use a small pretrained Transformer in inference mode, inspect its partitioned graph, measure
latency across workloads, and build a lightweight selector over existing compiler configurations.
We are **not** building a new compiler. Full proposal: [`docs/proposal.pdf`](docs/proposal.pdf).

## Experiments

| ID | Experiment | Owner | Deliverable | Folder |
|----|------------|-------|-------------|--------|
| E1 | Graph partitioning | Navya | Partition maps; latency vs. compiler settings | `experiments/e1_partitioning` |
| E2 | Workload sensitivity | Fahad | Heatmap of best settings by batch & sequence length | `experiments/e2_workload_sensitivity` |
| E3 | GPU bottleneck analysis | Demetri | Nsight traces explaining launch/dispatch overhead | `experiments/e3_gpu_profiling` |
| E4 | Workload-aware selection | Tanish | Cost-based selector; speedup & tuning-cost results | `experiments/e4_workload_selector` |

All experiments share the same model, GPU, software versions, and the benchmark harness in `harness/`.

## Evaluation
Baselines: PyTorch eager, default Torch-TensorRT, one fixed tuned config, our workload-aware selector,
and the best tested config (upper bound).
Metrics: warmed-up median & p95 latency, throughput, GPU memory, numerical agreement vs. PyTorch,
partition counts, tuning time. FP16 where supported; identical inputs across runs.

## Layout
```
configs/       shared model / workload / compiler-setting definitions
harness/       common benchmarking + compilation helpers
experiments/   one folder per experiment (E1–E4)
results/       raw CSV/JSON outputs and figures (large traces are git-ignored)
docs/          proposal and write-ups
```

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
Record the exact GPU, driver, CUDA, PyTorch and Torch-TensorRT versions in `configs/environment.md`
before collecting results.

## References
1. PyTorch, "Torch-TensorRT: Partitioning Phase," https://docs.pytorch.org/TensorRT/contributors/partitioning.html
2. PyTorch, "Torch-TensorRT: Performance Tuning Guide," https://docs.pytorch.org/TensorRT/user_guide/performance_tuning.html
