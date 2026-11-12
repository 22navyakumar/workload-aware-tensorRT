# E4 run log

What was run, what it showed, and what changed because of it. Times are UTC. Newest entry last.
Hardware and software versions: [`configs/environment.md`](../../configs/environment.md).

## Status (2026-11-12 14:15)

- Partition check: done. Results in `results/e4_workload_selector/partition_check.json`.
- Sweep v1 (short warm-up): 83 of 340 runs, then cancelled by the cluster. Timings are flawed (see below). Kept as `results/sweep/sweep_v1_short_warmup.csv` for reference only.
- Sweep v2 (2 s warm-up, 5 repeats): running. Job 29729 started at 14:02 on node b-19-51 and had 26 of 340 runs at 14:12. Seven more jobs are chained behind it (29730 to 29732, 29750 to 29753).
- The warm-up fix is confirmed on the first 15 runs: the five repeats agree within 1% for the GPU-bound configs.
- Selector evaluation and figures: code works on synthetic data and on the v1 partial data. No reportable numbers yet.

## 2026-11-12

### Environment (jobs 29499, 29507, 29509)

- Built the Conda environment in a Singularity overlay with `slurm/setup_env.sbatch` on the CPU partition. Third attempt worked.
- 29499: `--fakeroot` unpacked the 8.5 GB image to a sandbox on NFS and was too slow. Cancelled. A plain `--overlay file.ext3:rw` mount works.
- 29507: held by Slurm ("user env retrieval failed"). Cancelled and resubmitted.
- 29509: packages installed. The job still exited non-zero because the post-install check imported `torch_tensorrt`, which cannot be imported on a node without a GPU. The check now reads package metadata instead.

### Partition check (jobs 29537, 29550)

Dry-run compile of every config at batch 4 × 128 tokens, FP16. DistilBERT lowers to 127 operators.

| Config | TensorRT blocks | PyTorch blocks | Ops per TensorRT block | Ops per PyTorch block |
|---|---|---|---|---|
| `trt_default`, `trt_opt1`, `trt_opt5` | 1 | 0 | 127 | |
| `gelu_mbs5`, `gelu_global` | 6 | 6 | 23, 19 × 5 | 1 × 5, 4 |
| `gelu_mbs1` | 7 | 6 | 23, 19 × 5, 3 | 1 × 6 |
| `gelu_mbs22` | 1 | 1 | 23 | 104 |
| `ln_mbs5` | 7 | 7 | 6, 14 × 6 | 1, 6 × 6 |
| `ln_mbs1` | 13 | 13 | 6, 14, then 4 and 14 alternating | 1 × 13 |
| `ln_mbs10` | 6 | 7 | 14 × 6 | 7, 6 × 6 |
| `ln_mbs20` | 0 | 1 | | 127 |
| `ln_global` | 6 | 7 | 18, 14 × 5 | 3, 6 × 6 |

What this changed:

- **Every DistilBERT operator is supported, so the default compile is one engine with no fallback.** `min_block_size` and the partitioner choice change nothing on the unmodified model. Partition tuning only exists once an operator is forced to PyTorch, so the sweep has three scenarios: `natural`, `break_gelu` and `break_layernorm`.
- Dropped `trt_mbs1` and `trt_global` (identical to the default). Added `trt_default_rebuild` to measure build-to-build noise.
- `gelu_mbs10` and `gelu_mbs20` partition exactly like `gelu_mbs5` (the TensorRT blocks are 19 to 23 ops), so they were replaced by `gelu_mbs22`.
- The LayerNorm operator after lowering is `aten.native_layer_norm.default`, not `aten.layer_norm.default`. Job 29537 used the wrong name and excluded nothing; 29550 is the corrected run.
- The global partitioner leaves PyTorch operators inline in the top-level graph instead of in `_run_on_gpu_N` submodules. `harness/trt.py: partition_summary` counts both forms.

### Sweep v1 (jobs 29551, 29586)

- 29551: cancelled by Slurm after 12 minutes because the A100 node never booted. GPU jobs are now submitted with `--requeue`.
- 29586: ran 30 minutes on node b-19-31, finished 83 runs (batch 1 × 128, 256, 512; batch 4 × 128; most of batch 4 × 256), then was cancelled.

Findings from those 83 runs (three timing repeats each, 20 warm-up iterations):

| Config, batch 1 × 128 tokens | Median latency (ms) | Blocks (TensorRT / PyTorch) |
|---|---|---|
| `eager` | 3.79 | |
| `trt_default` | 0.59 to 0.71 | 1 / 0 |
| `trt_cudagraphs` | 0.55 | 1 / 0 |
| `gelu_mbs5` | 1.79 | 6 / 6 |
| `gelu_mbs22` | 3.95 | 1 / 1 |
| `gelu_cudagraphs` | 0.68 | 6 / 6 |
| `ln_mbs5` | 3.11 | 7 / 7 |
| `ln_mbs1` | 2.89 | 13 / 13 |
| `ln_mbs20` | 4.53 | 0 / 1 |
| `ln_cudagraphs` | 0.60 | 7 / 7 |

- Outputs agree with eager PyTorch: cosine similarity 1.0, largest absolute difference 0.004 to 0.010 in FP16.
- `ln_global` fails on every workload: `AttributeError: 'GraphModule' object has no attribute '_frozen_param0'`. It is recorded with `status=error` and is not a candidate for selection.
- Engine build time depends on what was built before: 39 s for the first build, 6 to 13 s for later builds of the same graph (shared timing cache).

**Problem 1: the cluster cancels low-utilization jobs.** `sacct` showed `CANCELLED by 0`. The cause is `/share/apps/admins/scripts/check_gpu_utilizations.bash`: a GPU job whose mean GPU utilization is 5% or less is cancelled once it is about 30 minutes old. The sweep spends most of its time compiling on the CPU. No artificial GPU load was added. The sweep is resumable, so it is submitted as a chain of jobs; each continues where the last stopped.

**Problem 2: the v1 timings are not trustworthy.** For 30 of 79 config and workload pairs, the three repeats disagreed by 8% or more. The pattern is always the same: the first one or two repeats are about 20% slower, then the latency drops and stays there. It only affects GPU-bound configs (a single engine, or CUDA Graphs); CPU-bound ones (eager, PyTorch fallback blocks) are stable. The GPU needs a few hundred milliseconds of sustained load after an idle spell, and the shared 20-iteration warm-up is over before that for a sub-millisecond engine.

### Sweep v2 (jobs 29727 to 29732)

Changes from v1:

- `harness/sweep.py: settle` keeps the GPU busy for `settle_s` (2.0 s) before timing. The shared `harness/bench.py` is unchanged and still does its own 20 warm-up iterations.
- `repeats` raised from 3 to 5.
- New CSV columns: `settle_s`, `host`, `job_id`. `host` shows which runs came from which node, because a chained sweep lands on several.
- The warm-up time is counted in the tuning cost (`measure_s` in `selector.py`).
- v1 results are not reused; v2 starts from an empty `results/sweep/sweep.csv`.

Submitted 12:32 as `sbatch --requeue slurm/sweep.sbatch`, then five more with `--dependency=afterany:<previous>`.

- 12:33 to 12:40: node b-19-30 failed to boot. Requeued.
- 12:42 to 12:50: node b-19-64 failed to boot. Requeued.
- 12:54 to 13:50: jobs 29727 and 29728 each waited 13 minutes on a node that never booted (b-19-43, b-19-63) and were cancelled by Slurm. Neither wrote a log.
- 14:02: job 29729 started on b-19-51. First v2 rows.
- 14:12: four more jobs (29750 to 29753) added to the end of the chain to make up for the two lost ones.

First 15 v2 runs (batch 1 × 128 tokens), five repeats each:

| Config | Median latency (ms) | Spread across repeats |
|---|---|---|
| `eager` | 3.87 | 1.8% |
| `trt_default` | 0.588 | 0.9% |
| `trt_default_rebuild` | 0.595 | 0.2% |
| `trt_opt1` | 0.637 | 0.5% |
| `trt_opt5` | 0.592 | 0.7% |
| `trt_cudagraphs` | 0.493 | 0.4% |
| `gelu_mbs5` | 1.83 | 1.7% |
| `gelu_cudagraphs` | 0.559 | 0.4% |
| `ln_mbs5` | 3.03 | 2.1% |

The 20% jumps between repeats are gone, and the values match the faster of the two levels seen in v1 (0.588 ms for `trt_default`). Median spread over the 15 runs is 1.7%; the largest is 5.8% (`gelu_mbs22`, which runs mostly in PyTorch).

## Open items

- Whether the v2 jobs stay above the 5% utilization threshold. If not, the chain covers it.
- Team decision: DistilBERT's position limit is 512 tokens, so the 768 and 1024-token shapes in the proposal cannot run with this model. They are dropped by the sweep.
- The other experiments use `harness/bench.py` directly. On this cluster its 20-iteration warm-up reads about 20% high for sub-millisecond engines.
