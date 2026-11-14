# E4 results: workload-aware selection

**A workload-aware selector is not worth having for DistilBERT on this A100.** One fixed tuned config (default settings plus CUDA Graphs) is within 0.5% of the best tested config on every held-out shape, and the lookup and regression selectors only match it or do slightly worse. The gain comes from tuning at all, not from tuning per workload: 1.11× over default Torch-TensorRT on the unmodified model, and 1.6× to 2.3× when an operator falls back to PyTorch.

Data: `results/sweep/sweep.csv` (job 29729, one A100-SXM4-40GB node, 2026-11-12). 17 configs × 20 shapes, FP16, 5 timing repeats of 100 iterations after a 2 s warm-up. Reproduce with `python experiments/e4_workload_selector/evaluate.py --csv results/sweep/sweep.csv`.

## Setup

- **Shapes.** Training grid: batch 1, 4, 8, 16 × 128, 256, 512 tokens (12 shapes). Held-out: 8 shapes off the grid, from 1 × 384 to 16 × 448. Shapes above 512 tokens are not possible with DistilBERT.
- **Scenarios.** DistilBERT converts to one TensorRT engine, so partitioning has nothing to tune on the unmodified model (`natural`). Two more scenarios force an operator back to PyTorch to create a real partitioning problem: `break_gelu` (6 PyTorch blocks) and `break_layernorm` (7 to 13).
- **Methods.** PyTorch eager; default Torch-TensorRT for the scenario; one fixed tuned config (best geometric-mean latency on the training grid); a lookup selector (nearest training shape); a regression selector (per-config latency model); and the best tested config per shape.
- **Noise floor.** Re-timing the same engine moves the median by 0.7%; building the same config twice by 0.3%. Differences smaller than about 1% mean nothing.

## Held-out shapes

Extra latency over the best tested config (geometric mean over the 8 held-out shapes; lower is better), and speedup over PyTorch eager.

| Method | `natural` | `break_gelu` | `break_layernorm` |
|---|---|---|---|
| PyTorch eager | +210% | +181% | +154% |
| Default Torch-TensorRT | +11.0% (2.79×) | +61.3% (1.74×) | +132.7% (1.09×) |
| One fixed tuned config | +0.0% (3.10×) | +0.0% (2.81×) | +0.5% (2.52×) |
| Lookup selector | +0.0% (3.10×) | +3.2% (2.72×) | +1.4% (2.50×) |
| Regression selector | +1.5% (3.05×) | +0.5% (2.79×) | +0.0% (2.54×) |
| Best tested config | 0 (3.10×) | 0 (2.81×) | 0 (2.54×) |

- The fixed tuned config is the CUDA Graphs variant in all three scenarios (`trt_cudagraphs`, `gelu_cudagraphs`, `ln_cudagraphs`).
- Speedup of the selectors over the fixed tuned config: 0.97× to 1.01×. The only case where a selector wins is `break_layernorm`, by 0.5%, which is inside the noise.
- Leave-one-out over the 12 grid shapes gives the same picture (`summary.csv`, rows with `eval = loo`).

## Why the selector has nothing to select

The best config barely depends on the workload.

- `natural`: `trt_cudagraphs` is fastest on all 20 shapes.
- `break_gelu` and `break_layernorm`: the CUDA Graphs variant is fastest on every shape except the largest ones (16 × 512, and 16 × 448 for LayerNorm), where plain PyTorch is 1% to 11% faster.

That single crossover is the only workload-dependent decision in the data, and the selectors mostly get it wrong: of the 7 times a selector chose PyTorch for a held-out shape, 2 were correct. The worst case is the lookup selector sending 12 × 384 to PyTorch in `break_gelu` because its nearest training shape is 16 × 512, which costs 23% on that shape.

## What the tuning knobs are worth

Geometric mean over all 20 shapes, with the range across shapes.

| Change | Effect on latency |
|---|---|
| Default TensorRT instead of PyTorch eager | 2.68× faster (6.6× at 1 × 128, 1.07× at 16 × 512) |
| CUDA Graphs on the single engine | 1.11× faster (1.20× at small shapes, 1.02× at 16 × 512) |
| `optimization_level` 5 instead of 3 | no change (0.998×) |
| `optimization_level` 1 instead of 3 | 6% slower, builds in 8 s instead of 14 s |
| GELU forced to PyTorch (6 engines) | 1.63× slower than one engine |
| LayerNorm forced to PyTorch (7 engines) | 2.49× slower than one engine; about the same as eager |
| `min_block_size` 1 instead of 5 | GELU: 1% faster. LayerNorm: 5% faster (13 engines instead of 7) |
| Global partitioner instead of fast | GELU: 2% slower. LayerNorm: fails to build |
| `min_block_size` large enough to drop most engines | 1.3× to 1.7× slower; close to or worse than eager |
| CUDA Graphs on a partitioned model | GELU: 1.63× faster. LayerNorm: 2.26× faster |

- **The cost of a partitioned model is launch overhead, not GPU work.** PyTorch eager takes 3.2 to 4.1 ms on every shape below 16 × 448, almost regardless of size, and each PyTorch fallback block brings part of that fixed cost back. CUDA Graphs removes most of it: with CUDA Graphs on both, the GELU-partitioned model is 11% slower than the single engine and the LayerNorm one 23% slower.
- **`min_block_size` and the partitioner choice move latency by 5% or less** as long as the engines survive. They matter only when a setting drops engines altogether.
- **Numerical agreement.** Cosine similarity against eager is at least 0.99999 for every run; the largest absolute difference is 0.02 (FP16).

## Tuning cost

Cost of measuring every config of a scenario on training shapes (build, first call, warm-up, timing), and how many requests it takes to earn that back against the default config (mean saving per request on held-out shapes).

| | `natural` | `break_gelu` | `break_layernorm` |
|---|---|---|---|
| Saving per request vs default | 0.12 ms | 0.70 ms | 1.76 ms |
| Full 12-shape grid | 16.6 GPU-min | 30.1 GPU-min | 33.3 GPU-min |
| Break-even, full grid | 8.3 M requests | 2.6 M | 1.1 M |
| Two shapes only | 2.7 GPU-min | 5.1 GPU-min | 5.5 GPU-min |
| Break-even, two shapes | 1.4 M requests | 0.43 M | 0.19 M |

Measuring two training shapes is enough to find the fixed tuned config (within 0.5% of the best). The other ten shapes of the grid buy nothing; `budget.csv` and `figures/budget_*.png` show the full curve. With a single shape the result depends on which one: if it is 16 × 512, every method picks PyTorch eager.

## Limits

- One model, one GPU, one node, FP16 only. A model with operators TensorRT cannot convert would give the `break_*` scenarios for real rather than by forcing.
- The broken scenarios are artificial: both operators convert fine, so nobody would exclude them in practice.
- Build times share a timing cache within the job, so `compile_s` for a config depends on what was built before it. The first build took 40 s; the same graph later took 6 to 13 s.
- `ln_global` failed to build on every shape (`AttributeError: 'GraphModule' object has no attribute '_frozen_param0'`) and is excluded.
- `peak_mem_mb` counts PyTorch's allocator only, not memory held by TensorRT engines, so it is not a fair memory comparison.

## Files

| File | Contents |
|---|---|
| `summary.csv` | One row per scenario, evaluation and method: regret, near-best rate, speedups, tuning cost, break-even |
| `per_workload.csv` | The config each method chose on each test shape and its latency |
| `budget.csv` | Selector regret when only k of the 12 grid shapes are measured |
| `noise.csv` | Timing and build noise floor |
| `partition_check.json` | How each config partitions the model (blocks and operators per block) |
| `figures/baselines_*.png` | Methods compared per scenario, held-out and leave-one-out |
| `figures/choices_*.png` | Chosen config per shape for each selector |
| `figures/budget_*.png` | Regret against tuning budget |
