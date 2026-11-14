# E4 — Workload-aware selection (Tanish)

**Plan:** Use part of the benchmark results to build a lookup-table or regression-based selector, then test on held-out workload shapes.

**Deliverable:** Cost-based selector and speedup / tuning-cost results.

Use the shared harness (`harness/`) and `configs/workloads.yaml`. Write outputs to `results/e4_workload_selector/`.

**Result:** [`results/e4_workload_selector/README.md`](../../results/e4_workload_selector/README.md). One fixed tuned config matches the best tested config on held-out shapes; a per-workload selector adds nothing for DistilBERT on the A100.

## How to run

Plan and background: [`PLAN.md`](PLAN.md). What has been run so far and what it showed: [`LOG.md`](LOG.md).
On the cluster, from the repo root (see `slurm/`):

```bash
sbatch slurm/partition_check.sbatch                          # 1. how each config partitions the model (no engines built)
sbatch --time=00:30:00 slurm/sweep.sbatch --limit 1          # 2. smoke test: every config on one workload
J=$(sbatch --parsable --requeue slurm/sweep.sbatch)          # 3. full sweep -> results/sweep/sweep.csv (resumable)
for i in 1 2 3 4 5; do J=$(sbatch --parsable --requeue --dependency=afterany:$J slurm/sweep.sbatch); done
```

Step 3 is a chain of jobs because the cluster cancels GPU jobs with low GPU utilization after about
30 minutes (see `configs/environment.md`). Each job continues where the previous one stopped; jobs
left over once the sweep is complete exit after finding nothing to do.

Anywhere (no GPU needed):

```bash
python experiments/e4_workload_selector/evaluate.py --csv results/sweep/sweep.csv
```

writes `summary.csv`, `per_workload.csv`, `budget.csv` and `figures/` to `results/e4_workload_selector/`.

| File | What it does |
|------|--------------|
| `partition_check.py` | Dry-run compile of every config; block counts, ops per PyTorch block, software versions |
| `selector.py` | Fixed, best-fixed, lookup and regression selectors |
| `evaluate.py` | Held-out and leave-one-out comparison against the baselines; regret vs tuning budget |
| `figures.py` | Report figures |
| `make_synthetic.py` | Fake sweep CSV for testing the pipeline without a GPU; never report its numbers |

Configs, scenarios and held-out shapes are defined in `configs/workloads.yaml`.
