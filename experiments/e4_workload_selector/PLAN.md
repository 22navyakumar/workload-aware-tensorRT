# E4 plan — workload-aware selector (Tanish)

Goal: take the latency measurements the team collects, build a selector that picks a
Torch-TensorRT configuration for a given batch size and sequence length, and show how close
it gets to the best tested configuration and what the tuning costs.

Written 2026-11-11 from the proposal, the starter repo and the Torch-TensorRT docs.
Nothing here has been run on a GPU yet; the harness issues below come from reading the code
and docs only.

## Findings that shape the plan

- **All GPU work runs on NYU HPC Cloud Bursting** (see "Compute" below). Torch-TensorRT 2.14
  has wheels for Linux and Windows only and needs an NVIDIA GPU, so nothing compiles on
  macOS. The selector itself is pandas/scikit-learn over CSVs and can be built anywhere.
- **The swept config space may do nothing.** `configs/workloads.yaml` only varies
  `min_block_size`. If DistilBERT converts fully (likely, unverified), every value gives the
  same single engine and there is nothing to select between.
- **`seq_len: 1024` will fail on DistilBERT**, which has 512 position embeddings. The team
  needs a model that supports 1024 or a shorter grid.
- **The harness does not record what E4 needs.** No compile time (the "tuning cost"), no
  partition counts, no numerical agreement, no sweep runner, no shared results format.
- **`enabled_precisions` in `harness/compile.py` looks outdated.** Current docs say typing
  follows the model's dtype in the Dynamo path, and the model is already loaded as FP16.
  Verify against whichever version the team pins.
- **16 workloads is very little data** for a held-out-shapes test.

## Compute: NYU HPC Cloud Bursting

- Access: Open OnDemand at https://ood.burst.hpc.nyu.edu/ (NYU VPN off campus). Jupyter,
  a terminal inside Jupyter, file upload, and `sbatch` from that terminal.
- Slurm account (every job): `csci_ga_3033-2026fa`. 300 GPU hours per student.
- Partition for all measurements: `c12m85-a100-1` (1 × A100 40GB). Every team member must
  use this same partition so numbers are comparable. `interactive` for setup,
  `n2c48m24` (CPU only) for selector fitting if needed.
- A100 40GB should fit batch 16 × length 1024 for a base-size model, so the memory caveat
  in the proposal is unlikely to bite; the DistilBERT 512-position limit still applies.

### One-time environment setup (per NYU's Apptainer + Conda guide)

> Written before the first cluster session. What actually works is in `slurm/` (the cluster has
> `singularity`, not `apptainer`, and `--fakeroot` is too slow); see [`LOG.md`](LOG.md).
Paths below are from the general HPC guide, which has no burst-specific notes. Check that
`/share/apps/overlay-fs-ext3` and `/share/apps/images` exist on the burst cluster first.

```bash
mkdir -p /scratch/$USER/trt && cd /scratch/$USER/trt
cp -rp /share/apps/overlay-fs-ext3/overlay-15GB-500K.ext3.gz . && gunzip overlay-15GB-500K.ext3.gz
ls /share/apps/images/            # pick a CUDA image that matches the pinned torch build
apptainer exec --fakeroot --overlay overlay-15GB-500K.ext3:rw /share/apps/images/<cuda-image>.sif /bin/bash
# inside the container:
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p /ext3/miniforge3
# create /ext3/env.sh exactly as in the NYU guide, then:
source /ext3/env.sh
conda install pip ipykernel -y
pip install -r requirements.txt   # after versions are pinned
```

Install packages from a compute node (`srun ... --pty /bin/bash`), not the login node.
Batch jobs mount the overlay read-only and pass the GPU through:

```bash
#SBATCH --account=csci_ga_3033-2026fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1              # confirm the GPU request syntax on the burst cluster
apptainer exec --nv --overlay /scratch/$USER/trt/overlay-15GB-500K.ext3:ro \
  /share/apps/images/<cuda-image>.sif \
  /bin/bash -c "source /ext3/env.sh; python -m harness.sweep ..."
```

An overlay open read/write cannot be opened by any other job, so finish installing before
launching sweeps. Each team member needs their own overlay copy or a shared read-only one.

### GPU-hour budget (rough estimate, not measured)
About 20 configs × 25 workloads = 500 compiles. At 1–3 minutes each plus a few seconds of
timing, that is roughly 10–25 GPU hours for a full sweep, well inside 300. Time the first
few compiles and revise. Run timing jobs one at a time on the node; do not share the GPU
with another process while benchmarking.

## Steps

### 1. Agree with the team before anyone collects data
- Everyone uses the `c12m85-a100-1` partition, the same container image and pinned versions
  (`requirements.txt` is unpinned); fill in `configs/environment.md`.
- The model and the sequence-length grid.
- One results CSV schema: `config_id, batch, seq_len, median_ms, p95_ms, peak_mem_mb,
  compile_s, n_trt_blocks, n_torch_blocks, max_abs_diff`.

### 2. Partition check
Compile with `dryrun=True` at default settings and count the blocks. If it is one engine,
widen the config space with knobs that change the graph or the runtime:
- `torch_executed_ops` (forced breaks, which E3 already needs)
- `use_fast_partitioner`
- `optimization_level`
- CUDA Graphs on or off
- PyTorch eager as a selectable option

### 3. Sweep runner (`harness/sweep.py`)
For each config × workload: compile, time the compile, benchmark, check agreement against
eager, append one CSV row. E4 is its main consumer; E1 and E2 can reuse it.

### 4. Selector (developed locally, on a synthetic CSV until real data arrives)
- Lookup table: nearest measured workload in log(batch), log(seq_len) space.
- Regression: one latency model per config on features such as batch × length and
  batch × length², then pick the lowest prediction.
- Baselines: eager, default Torch-TensorRT, one fixed tuned config (best on average over
  training shapes), and the oracle (best tested config per workload).

### 5. Evaluation on held-out shapes
Add a few off-grid shapes as the test set (for example batch 2 and 12, lengths 192, 384,
768), plus leave-one-out over the grid. Report:
- Latency regret against the oracle.
- Speedup over default and over the fixed tuned config.
- How often the selector picks the true best.
- Regret against tuning budget (how many workloads had to be measured).

Repeat each measurement about three times so config differences can be told apart from
noise.

### 6. Write-up
One results table, the regret-vs-budget figure, and a selector decision map that lines up
with the E2 heatmap. If no config beats default meaningfully, that is a valid result under
the proposal; explain it with the E3 traces.

## Risks
- E4 depends on E1/E2 data. Steps 3 and 4 are how to avoid being blocked.
- No deadline is recorded here; steps are ordered by dependency, not by date.

## Sources
- https://docs.pytorch.org/TensorRT/contributors/partitioning.html
- https://docs.pytorch.org/TensorRT/user_guide/performance_tuning.html
- https://docs.pytorch.org/TensorRT/_sources/user_guide/compilation/compilation_settings.rst.txt
- https://pypi.org/project/torch-tensorrt/
