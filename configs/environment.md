# Environment

Recorded from the partition-check job on 2026-11-12 (`experiments/e4_workload_selector/partition_check.py`).

- Cluster: NYU HPC Cloud Bursting (Open OnDemand), Slurm account `csci_ga_3033-2026fa`
- Partition: `c12m85-a100-1` (use this one for every measurement)
- GPU: NVIDIA A100-SXM4-40GB, driver 610.43.02
- Container image: `/share/apps/images/cuda13.0.1-cudnn9.13.0-ubuntu-24.04.3.sif` (Singularity CE 4.5, Ubuntu 24.04)
- Python: 3.14 (Miniforge, in the overlay at `/scratch/$USER/trt/trt-env.ext3`)
- PyTorch: 2.14.1+cu130 (CUDA 13.0)
- Torch-TensorRT: 2.14.0
- TensorRT: 11.1.0.106
- transformers: 5.22.2

Build the same environment with `sbatch slurm/setup_env.sbatch` (CPU only, about 10 minutes).

## Things this cluster does that affect measurements

- **Low-utilization jobs are cancelled.** A watchdog (`/share/apps/admins/scripts/check_gpu_utilizations.bash`) cancels a GPU job whose mean GPU utilization is 5% or less once the job is about 30 minutes old; `sacct` shows `CANCELLED by 0`. A sweep that mostly compiles engines can fall under that. `harness/sweep.py` is resumable, so chain jobs: `sbatch --dependency=afterany:<previous job id> slurm/sweep.sbatch`.
- **The GPU is slow for the first moments after an idle spell.** After a compile (CPU-bound), GPU-bound configs measured about 20% slower for the first few hundred milliseconds of load, then switched to the faster level and stayed there. The shared `warmup_iters: 20` is too short to get past this for sub-millisecond engines, so `harness/sweep.py` also keeps the GPU busy for `settle_s` seconds before timing.
- **Engine builds share a timing cache.** The first build of a session took 39 s and later builds of the same graph 6 to 13 s, so `compile_s` depends on what was built before.
