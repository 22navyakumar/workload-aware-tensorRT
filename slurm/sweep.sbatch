#!/bin/bash
#SBATCH --job-name=trt-sweep
#SBATCH --account=csci_ga_3033-2026fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48GB
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%j.out

# Submit from the repo root:  sbatch slurm/sweep.sbatch [extra harness.sweep arguments]
# Smoke test first:           sbatch --time=00:30:00 slurm/sweep.sbatch --limit 1
# The sweep is resumable; resubmit the same command if the job runs out of time.
source slurm/env.sh
REPO=$SLURM_SUBMIT_DIR
nvidia-smi
in_container python -m harness.sweep --out results/sweep/sweep.csv "$@"
