#!/bin/bash
#SBATCH --job-name=trt-partition-check
#SBATCH --account=csci_ga_3033-2026fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%x-%j.out

# Submit from the repo root:  sbatch slurm/partition_check.sbatch
source slurm/env.sh
REPO=$SLURM_SUBMIT_DIR
nvidia-smi
in_container python experiments/e4_workload_selector/partition_check.py
