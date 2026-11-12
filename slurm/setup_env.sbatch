#!/bin/bash
#SBATCH --job-name=trt-setup-env
#SBATCH --account=csci_ga_3033-2026fa
#SBATCH --partition=n2c48m24
#SBATCH --cpus-per-task=8
#SBATCH --mem=20GB
#SBATCH --time=01:30:00
#SBATCH --output=slurm-%x-%j.out

# One-time environment build (CPU only, no GPU hours). Submit from the repo root:
#   sbatch slurm/setup_env.sbatch
# The interactive nodes have 1.5 GB of memory, too little for installing PyTorch.
set -euo pipefail
source slurm/env.sh
REPO=$SLURM_SUBMIT_DIR
TEMPLATE=/share/apps/overlay-fs-ext3/overlay-25GB-500K.ext3.gz

if [ ! -f "$OVERLAY" ]; then
  cp "$TEMPLATE" "$OVERLAY.gz"
  gunzip "$OVERLAY.gz"
fi
export FREEZE_OUT=$REPO/configs/pip-freeze.txt
# A plain read/write overlay mount is fast. --fakeroot (what NYU's guide shows) also works but
# first unpacks the whole image into a sandbox, so it is only the fallback, on node-local disk.
if singularity exec --overlay "$OVERLAY:rw" "$IMAGE" touch /ext3/.rw_test; then
  RW_FLAGS=
else
  RW_FLAGS=--fakeroot
  export SINGULARITY_TMPDIR=/tmp/$USER-singularity && mkdir -p "$SINGULARITY_TMPDIR"
fi
echo "read/write flags: '${RW_FLAGS}'"
singularity exec $RW_FLAGS --overlay "$OVERLAY:rw" "$IMAGE" /bin/bash "$REPO/slurm/setup_env.sh"
echo SETUP-OK
