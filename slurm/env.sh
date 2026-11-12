# Shared settings for the Slurm jobs on NYU HPC Cloud Bursting. Source this file.
export SLURM_ACCOUNT_NAME=csci_ga_3033-2026fa
export GPU_PARTITION=c12m85-a100-1                       # 1 x A100 40GB; use for every measurement
export WORK=/scratch/$USER/trt
export OVERLAY=${OVERLAY:-$WORK/trt-env.ext3}
export IMAGE=${IMAGE:-/share/apps/images/cuda13.0.1-cudnn9.13.0-ubuntu-24.04.3.sif}
export REPO=${REPO:-$WORK/workload-aware-tensorRT}
export HF_HOME=$WORK/hf_cache                            # model downloads go to scratch, not $HOME
export SINGULARITY_CACHEDIR=$WORK/.singularity/cache     # the cluster default points at a missing /tmp dir
export SINGULARITY_TMPDIR=$WORK/.singularity/tmp
mkdir -p "$SINGULARITY_CACHEDIR" "$SINGULARITY_TMPDIR" "$HF_HOME"

# Run a command inside the container with the GPU and the read-only environment.
in_container() {
  singularity exec --nv --overlay "$OVERLAY:ro" "$IMAGE" \
    /bin/bash -c "source /ext3/env.sh; export HF_HOME=$HF_HOME; cd $REPO; $*"
}
