#!/bin/bash
# Runs INSIDE the container with the overlay mounted read/write (see setup_env.sbatch).
# Follows NYU's "Singularity with Conda" guide: Miniforge in /ext3, then pip packages.
set -euo pipefail

TORCH_TRT_VERSION=${TORCH_TRT_VERSION:-2.14.0}

if [ ! -d /ext3/miniforge3 ]; then
  cd "${SINGULARITY_TMPDIR:-/tmp}"
  url=https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
  if command -v wget >/dev/null; then wget -q "$url"; else curl -fsSL -O "$url"; fi
  bash Miniforge3-Linux-x86_64.sh -b -p /ext3/miniforge3
  rm Miniforge3-Linux-x86_64.sh
fi

cat > /ext3/env.sh <<'EOS'
#!/bin/bash
unset -f which
source /ext3/miniforge3/etc/profile.d/conda.sh
export PATH=/ext3/miniforge3/bin:$PATH
export PYTHONPATH=/ext3/miniforge3/bin:$PATH
EOS

source /ext3/env.sh
python --version
# torch-tensorrt pins the matching torch and tensorrt versions itself.
pip install --no-cache-dir "torch-tensorrt==${TORCH_TRT_VERSION}" transformers numpy pandas scipy matplotlib pyyaml ipykernel

# torch_tensorrt needs a GPU just to be imported, and this job runs on a CPU node,
# so report the installed versions from package metadata instead of importing.
python - <<'EOS'
from importlib.metadata import version
for pkg in ("torch", "torch-tensorrt", "tensorrt-cu13", "transformers", "numpy"):
    try:
        print(pkg, version(pkg))
    except Exception as e:
        print(pkg, "not found:", e)
EOS
pip freeze > "${FREEZE_OUT:-/tmp/pip-freeze.txt}"
du -sh /ext3
df -h /ext3 | tail -1
