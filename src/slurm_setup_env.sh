#!/bin/bash
#SBATCH --job-name=fake_review_env
#SBATCH --partition=gpu
#SBATCH --nodelist=ant2
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=logs/env_%j.out
#SBATCH --error=logs/env_%j.err

# Build the BERT venv ON A COMPUTE NODE:   sbatch src/slurm_setup_env.sh
#
# Do NOT run `pip install -r requirements.txt` in a login shell or a code-server terminal.
# torch pulls the bundled CUDA runtime (nvidia-cublas, nvidia-cudnn, ...), several GB of
# wheels to download, unpack and byte-compile, and the resolver holds much of it in RAM.
# On a shared interactive node that OOMs the cgroup and takes code-server down with it.
#
# This runs once. Afterwards slurm_bert.sh finds the venv and skips straight to training.

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO"

VENV="${VENV:-$HOME/.venvs/fake_reviews_bert}"

# Keep pip's scratch and cache off $HOME (quota) and off the node's small /tmp. Node-local
# scratch keyed by job id, cleaned up on exit.
SCRATCH="${SCRATCH:-/tmp/pip_${SLURM_JOB_ID:-$$}}"
export TMPDIR="$SCRATCH/tmp"
export PIP_CACHE_DIR="$SCRATCH/cache"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$REPO/logs"
trap 'rm -rf "$SCRATCH"' EXIT

echo "======================================"
echo "Job ${SLURM_JOB_ID:-local} on $(hostname)  |  $(date)"
echo "venv=$VENV"
echo "TMPDIR=$TMPDIR  PIP_CACHE_DIR=$PIP_CACHE_DIR"
echo "======================================"

if [ -d "$VENV" ]; then
  echo "venv already exists at $VENV"
  echo "Delete it first if you want a clean rebuild:  rm -rf $VENV"
else
  echo "Creating venv..."
  python3 -m venv "$VENV"
fi

PIP="$VENV/bin/pip"

"$PIP" install --no-cache-dir --upgrade pip

# torch first and alone. It is by far the largest install, and giving it its own resolver
# pass keeps peak memory well below what a single combined resolve would need.
echo "--- installing torch ---"
"$PIP" install --no-cache-dir torch

echo "--- installing the rest ---"
"$PIP" install --no-cache-dir -r "$REPO/requirements.txt"

echo "--- verifying ---"
"$VENV/bin/python" - <<'EOF'
import torch, transformers, datasets, sklearn, yaml, pandas
print(f"torch        {torch.__version__}")
print(f"transformers {transformers.__version__}")
print(f"datasets     {datasets.__version__}")
print(f"sklearn      {sklearn.__version__}")
print(f"cuda available: {torch.cuda.is_available()}"
      + (f"  device: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else ""))
EOF

echo "======================================"
echo "venv ready: $VENV"
echo "Next:  sbatch src/slurm_bert.sh"
echo "Job finished at $(date)"
echo "======================================"
