#!/bin/bash
#SBATCH --job-name=fake_review_bert
#SBATCH --partition=gpu
#SBATCH --nodelist=ant2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/bert_%j.out
#SBATCH --error=logs/bert_%j.err

# Submit from the repo root:   sbatch src/slurm_bert.sh
# Smaller grid / other config: GRID=src/config/bert_smoke.yaml sbatch src/slurm_bert.sh
# Build the venv and stop:     SETUP_ONLY=1 sbatch src/slurm_bert.sh
# Force a clean rebuild:       REBUILD_VENV=1 sbatch src/slurm_bert.sh
#
# The venv is built here, inside the job, on the first run. Never install the requirements
# interactively: torch drags in the bundled CUDA runtime (nvidia-cublas, nvidia-cudnn, ...),
# several GB of wheels to download, unpack and byte-compile, and the resolver holds much of
# it in RAM. On a shared login node that OOMs the cgroup and takes code-server down with it.

set -euo pipefail

# -----------------------------
# CONFIG
# -----------------------------
REPO="${SLURM_SUBMIT_DIR:-$PWD}"

# train.py's --config default is src/config/config.yaml and data.path inside it is
# data/..., both relative to the repo root; `-m src.bert_classifier.train` also needs the
# root on sys.path. Everything below therefore runs from here.
cd "$REPO"

CONFIG="${CONFIG:-src/config/config.yaml}"
GRID="${GRID:-src/config/bert_grid.yaml}"

# A venv of its own, NOT the generation one. slurm_synthetic_data.sh guards its install
# with [ ! -d "$VENV" ], and $HOME/.venvs/fake_reviews already exists holding only
# pandas/requests/tqdm -- pointing here at that path would skip the install entirely and
# die on `ModuleNotFoundError: datasets`.
VENV="${VENV:-$HOME/.venvs/fake_reviews_bert}"

# Persist the model cache so bert-base-uncased is downloaded once, not once per job.
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

RESULTS="$REPO/results/bert"

echo "======================================"
echo "Job ${SLURM_JOB_ID:-local} on $(hostname)  |  $(date)"
echo "config=$CONFIG  grid=$GRID"
echo "venv=$VENV  HF_HOME=$HF_HOME"
echo "output=$RESULTS"
echo "======================================"
nvidia-smi || echo "WARNING: nvidia-smi unavailable"

mkdir -p "$REPO/logs" "$RESULTS" "$HF_HOME"

# -----------------------------
# PYTHON ENV
# -----------------------------
[ -n "${REBUILD_VENV:-}" ] && { echo "REBUILD_VENV set, removing $VENV"; rm -rf "$VENV"; }

if [ ! -d "$VENV" ]; then
  echo "--- building venv at $VENV ---"

  # Keep pip's scratch and cache off $HOME (quota) and off the node's small /tmp root.
  # Node-local, keyed by job id, removed when this block finishes.
  SCRATCH="/tmp/pip_${SLURM_JOB_ID:-$$}"
  export TMPDIR="$SCRATCH/tmp"
  export PIP_CACHE_DIR="$SCRATCH/cache"
  mkdir -p "$TMPDIR" "$PIP_CACHE_DIR"

  python3 -m venv "$VENV"
  "$VENV/bin/pip" install --no-cache-dir --upgrade pip

  # torch alone first. It is by far the largest install, and giving it its own resolver
  # pass keeps peak memory well below a single combined resolve over the whole file.
  echo "--- installing torch ---"
  "$VENV/bin/pip" install --no-cache-dir torch

  echo "--- installing the rest ---"
  "$VENV/bin/pip" install --no-cache-dir -r "$REPO/requirements.txt"

  rm -rf "$SCRATCH"
  echo "--- venv ready ---"
else
  echo "Reusing existing venv at $VENV"
fi

PYTHON="$VENV/bin/python"
echo "Python: $($PYTHON --version) at $PYTHON"

# Import everything the grid needs now, so a half-built venv fails here in seconds rather
# than partway into training.
"$PYTHON" - <<'EOF'
import torch, transformers, datasets, sklearn, yaml, pandas
print(f"torch        {torch.__version__}")
print(f"transformers {transformers.__version__}")
print(f"datasets     {datasets.__version__}")
print(f"sklearn      {sklearn.__version__}")
print(f"cuda available: {torch.cuda.is_available()}"
      + (f"  device: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else ""))
EOF

if [ -n "${SETUP_ONLY:-}" ]; then
  echo "======================================"
  echo "SETUP_ONLY set; venv ready at $VENV, skipping the grid."
  echo "Next:  sbatch src/slurm_bert.sh"
  echo "======================================"
  exit 0
fi

# -----------------------------
# RUN THE GRID
# -----------------------------
# The corpus CSV under data/ is a read-only input; the job only reads it.
echo "Running BERT grid..."

"$PYTHON" -m src.bert_classifier.train \
  --config "$CONFIG" \
  --grid "$GRID"

# -----------------------------
# SUMMARY
# -----------------------------
# The grid writes validation_results.json before the final test evaluation, so a job that
# died in between leaves the first file and not the second. Say which happened here rather
# than making it a directory listing after the fact.
echo "======================================"
if [ -f "$RESULTS/final_test_results.json" ]; then
  echo "Test metrics written:"
  "$PYTHON" -c "
import json, sys
d = json.load(open('$RESULTS/final_test_results.json'))
m = d['test_metrics']
print('  best params:', d['best_validation_params'])
for k in ('accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc'):
    v = m.get(k, m.get('eval_' + k))
    if v is not None:
        print(f'  {k:<10} {v:.4f}')
"
else
  echo "WARNING: $RESULTS/final_test_results.json missing -- the run did not complete."
  [ -f "$RESULTS/validation_results.json" ] && echo "  (validation_results.json exists, so it died during the final test evaluation)"
fi
echo "Job finished at $(date)"
echo "======================================"
