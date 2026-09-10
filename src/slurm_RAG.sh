#!/bin/bash
#SBATCH --job-name=fake_review_rag
#SBATCH --partition=gpu
#SBATCH --nodelist=ant1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/rag_%j.out
#SBATCH --error=logs/rag_%j.err

# Submit from the repo root:   sbatch src/slurm_RAG.sh
#
# Runs the retrieval-augmented LLM classifier over every dataset x model pair in
# src/config/rag.yaml. Full volume is 3 datasets x 398 test rows x 4 models = 4776 LLM
# calls, roughly two hours.
#
# SMOKE RUN FIRST. Set limit_test_rows in the config, or:
#     CONFIG=src/config/rag_smoke.yaml sbatch src/slurm_RAG.sh
# Two hours is a long time to discover the model tag was wrong.
#
# Other overrides:  CONFIG=... , VENV=... , REBUILD_VENV=1

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO"

CONFIG="${CONFIG:-src/config/rag.yaml}"
OLLAMA_DIR="${OLLAMA_DIR:-$HOME/ollama}"        # model cache, persists across jobs
OLLAMA_SIF="${OLLAMA_SIF:-$HOME/ollama.sif}"

# A venv of its own. NOT $HOME/.venvs/fake_reviews -- that one holds only
# pandas/requests/tqdm for generation, and the guard below is `[ ! -d ]`, so pointing here
# at it would skip the install entirely and die on `ModuleNotFoundError:
# sentence_transformers`. Same trap slurm_bert.sh documents.
VENV="${VENV:-$HOME/.venvs/fake_reviews_rag}"

# Per-job port. A fixed 11434 collides with any other Ollama on this node, and the
# readiness check would then pass against THAT server -- silently classifying with
# whatever model it happens to be serving.
PORT=$(( 11434 + (${SLURM_JOB_ID:-0} % 1000) ))

echo "======================================"
echo "Job ${SLURM_JOB_ID:-local} on $(hostname)  |  $(date)"
echo "config=$CONFIG  port=$PORT"
echo "======================================"
nvidia-smi || echo "WARNING: nvidia-smi unavailable"

mkdir -p "$OLLAMA_DIR" "$REPO/logs" "$REPO/results/rag"

if [ ! -f "$OLLAMA_SIF" ]; then
  echo "ERROR: Apptainer image not found at $OLLAMA_SIF"
  echo "Build it once on a login node:  apptainer pull \"$OLLAMA_SIF\" docker://ollama/ollama:latest"
  exit 1
fi

# -----------------------------
# OLLAMA SERVER
# -----------------------------
echo "Starting Ollama server..."
apptainer exec --nv \
  --env OLLAMA_HOST=0.0.0.0:$PORT \
  --bind "$OLLAMA_DIR":/root/.ollama \
  "$OLLAMA_SIF" \
  ollama serve &
OLLAMA_PID=$!

# Trap, not a trailing kill: on a crash a --time=06:00:00 reservation would otherwise hold
# the GPU with a live server for the full allocation.
cleanup() {
  echo "Stopping Ollama (pid $OLLAMA_PID)..."
  kill "$OLLAMA_PID" 2>/dev/null || true
  wait "$OLLAMA_PID" 2>/dev/null || true
}
trap cleanup EXIT

echo "Waiting for Ollama on port $PORT..."
for _ in $(seq 1 60); do
  if curl -sf "http://localhost:$PORT/api/tags" > /dev/null; then
    echo "Ollama is ready."
    break
  fi
  if ! kill -0 "$OLLAMA_PID" 2>/dev/null; then
    echo "ERROR: Ollama server exited during startup (port $PORT already in use?)"
    exit 1
  fi
  sleep 2
done
curl -sf "http://localhost:$PORT/api/tags" > /dev/null || {
  echo "ERROR: Ollama did not become ready within 120s"
  exit 1
}

# -----------------------------
# PYTHON ENV
# -----------------------------
if [ -n "${REBUILD_VENV:-}" ]; then rm -rf "$VENV"; fi
if [ ! -d "$VENV" ]; then
  echo "Creating venv at $VENV..."
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install --quiet --upgrade pip
  # sentence-transformers drags in torch and the bundled CUDA wheels -- several GB. Done
  # here inside the job, never interactively on a shared login node.
  "$VENV/bin/pip" install --quiet \
    pandas numpy scikit-learn PyYAML requests sentence-transformers \
    langchain-core langchain-community langchain-huggingface langchain-ollama faiss-cpu
fi
PYTHON="$VENV/bin/python"
echo "Python: $($PYTHON --version) at $PYTHON"

# -----------------------------
# PULL EVERY MODEL THE CONFIG NAMES
# -----------------------------
# Up front, not lazily: a missing tag discovered 90 minutes in wastes the whole run.
MODELS=$("$PYTHON" -c "import yaml,sys; print('\n'.join(yaml.safe_load(open('$CONFIG'))['models']))")
echo "Models from $CONFIG:"; echo "$MODELS" | sed 's|^|  |'

while IFS= read -r MODEL; do
  [ -z "$MODEL" ] && continue
  echo "Ensuring $MODEL is available..."
  apptainer exec --nv \
    --env OLLAMA_HOST=0.0.0.0:$PORT \
    --bind "$OLLAMA_DIR":/root/.ollama \
    "$OLLAMA_SIF" \
    ollama pull "$MODEL"
  # `ollama pull` can report progress and still fail on a full quota, and the classifier
  # would then 404 on every single call and score the dataset as all-unknown.
  curl -sf "http://localhost:$PORT/api/tags" | grep -q "$MODEL" || {
    echo "ERROR: $MODEL not present after pull"
    exit 1
  }
done <<< "$MODELS"

# -----------------------------
# CLASSIFY
# -----------------------------
export OLLAMA_HOST="localhost:$PORT"

echo "======================================"
echo "Running RAG classification..."
echo "======================================"
"$PYTHON" src/rag_classifier/RAG.py --config "$CONFIG"

echo "======================================"
echo "Finished at $(date). Results:"
ls -1 "$REPO/results/rag/" | sed 's|^|  |'
echo "======================================"
