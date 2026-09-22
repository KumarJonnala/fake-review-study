#!/bin/bash
#SBATCH --job-name=fake_review_zeroshot
#SBATCH --partition=gpu
#SBATCH --nodelist=ant1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/zeroshot_%j.out
#SBATCH --error=logs/zeroshot_%j.err

# Submit from the repo root:   sbatch src/slurm_zeroshot.sh
#
# Runs the ZERO-SHOT LLM classifier over every dataset x model pair in
# src/config/zeroshot.yaml -- the control condition for slurm_RAG.sh. Same prompt, same
# judges, same test rows, no retrieved examples.
#
# Full volume is 4145 test rows x 4 models = 16580 LLM calls. RAG measured ~1.5 s/call at
# ~785 prompt words; dropping the five examples leaves ~181, so prefill is ~4.3x cheaper
# while decode is unchanged (one word either way). Expect 5-7h against a 6.9h worst case
# that assumes no speedup at all. The 12h wall clock is ~1.7x over that worst case.
#
# SMOKE RUN FIRST:
#     CONFIG=src/config/zeroshot_smoke.yaml sbatch src/slurm_zeroshot.sh
# Twelve hours is a long time to discover the model tag was wrong.
#
# Resume after a timeout:  SKIP_EXISTING=1 sbatch src/slurm_zeroshot.sh
# One dataset:             DATASETS=data/datasets/d1_human_real_vs_human_fake.csv sbatch ...
# Other overrides:         CONFIG=... , VENV=... , REBUILD_VENV=1

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO"

CONFIG="${CONFIG:-src/config/zeroshot.yaml}"
OLLAMA_DIR="${OLLAMA_DIR:-$HOME/ollama}"        # model cache, persists across jobs
OLLAMA_SIF="${OLLAMA_SIF:-$HOME/ollama.sif}"

# A venv of its own. The guard below is `[ ! -d "$VENV" ]`, so pointing VENV at any
# existing directory SKIPS the install entirely and the job dies later on
# ModuleNotFoundError -- the same trap slurm_RAG.sh:34 and slurm_bert.sh document.
#
# Zero-shot's dependencies are a strict SUBSET of the RAG venv's, so
# VENV=$HOME/.venvs/fake_reviews_rag does work as a shortcut where that already exists.
# The default is separate anyway because building this one takes seconds: there is no
# vector store, hence no faiss-cpu, no sentence-transformers and no torch -- several GB of
# CUDA wheels that slurm_common.sh:99 has to install in its own resolver pass.
VENV="${VENV:-$HOME/.venvs/fake_reviews_zeroshot}"

# Per-job port. A fixed 11434 collides with any other Ollama on this node, and the
# readiness check would then pass against THAT server -- silently classifying with
# whatever model it happens to be serving.
PORT=$(( 11434 + (${SLURM_JOB_ID:-0} % 1000) ))

echo "======================================"
echo "Job ${SLURM_JOB_ID:-local} on $(hostname)  |  $(date)"
echo "config=$CONFIG  port=$PORT"
echo "======================================"
nvidia-smi || echo "WARNING: nvidia-smi unavailable"

OUT_DIR=$("$(command -v python3)" -c "import yaml;print(yaml.safe_load(open('$CONFIG'))['output_dir'])" 2>/dev/null || echo "results/zeroshot")
mkdir -p "$OLLAMA_DIR" "$REPO/logs" "$REPO/$OUT_DIR"

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

# Trap, not a trailing kill: on a crash a --time=12:00:00 reservation would otherwise hold
# the GPU with a live server for half a day.
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
  # Deliberately NOT `-r requirements.txt`: that pulls sentence-transformers, faiss-cpu and
  # torch for a job with no vector store, and requirements.txt currently carries a typo on
  # its sentence-transformers line that makes the whole file unresolvable.
  #
  # scikit-learn is here because src/evaluation.py:3 imports sklearn.metrics.
  # langchain-community and langchain-huggingface are NOT: zero-shot imports only
  # PromptTemplate (langchain-core) and OllamaLLM (langchain-ollama).
  "$VENV/bin/pip" install --quiet \
    pandas numpy scikit-learn PyYAML langchain-core langchain-ollama
fi
PYTHON="$VENV/bin/python"
echo "Python: $($PYTHON --version) at $PYTHON"

# Fail here in seconds rather than hours into the run, the way slurm_common.sh:120 does.
# slurm_RAG.sh has no equivalent, so a half-built venv there surfaces as a crash partway
# through the first dataset.
"$PYTHON" -c "import pandas, sklearn, yaml, langchain_core, langchain_ollama; print('imports ok')"

# -----------------------------
# PULL EVERY MODEL THE CONFIG NAMES
# -----------------------------
# Up front, not lazily: a missing tag discovered ten hours in wastes the whole run.
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

EXTRA_ARGS=""
[ -n "${SKIP_EXISTING:-}" ] && EXTRA_ARGS="--skip-existing"
for d in ${DATASETS:-}; do EXTRA_ARGS="$EXTRA_ARGS --dataset $d"; done

echo "======================================"
echo "Running zero-shot classification...${EXTRA_ARGS:+  args:$EXTRA_ARGS}"
echo "======================================"
# shellcheck disable=SC2086
"$PYTHON" src/zeroshot_classifier/zeroshot.py --config "$CONFIG" $EXTRA_ARGS

echo "======================================"
echo "Finished at $(date). Results:"
ls -1 "$REPO/$OUT_DIR/" | sed 's|^|  |'
echo "======================================"
