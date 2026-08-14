#!/bin/bash
#SBATCH --job-name=fake_review_llm
#SBATCH --partition=gpu
#SBATCH --nodelist=ant2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=logs/job_%j.out
#SBATCH --error=logs/job_%j.err

# Submit from the repo root:   sbatch src/slurm_synthetic_data.sh
# Override the model/volume:   MODEL=llama3.1:8b N_PER_CELL=10 sbatch src/slurm_synthetic_data.sh

set -euo pipefail

# -----------------------------
# CONFIG
# -----------------------------
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
OLLAMA_DIR="${OLLAMA_DIR:-$HOME/ollama}"          # model cache, persists across jobs
OLLAMA_SIF="${OLLAMA_SIF:-$HOME/ollama.sif}"
VENV="${VENV:-$HOME/.venvs/fake_reviews}"

MODEL="${MODEL:-qwen2.5:32b}"
N_PER_CELL="${N_PER_CELL:-10}"
SEED="${SEED:-12}"

# Per-job port. A fixed 11434 collides with any other Ollama on this node, and the
# readiness check would then pass against THAT server -- silently generating with
# whatever model it happens to be serving.
PORT=$(( 11434 + (${SLURM_JOB_ID:-0} % 1000) ))

# Model-tagged output, so a qwen run never lands in (or --resumes into) a llama file.
MODEL_TAG=$(echo "$MODEL" | tr ':/' '__')
OUTPUT="$REPO/data/generated/Hotel_LLM_Reviews_${MODEL_TAG}.csv"

echo "======================================"
echo "Job ${SLURM_JOB_ID:-local} on $(hostname)  |  $(date)"
echo "model=$MODEL  n_per_cell=$N_PER_CELL  seed=$SEED  port=$PORT"
echo "output=$OUTPUT"
echo "======================================"
nvidia-smi || echo "WARNING: nvidia-smi unavailable"

mkdir -p "$OLLAMA_DIR" "$REPO/logs" "$REPO/data/generated"

if [ ! -f "$OLLAMA_SIF" ]; then
  echo "ERROR: Apptainer image not found at $OLLAMA_SIF"
  echo "Build it once on a login node:  apptainer pull \"$OLLAMA_SIF\" docker://ollama/ollama:latest"
  exit 1
fi

# -----------------------------
# START OLLAMA SERVER
# -----------------------------
echo "Starting Ollama server..."

apptainer exec --nv \
  --env OLLAMA_HOST=0.0.0.0:$PORT \
  --bind "$OLLAMA_DIR":/root/.ollama \
  "$OLLAMA_SIF" \
  ollama serve &

OLLAMA_PID=$!

# Trap, not a trailing kill: on a crash, a --time=08:00:00 reservation would otherwise
# hold the GPU with a live server for the full allocation.
cleanup() {
  echo "Stopping Ollama (pid $OLLAMA_PID)..."
  kill "$OLLAMA_PID" 2>/dev/null || true
  wait "$OLLAMA_PID" 2>/dev/null || true
}
trap cleanup EXIT

# -----------------------------
# WAIT FOR OLLAMA TO BE READY
# -----------------------------
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
# PULL MODEL (cached in $OLLAMA_DIR after the first job)
# -----------------------------
echo "Ensuring $MODEL is available..."

apptainer exec --nv \
  --env OLLAMA_HOST=0.0.0.0:$PORT \
  --bind "$OLLAMA_DIR":/root/.ollama \
  "$OLLAMA_SIF" \
  ollama pull "$MODEL"

# Confirm it actually landed -- `ollama pull` can report progress and still fail on a
# full quota, and the generator would then 404 on every single call.
curl -sf "http://localhost:$PORT/api/tags" | grep -q "$MODEL" || {
  echo "ERROR: $MODEL not present after pull"
  exit 1
}

# -----------------------------
# PYTHON ENV
# -----------------------------
# A venv built once and reused, rather than `pip install --user` on every job:
# concurrent jobs writing the same ~/.local tree race against each other.
if [ ! -d "$VENV" ]; then
  echo "Creating venv at $VENV..."
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install --quiet --upgrade pip
  "$VENV/bin/pip" install --quiet pandas requests tqdm
fi
PYTHON="$VENV/bin/python"
echo "Python: $($PYTHON --version) at $PYTHON"

# -----------------------------
# RUN GENERATION
# -----------------------------
echo "Running generation pipeline..."

export OLLAMA_HOST="localhost:$PORT"

# The two corpus CSVs under data/ are read-only inputs; the generator only reads them
# for few-shot examples and the 20 hotel names.
"$PYTHON" "$REPO/src/generate_synthetic_reviews.py" \
  --model "$MODEL" \
  --n-per-cell "$N_PER_CELL" \
  --seed "$SEED" \
  --host "localhost:$PORT" \
  --output "$OUTPUT" \
  --resume

echo "======================================"
echo "Rows written: $(( $(wc -l < "$OUTPUT") - 1 ))"
FAILURES="${OUTPUT%.csv}_failures.jsonl"
[ -s "$FAILURES" ] && echo "Failures: $(wc -l < "$FAILURES")  -> $FAILURES"
echo "Job finished at $(date)"
echo "======================================"
