# shellcheck shell=bash
#
# Shared body for the per-model Slurm scripts. NOT executable and NOT submittable on its
# own -- src/slurm_xgboost.sh, src/slurm_svm.sh and src/slurm_bert.sh each set a few
# variables and then source this.
#
# There is one script per model so each can be submitted and rerun independently, which
# leaves three near-identical files. Everything those files would otherwise duplicate --
# the venv build, the thread pinning, the trainer invocation, the results summary -- lives
# here so it cannot drift between them. The per-model scripts hold only their SBATCH
# resource request and their grid.
#
# Variables the caller MUST set before sourcing:
#   MODEL         xgboost | svm | bert   ->  runs src.<MODEL>_classifier.train
#                                            and reads results/<MODEL>/<dataset>/
#   GRID          path to that model's hyperparameter grid yaml
#   IMPORT_CHECK  space-separated modules to import before training starts
#
# Optional:
#   CONFIG        defaults to src/config/config.yaml
#   VENV          defaults to $HOME/.venvs/fake_reviews_bert
#   EXTRA_ARGS    extra flags for the trainer (BERT passes --skip-existing through this)
#   NEEDS_GPU     non-empty to run nvidia-smi and report torch's CUDA visibility
#
# Switches honoured for every model:
#   SETUP_ONLY=1     build the venv and stop, without training
#   REBUILD_VENV=1   delete and rebuild the venv first

: "${MODEL:?slurm_common.sh: MODEL must be set before sourcing}"
: "${GRID:?slurm_common.sh: GRID must be set before sourcing}"
: "${IMPORT_CHECK:?slurm_common.sh: IMPORT_CHECK must be set before sourcing}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

# -----------------------------
# CONFIG
# -----------------------------
REPO="${SLURM_SUBMIT_DIR:-$PWD}"

# The trainer is addressed as `-m src.<model>_classifier.train`, which needs the repo root
# on sys.path, and dataset_dir inside the config is relative to that root too. Everything
# below therefore runs from here.
cd "$REPO"

CONFIG="${CONFIG:-src/config/config.yaml}"

# ONE venv for all three models, so they never end up on different sklearn or numpy
# versions and produce results that cannot be compared. The name says bert for historical
# reasons and is kept deliberately: this path already exists on the cluster with every
# dependency installed, and renaming it would force a multi-GB rebuild for no gain.
VENV="${VENV:-$HOME/.venvs/fake_reviews_bert}"

# Size the numeric libraries to the ALLOCATION, not the node. XGBoost's n_jobs=-1 and
# sklearn's BLAS calls both read the machine's total core count, which on a shared node is
# far more than this job was granted -- they oversubscribe the cgroup and run slower than
# if correctly pinned.
THREADS="${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_PER_GPU:-8}}"
export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"

RESULTS="$REPO/results"

echo "======================================"
echo "Job ${SLURM_JOB_ID:-local} on $(hostname)  |  $(date)"
echo "model=$MODEL  config=$CONFIG  grid=$GRID"
echo "venv=$VENV  threads=$THREADS"
echo "output=$RESULTS/$MODEL/<dataset>/"
echo "======================================"

if [ -n "${NEEDS_GPU:-}" ]; then
  nvidia-smi || echo "WARNING: nvidia-smi unavailable"
fi

mkdir -p "$REPO/logs" "$RESULTS"
[ -n "${HF_HOME:-}" ] && mkdir -p "$HF_HOME"

# -----------------------------
# PYTHON ENV
# -----------------------------
# Never install the requirements interactively on a login node: torch drags in the bundled
# CUDA runtime (nvidia-cublas, nvidia-cudnn, ...), several GB of wheels to download, unpack
# and byte-compile, and the resolver holds much of it in RAM. On a shared login node that
# OOMs the cgroup and takes code-server down with it. Build it inside a job, here.
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
  # Installed even for the CPU-only models: one shared venv is the point, and a second
  # lighter one would put them on different dependency versions.
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

# Import everything this job needs NOW, so a half-built venv fails here in seconds rather
# than partway into a grid.
"$PYTHON" - "$IMPORT_CHECK" <<'IMPORTCHECK'
import importlib, sys
for name in sys.argv[1].split():
    module = importlib.import_module(name)
    print(f"  {name:14s} {getattr(module, '__version__', 'n/a')}")
IMPORTCHECK

if [ -n "${NEEDS_GPU:-}" ]; then
  "$PYTHON" - <<'CUDACHECK'
import torch
available = torch.cuda.is_available()
print(f"  cuda available: {available}"
      + (f"  device: {torch.cuda.get_device_name(0)}" if available else ""))
CUDACHECK
fi

if [ -n "${SETUP_ONLY:-}" ]; then
  echo "======================================"
  echo "SETUP_ONLY set; venv ready at $VENV, skipping the grid."
  echo "Next:  sbatch src/slurm_${MODEL}.sh"
  echo "======================================"
  exit 0
fi

# -----------------------------
# RUN THE GRID
# -----------------------------
# The dataset CSVs under data/ are read-only inputs; the job only reads them. The trainer
# loops every dataset listed in $CONFIG and writes each one's results as it finishes, so an
# interrupted job keeps whatever already completed.
echo "Running $MODEL grid over every dataset in $CONFIG..."

# EXTRA_ARGS is deliberately unquoted: it carries zero or more separate flags.
# shellcheck disable=SC2086
"$PYTHON" -m "src.${MODEL}_classifier.train" \
  --config "$CONFIG" \
  --grid "$GRID" \
  $EXTRA_ARGS

# -----------------------------
# SUMMARY
# -----------------------------
# One line per dataset. Each writes validation_results.json before its final test
# evaluation, so a dataset that died in between leaves the first file and not the second;
# say which happened rather than making it a directory listing after the fact.
echo "======================================"
"$PYTHON" - "$MODEL" <<'PYSUM'
import json, pathlib, sys, yaml

model = sys.argv[1]
base = yaml.safe_load(open("src/config/config.yaml"))
root = pathlib.Path(base["experiment"]["output_dir"])
done = missing = 0

for name in base["data"]["datasets"]:
    final = root / model / name / "final_test_results.json"
    if not final.exists():
        missing += 1
        partial = final.with_name("validation_results.json")
        note = "  (validation_results.json exists -- died during the final test eval)" if partial.exists() else ""
        print(f"  MISSING  {name}{note}")
        continue
    done += 1
    d = json.load(open(final))
    m = d["test_metrics"]
    # XGBoost and SVM write bare metric names; the HF Trainer prefixes them with eval_.
    got = []
    for k in ("accuracy", "f1", "roc_auc", "pr_auc"):
        v = m.get(k, m.get("eval_" + k))
        if v is not None:
            got.append(f"{k}={v:.4f}")
    print(f"  OK       {name}  " + "  ".join(got))
    print(f"           majority baseline={d['majority_class_accuracy']:.4f}  "
          f"params={d['best_validation_params']}")

print(f"\n{done} of {done + missing} datasets complete for {model}.")
if missing and model == "bert":
    print("Resume the rest with:  SKIP_EXISTING=1 sbatch src/slurm_bert.sh")
elif missing:
    print(f"Rerun with:  sbatch src/slurm_{model}.sh")
PYSUM
echo "Job finished at $(date)"
echo "======================================"
