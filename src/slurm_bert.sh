#!/bin/bash
#SBATCH --job-name=fr_bert
#SBATCH --partition=gpu
#SBATCH --nodelist=ant2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/bert_%j.out
#SBATCH --error=logs/bert_%j.err

# BERT fine-tuning over every dataset in src/config/config.yaml.
#
# Submit from the repo root:   sbatch src/slurm_bert.sh
# Smaller grid:                GRID=src/config/bert_smoke.yaml sbatch src/slurm_bert.sh
# Resume after an interrupt:   SKIP_EXISTING=1 sbatch src/slurm_bert.sh
# Build the venv and stop:     SETUP_ONLY=1 sbatch src/slurm_bert.sh
# Force a clean rebuild:       REBUILD_VENV=1 sbatch src/slurm_bert.sh
#
# WALL CLOCK: 24 combinations x 8 datasets = 192 fine-tunings. Job 1675581 ran the full
# 24-combination grid on one 1014-row dataset in 8 minutes on this node's H100, with
# individual fine-tunings taking 8-46 seconds. The eight datasets total 10563 training
# rows, 10.4x that one, which projects to roughly 1h30. The 4h below is left generous
# rather than trimmed: it costs nothing in a backfill scheduler and absorbs a shared GPU.
#
# The trainer writes each dataset's results as that dataset finishes, so an interrupted job
# keeps whatever already completed and SKIP_EXISTING=1 resumes from there.

set -euo pipefail

MODEL=bert
GRID="${GRID:-src/config/bert_grid.yaml}"
IMPORT_CHECK="torch transformers datasets sklearn yaml pandas"
NEEDS_GPU=1

# Persist the model cache so bert-base-uncased is downloaded once, not once per job.
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

# Skip datasets that already have a final_test_results.json.
EXTRA_ARGS=""
[ -n "${SKIP_EXISTING:-}" ] && EXTRA_ARGS="--skip-existing"

source "${SLURM_SUBMIT_DIR:-$PWD}/src/slurm_common.sh"
