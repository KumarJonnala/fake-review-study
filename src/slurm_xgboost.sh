#!/bin/bash
#SBATCH --job-name=fr_xgboost
#SBATCH --partition=gpu
#SBATCH --nodelist=ant2
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=logs/xgboost_%j.out
#SBATCH --error=logs/xgboost_%j.err

# NO --gres HERE, ON PURPOSE. TF-IDF + XGBoost is CPU-only work; requesting a GPU would
# hold an H100 idle for the whole job. `gpu` is the only partition available on this
# cluster, so the job still lands on a GPU node -- it just does not reserve the device.
# If the partition turns out to REJECT jobs that request no gres, uncomment this line:
##SBATCH --gres=gpu:1

# TF-IDF + XGBoost over every dataset in src/config/config.yaml.
#
# Submit from the repo root:   sbatch src/slurm_xgboost.sh
# Smaller grid:                GRID=<path> sbatch src/slurm_xgboost.sh
# Build the venv and stop:     SETUP_ONLY=1 sbatch src/slurm_xgboost.sh
# Force a clean rebuild:       REBUILD_VENV=1 sbatch src/slurm_xgboost.sh
#
# WALL CLOCK: the grid is 36 combinations x 8 datasets = 288 fits. Worst-case combinations
# (800 trees, depth 7) were measured at 33s on the 1014-train datasets, 52s on
# d3_unsampled and 75s on the largest, d3.5_unsampled. If EVERY combination were the worst
# case the job runs about 3h15; the grid's mean is far below that, since n_estimators and
# max_depth both vary. 6h leaves roughly 2x headroom over the pessimistic figure.

set -euo pipefail

MODEL=xgboost
GRID="${GRID:-src/config/xgboost_grid.yaml}"
IMPORT_CHECK="sklearn xgboost yaml pandas scipy joblib"

source "${SLURM_SUBMIT_DIR:-$PWD}/src/slurm_common.sh"
