#!/bin/bash
#SBATCH --job-name=fr_svm
#SBATCH --partition=gpu-stud
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/svm_%j.out
#SBATCH --error=logs/svm_%j.err

# NO --gres HERE, ON PURPOSE. TF-IDF + SVM is CPU-only work; requesting a GPU would hold
# an H100 idle for the whole job. `gpu` is the only partition available on this cluster,
# so the job still lands on a GPU node -- it just does not reserve the device.
# If the partition turns out to REJECT jobs that request no gres, uncomment this line:
##SBATCH --gres=gpu:1

# TF-IDF + SVM over every dataset in src/config/config.yaml.
#
# Submit from the repo root:   sbatch src/slurm_svm.sh
# Smaller grid:                GRID=<path> sbatch src/slurm_svm.sh
# Build the venv and stop:     SETUP_ONLY=1 sbatch src/slurm_svm.sh
# Force a clean rebuild:       REBUILD_VENV=1 sbatch src/slurm_svm.sh
#
# WALL CLOCK: the grid is 24 combinations x 8 datasets = 192 fits. A single rbf fit with
# probability=True was measured at 8s on the 1014-train datasets, 16s on d3_unsampled and
# d2.5_unsampled, and 31s on the largest, d3.5_unsampled -- roughly 45 minutes if every
# combination took the worst-case kernel. 2h covers that with room to spare.
#
# This is the cheapest of the three jobs, so it is the one to submit first when validating
# a change to the shared body in src/slurm_common.sh.

set -euo pipefail

MODEL=svm
GRID="${GRID:-src/config/svm_grid.yaml}"
IMPORT_CHECK="sklearn yaml pandas scipy joblib"

source "${SLURM_SUBMIT_DIR:-$PWD}/src/slurm_common.sh"
