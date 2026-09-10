"""Train all three model families across every dataset in src/config/config.yaml.

    python3 src/run_all.py

Each training module loops the four study datasets itself, so this is just the three
families in sequence. Run BERT on the cluster instead (sbatch src/slurm_bert.sh) if a GPU
is not available locally -- 24 combinations x 4 datasets is a long CPU job.
"""
import subprocess
import sys
from pathlib import Path

# The training modules are addressed as src.* and their --config/--grid defaults are
# relative to the repo root, so pin the working directory here rather than relying on
# where this script happens to be invoked from.
REPO = Path(__file__).resolve().parent.parent

commands = [
    [sys.executable, "-m", "src.xgboost_classifier.train"],
    [sys.executable, "-m", "src.svm_classifier.train"],
    [sys.executable, "-m", "src.bert_classifier.train"],
]

for cmd in commands:
    print("\n>>>", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=REPO)
