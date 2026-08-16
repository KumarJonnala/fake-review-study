import subprocess
import sys
from pathlib import Path

# The training modules are addressed as src.* and their --config/--grid defaults are
# relative to the repo root, so pin the working directory here rather than relying on
# where this script happens to be invoked from.
REPO = Path(__file__).resolve().parent.parent

commands = [
    #[sys.executable, "-m", "src.xgboost_classifier.train"],
    #[sys.executable, "-m", "src.svm_classifier.train"],
    [sys.executable, "-m", "src.bert_classifier.train"],
]

for cmd in commands:
    print("\n>>>", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=REPO)
