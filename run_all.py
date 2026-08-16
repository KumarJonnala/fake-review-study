import subprocess
import sys

commands = [
    #[sys.executable, "-m", "xgboost_classifier.train"],
    #[sys.executable, "-m", "svm_classifier.train"],
    [sys.executable, "-m", "bert_classifier.train"],
]

for cmd in commands:
    print("\n>>>", " ".join(cmd))
    subprocess.run(cmd, check=True)