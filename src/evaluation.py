from pathlib import Path
import json
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)

def evaluate_binary(y_true, y_pred, scores=None):
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    if scores is not None:
        out["roc_auc"] = float(roc_auc_score(y_true, scores))
        out["pr_auc"] = float(average_precision_score(y_true, scores))
    return out

def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def run_provenance(dataset, train, val, test):
    """Sizes and the majority-class accuracy floor, recorded alongside every result.

    d3_unsampled is 33% real / 67% fake, so its accuracy floor is 0.667 rather than 0.5,
    and its accuracy is NOT comparable to the three balanced datasets. Writing the floor
    into the results file itself means whoever reads it sees that without having to go
    and look the composition up.
    """
    fake_fraction = float(test["label"].mean())
    return {
        "dataset": dataset,
        "n_train": len(train),
        "n_validation": len(val),
        "n_test": len(test),
        "test_fake_fraction": fake_fraction,
        "majority_class_accuracy": max(fake_fraction, 1.0 - fake_fraction),
    }


def results_dir(base, dataset, model_name):
    """results/<dataset>/<model>/ -- one directory per (dataset, model) pair."""
    path = Path(base["experiment"]["output_dir"]) / dataset / model_name
    path.mkdir(parents=True, exist_ok=True)
    return path
