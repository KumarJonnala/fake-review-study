"""TF-IDF + SVM over the study datasets.

Runs every dataset listed in src/config/config.yaml by default:
    python3 -m src.svm_classifier.train
    python3 -m src.svm_classifier.train --dataset d1_human_real_vs_human_fake

Train and test come from each dataset's own `split` column; validation is carved out of
the train half. Results land in results/svm/<dataset>/ and are written as each dataset
finishes, so an interrupted run keeps whatever already completed.
"""
import argparse
import itertools

import joblib
import numpy as np
import yaml
from scipy.sparse import vstack
from sklearn.svm import SVC

from src.data import prepare_dataset
from src.evaluation import evaluate_binary, results_dir, run_provenance, save_json
from src.vectorizer import build_vectorizer


def grid(cfg):
    """Hyperparameter combinations, with the linear-kernel duplicates removed.

    `gamma` is only read for the rbf kernel, so the full product repeats every linear
    combination once per gamma value -- 12 of 36 combos were identical refits scored
    identically. Pinning gamma for linear entries and de-duplicating cuts the grid to 24
    without changing which configurations are searched.
    """
    keys = list(cfg["hyperparameters"])
    vals = [cfg["hyperparameters"][k] for k in keys]

    seen, combos = set(), []
    for values in itertools.product(*vals):
        params = dict(zip(keys, values))
        if params.get("kernel") == "linear" and "gamma" in params:
            params["gamma"] = "scale"
        fingerprint = tuple(sorted((k, str(v)) for k, v in params.items()))
        if fingerprint not in seen:
            seen.add(fingerprint)
            combos.append(params)
    return combos


def run_one(dataset, base, grid_cfg):
    train, val, test = prepare_dataset(base, dataset)
    provenance = run_provenance(dataset, train, val, test)
    print(f"\n=== svm  {dataset}  "
          f"train {len(train)} / val {len(val)} / test {len(test)} ===")

    # Only `text` is ever vectorized. origin, source_dataset and cell_id_variation each
    # determine or narrow the label and must never become features.
    vec = build_vectorizer(base["features"])
    Xtr = vec.fit_transform(train["text"])
    Xv = vec.transform(val["text"])
    Xt = vec.transform(test["text"])

    seed = base["split"]["random_seed"]
    combos = grid(grid_cfg)
    results = []
    for i, params in enumerate(combos, 1):
        model = SVC(probability=True, random_state=seed, **params)
        model.fit(Xtr, train["label"])
        pv = model.predict_proba(Xv)[:, 1]
        metrics = evaluate_binary(val["label"], (pv >= 0.5).astype(int), pv)
        results.append({"run": i, "params": params, **metrics})
        print(f"  {i}/{len(combos)}: val_f1={metrics['f1']:.4f}")

    results.sort(key=lambda x: x["f1"], reverse=True)
    out = results_dir(base, dataset, "svm")
    save_json({**provenance, "validation_results": results}, out / "validation_results.json")

    best = results[0]["params"]
    final = SVC(probability=True, random_state=seed, **best)
    final.fit(vstack([Xtr, Xv]),
              np.concatenate([train["label"].values, val["label"].values]))
    pt = final.predict_proba(Xt)[:, 1]
    test_metrics = evaluate_binary(test["label"], (pt >= 0.5).astype(int), pt)
    save_json({**provenance, "best_validation_params": best, "test_metrics": test_metrics},
              out / "final_test_results.json")
    if base["experiment"].get("save_models", True):
        joblib.dump({"vectorizer": vec, "model": final}, out / "model.joblib")
    print(f"  -> test f1={test_metrics['f1']:.4f} "
          f"acc={test_metrics['accuracy']:.4f} "
          f"(majority baseline {provenance['majority_class_accuracy']:.4f})")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="src/config/config.yaml")
    ap.add_argument("--grid", default="src/config/svm_grid.yaml")
    ap.add_argument("--dataset", action="append",
                    help="dataset name (repeatable). Default: every entry in config.yaml")
    args = ap.parse_args()

    base = yaml.safe_load(open(args.config))
    grid_cfg = yaml.safe_load(open(args.grid))

    for dataset in (args.dataset or base["data"]["datasets"]):
        run_one(dataset, base, grid_cfg)


if __name__ == "__main__":
    main()
