"""TF-IDF + XGBoost over the study datasets.

Runs every dataset listed in src/config/config.yaml by default:
    python3 -m src.xgboost_classifier.train
    python3 -m src.xgboost_classifier.train --dataset d1_human_real_vs_human_fake

Train and test come from each dataset's own `split` column; validation is carved out of
the train half. Results land in results/<dataset>/xgboost/ and are written as each
dataset finishes, so an interrupted run keeps whatever already completed.
"""
import argparse
import itertools

import joblib
import numpy as np
import yaml
from scipy.sparse import vstack
from xgboost import XGBClassifier

from src.data import prepare_dataset
from src.evaluation import evaluate_binary, results_dir, run_provenance, save_json
from src.vectorizer import build_vectorizer


def grid(cfg):
    keys = list(cfg["hyperparameters"])
    vals = [cfg["hyperparameters"][k] for k in keys]
    return [dict(zip(keys, v)) for v in itertools.product(*vals)]


def run_one(dataset, base, grid_cfg):
    train, val, test = prepare_dataset(base, dataset)
    provenance = run_provenance(dataset, train, val, test)
    print(f"\n=== xgboost  {dataset}  "
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
        model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=seed,
            n_jobs=-1,
            tree_method="hist",
            **params,
        )
        model.fit(Xtr, train["label"])
        pv = model.predict_proba(Xv)[:, 1]
        metrics = evaluate_binary(val["label"], (pv >= 0.5).astype(int), pv)
        results.append({"run": i, "params": params, **metrics})
        print(f"  {i}/{len(combos)}: val_f1={metrics['f1']:.4f}")

    results.sort(key=lambda x: x["f1"], reverse=True)
    out = results_dir(base, dataset, "xgboost")
    save_json({**provenance, "validation_results": results}, out / "validation_results.json")

    best = results[0]["params"]
    final = XGBClassifier(
        objective="binary:logistic", eval_metric="logloss",
        random_state=seed, n_jobs=-1, tree_method="hist", **best,
    )
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
    ap.add_argument("--grid", default="src/config/xgboost_grid.yaml")
    ap.add_argument("--dataset", action="append",
                    help="dataset name (repeatable). Default: every entry in config.yaml")
    args = ap.parse_args()

    base = yaml.safe_load(open(args.config))
    grid_cfg = yaml.safe_load(open(args.grid))

    for dataset in (args.dataset or base["data"]["datasets"]):
        run_one(dataset, base, grid_cfg)


if __name__ == "__main__":
    main()
