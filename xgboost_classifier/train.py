import argparse, itertools, json, sys
from pathlib import Path
import joblib, yaml, numpy as np
from xgboost import XGBClassifier
from common.data import load_data, make_split
from common.tfidf import build_vectorizer
from common.evaluation import evaluate_binary, save_json

def grid(cfg):
    keys = list(cfg["hyperparameters"])
    vals = [cfg["hyperparameters"][k] for k in keys]
    return [dict(zip(keys, v)) for v in itertools.product(*vals)]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--grid", default="config/xgboost_grid.yaml")
    args = ap.parse_args()

    base = yaml.safe_load(open(args.config))
    grid_cfg = yaml.safe_load(open(args.grid))
    df = load_data(base["data"]["path"], base["data"]["text_column"], base["data"]["label_column"])
    train, val, test = make_split(df, **{
        "seed": base["split"]["random_seed"],
        "train_size": base["split"]["train_size"],
        "validation_size": base["split"]["validation_size"],
    })

    vec = build_vectorizer(base["features"])
    Xtr = vec.fit_transform(train["text"])
    Xv = vec.transform(val["text"])
    Xt = vec.transform(test["text"])

    results = []
    for i, params in enumerate(grid(grid_cfg), 1):
        model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=base["split"]["random_seed"],
            n_jobs=-1,
            tree_method="hist",
            **params,
        )
        model.fit(Xtr, train["label"])
        pv = model.predict_proba(Xv)[:, 1]
        yv = (pv >= 0.5).astype(int)
        metrics = evaluate_binary(val["label"], yv, pv)
        results.append({"run": i, "params": params, **metrics})
        print(f"{i}/{len(grid(grid_cfg))}: val_f1={metrics['f1']:.4f}")

    results = sorted(results, key=lambda x: x["f1"], reverse=True)
    out = Path(base["experiment"]["output_dir"]) / "xgboost"
    out.mkdir(parents=True, exist_ok=True)
    save_json({"validation_results": results}, out / "validation_results.json")

    best = results[0]["params"]
    final = XGBClassifier(
        objective="binary:logistic", eval_metric="logloss",
        random_state=base["split"]["random_seed"], n_jobs=-1,
        tree_method="hist", **best
    )
    X_trainval = __import__("scipy").sparse.vstack([Xtr, Xv])
    y_trainval = np.concatenate([train["label"].values, val["label"].values])
    final.fit(X_trainval, y_trainval)
    pt = final.predict_proba(Xt)[:, 1]
    yt = (pt >= 0.5).astype(int)
    test_metrics = evaluate_binary(test["label"], yt, pt)
    save_json({"best_validation_params": best, "test_metrics": test_metrics},
              out / "final_test_results.json")
    joblib.dump({"vectorizer": vec, "model": final}, out / "model.joblib")

if __name__ == "__main__":
    main()
