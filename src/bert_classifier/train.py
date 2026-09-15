"""BERT fine-tuning over the study datasets.

Runs every dataset listed in src/config/config.yaml by default:
    python3 -m src.bert_classifier.train
    python3 -m src.bert_classifier.train --dataset d1_human_real_vs_human_fake
    python3 -m src.bert_classifier.train --skip-existing     # resume after a timeout
On the cluster:
    sbatch src/slurm_bert.sh

Train and test come from each dataset's own `split` column; validation is carved out of
the train half. Hyperparameter combinations come from src/config/bert_grid.yaml, the
validation set selects the best run, and the test set is evaluated once for that run.

Results land in results/bert/<dataset>/ and are written as each dataset finishes. The
full grid is 24 runs per dataset, so a four-dataset run is 96 fine-tunings -- longer than
the wall clock currently set in slurm_bert.sh. --skip-existing makes a resubmit continue
from the datasets that already finished rather than starting over.

CHECKPOINT HYGIENE: only the best-scoring run's weights are kept, and superseded ones are
deleted as the grid proceeds. Saving all of them would leave roughly 420MB x 24 runs x 4
datasets, about 40GB, on disk for the sake of one checkpoint per dataset.
"""
import argparse
import inspect
import itertools
import shutil
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score, average_precision_score, precision_recall_fscore_support, roc_auc_score,
)
from transformers import (
    AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
    Trainer, TrainingArguments,
)

from src.data import prepare_dataset
from src.evaluation import results_dir, run_provenance, save_json


def tokenizer_kwarg(tokenizer):
    """Trainer's tokenizer argument, under whichever name this transformers has.

    It was `tokenizer=` until 4.46, when `processing_class=` superseded it, and the old
    name was removed outright in 4.57 -- passing it there is a TypeError. requirements.txt
    asks only for >=4.40, so both names are reachable and neither can be hardcoded.
    """
    if "processing_class" in inspect.signature(Trainer.__init__).parameters:
        return {"processing_class": tokenizer}
    return {"tokenizer": tokenizer}


def metrics_fn(eval_pred):
    logits, labels = eval_pred
    probs = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
    preds = (probs >= 0.5).astype(int)
    p, r, f, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": p, "recall": r, "f1": f,
        "roc_auc": roc_auc_score(labels, probs),
        "pr_auc": average_precision_score(labels, probs),
    }


def make_grid(cfg):
    keys = list(cfg["hyperparameters"])
    vals = [cfg["hyperparameters"][k] for k in keys]
    return [dict(zip(keys, v)) for v in itertools.product(*vals)]


def run_one(dataset, base, bcfg, tokenizer):
    train, val, test = prepare_dataset(base, dataset)
    provenance = run_provenance(dataset, train, val, test)
    out = results_dir(base, dataset, "bert")
    print(f"\n=== bert  {dataset}  "
          f"train {len(train)} / val {len(val)} / test {len(test)} ===")

    max_length = bcfg["model"]["max_length"]

    def tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    # Only `text` reaches the model. origin, source_dataset and cell_id_variation each
    # determine or narrow the label and must never become features.
    def encode(frame):
        return Dataset.from_pandas(
            frame[["text", "label"]], preserve_index=False
        ).map(tok, batched=True)

    dtrain, dval, dtest = encode(train), encode(val), encode(test)

    seed = base["split"]["random_seed"]
    combos = make_grid(bcfg)
    results = []
    best_dir, best_f1 = None, -1.0

    for i, hp in enumerate(combos, 1):
        run_dir = out / f"run_{i:03d}"
        model = AutoModelForSequenceClassification.from_pretrained(
            bcfg["model"]["pretrained_name"], num_labels=2
        )
        targs = TrainingArguments(
            output_dir=str(run_dir),
            learning_rate=float(hp["learning_rate"]),
            per_device_train_batch_size=int(hp["batch_size"]),
            per_device_eval_batch_size=int(hp["batch_size"]),
            num_train_epochs=float(hp["num_train_epochs"]),
            weight_decay=float(hp["weight_decay"]),
            warmup_ratio=float(bcfg["training"]["warmup_ratio"]),
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            logging_strategy="epoch",
            report_to="none",
            fp16=bool(bcfg["training"]["fp16"] and torch.cuda.is_available()),
            seed=seed,
        )
        trainer = Trainer(
            model=model, args=targs, train_dataset=dtrain, eval_dataset=dval,
            data_collator=DataCollatorWithPadding(tokenizer),
            compute_metrics=metrics_fn, **tokenizer_kwarg(tokenizer),
        )
        trainer.train()

        ev = trainer.evaluate(dval)
        clean = {k.replace("eval_", ""): float(v) for k, v in ev.items()
                 if isinstance(v, (float, int))}
        results.append({"run": i, "params": hp, **clean})
        f1 = clean.get("f1", -1.0)
        print(f"  {i}/{len(combos)}: val_f1={f1:.4f}")

        # save_strategy="epoch" leaves one checkpoint-*/ per epoch. load_best_model_at_end
        # means the in-memory model already IS the best epoch, so they add nothing.
        for ckpt in run_dir.glob("checkpoint-*"):
            shutil.rmtree(ckpt, ignore_errors=True)

        if f1 > best_f1:
            # save_model writes config.json + weights to the top of run_dir, which is what
            # the final test evaluation reloads. Without it the directory holds only the
            # checkpoints just deleted, and from_pretrained() raises OSError.
            trainer.save_model(str(run_dir))
            if best_dir is not None:
                shutil.rmtree(best_dir, ignore_errors=True)
            best_dir, best_f1 = run_dir, f1
        else:
            shutil.rmtree(run_dir, ignore_errors=True)

        del trainer, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    results.sort(key=lambda x: x["f1"], reverse=True)
    save_json({**provenance, "validation_results": results}, out / "validation_results.json")

    best = results[0]
    # Unlike the XGBoost and SVM scripts, this does NOT refit on train+val -- it reloads
    # the best run's checkpoint, which saw the train half only. Refitting would double the
    # GPU time; the asymmetry is deliberate and belongs in the write-up.
    best_trainer = Trainer(
        model=AutoModelForSequenceClassification.from_pretrained(str(best_dir)),
        args=TrainingArguments(output_dir=str(out / "final"), report_to="none"),
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=metrics_fn, **tokenizer_kwarg(tokenizer),
    )
    test_metrics = best_trainer.evaluate(dtest)
    save_json({**provenance,
               "best_validation_params": best["params"],
               "best_run_dir": best_dir.name,
               "test_metrics": test_metrics},
              out / "final_test_results.json")
    print(f"  -> test f1={test_metrics.get('eval_f1', float('nan')):.4f} "
          f"acc={test_metrics.get('eval_accuracy', float('nan')):.4f} "
          f"(majority baseline {provenance['majority_class_accuracy']:.4f})")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="src/config/config.yaml")
    ap.add_argument("--grid", default="src/config/bert_grid.yaml")
    ap.add_argument("--dataset", action="append",
                    help="dataset name (repeatable). Default: every entry in config.yaml")
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip datasets that already have a final_test_results.json")
    args = ap.parse_args()

    base = yaml.safe_load(open(args.config))
    bcfg = yaml.safe_load(open(args.grid))
    tokenizer = AutoTokenizer.from_pretrained(bcfg["model"]["pretrained_name"])

    for dataset in (args.dataset or base["data"]["datasets"]):
        done = (Path(base["experiment"]["output_dir"]) / "bert" / dataset
                / "final_test_results.json")
        if args.skip_existing and done.exists():
            print(f"\n=== bert  {dataset}  SKIPPED (already has {done.name}) ===")
            continue
        run_one(dataset, base, bcfg, tokenizer)


if __name__ == "__main__":
    main()
