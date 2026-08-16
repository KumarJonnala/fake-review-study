"""
BERT fine-tuning script.

For paper-quality experiments, run this separately from the TF-IDF models so
GPU memory and runtime are easy to control. Hyperparameter combinations are
read from src/config/bert_grid.yaml. The validation set selects the best run;
the test set is evaluated only for that selected configuration.

Run from the repo root:  python -m src.bert_classifier.train
On the cluster:          sbatch src/slurm_bert.sh
"""
import argparse, itertools, json, shutil
from pathlib import Path
import yaml, torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, average_precision_score

from src.data import load_data, make_split

def metrics_fn(eval_pred):
    logits, labels = eval_pred
    probs = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
    preds = (probs >= 0.5).astype(int)
    p, r, f, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="src/config/config.yaml")
    ap.add_argument("--grid", default="src/config/bert_grid.yaml")
    args = ap.parse_args()
    base = yaml.safe_load(open(args.config))
    bcfg = yaml.safe_load(open(args.grid))
    # Shared loader, same as the SVM and XGBoost scripts: adds the NaN guard and the
    # whitespace strip this path used to skip, and keeps the label mapping in one place.
    df = load_data(base["data"]["path"], base["data"]["text_column"],
                   base["data"]["label_column"])

    train, val, test = make_split(df, seed=base["split"]["random_seed"],
                                  train_size=base["split"]["train_size"],
                                  validation_size=base["split"]["validation_size"])

    tokenizer = AutoTokenizer.from_pretrained(bcfg["model"]["pretrained_name"])
    max_length = bcfg["model"]["max_length"]

    def tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    dtrain = Dataset.from_pandas(train[["text","label"]], preserve_index=False).map(tok, batched=True)
    dval = Dataset.from_pandas(val[["text","label"]], preserve_index=False).map(tok, batched=True)
    dtest = Dataset.from_pandas(test[["text","label"]], preserve_index=False).map(tok, batched=True)

    results = []
    out = Path(base["experiment"]["output_dir"]) / "bert"
    out.mkdir(parents=True, exist_ok=True)

    for i, hp in enumerate(make_grid(bcfg), 1):
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
            seed=base["split"]["random_seed"],
        )
        trainer = Trainer(
            model=model, args=targs, train_dataset=dtrain, eval_dataset=dval,
            tokenizer=tokenizer, data_collator=DataCollatorWithPadding(tokenizer),
            compute_metrics=metrics_fn
        )
        trainer.train()

        # load_best_model_at_end=True means the in-memory model is already the best epoch,
        # so this writes config.json + weights to the top of run_dir -- which is what the
        # final test evaluation below reloads. Without it that directory holds only
        # checkpoint-*/ subdirs and from_pretrained() raises OSError after the whole grid.
        trainer.save_model(str(run_dir))
        for ckpt in run_dir.glob("checkpoint-*"):
            shutil.rmtree(ckpt, ignore_errors=True)

        ev = trainer.evaluate(dval)
        clean = {k.replace("eval_", ""): float(v) for k, v in ev.items()
                 if isinstance(v, (float, int))}
        results.append({"run": i, "params": hp, **clean})
        print(f"{i}/{len(make_grid(bcfg))}: val_f1={clean.get('f1', float('nan')):.4f}")

    results.sort(key=lambda x: x["f1"], reverse=True)
    with open(out / "validation_results.json", "w") as f:
        json.dump({"validation_results": results}, f, indent=2)

    best = results[0]
    # The best validation checkpoint is retained in its run directory.
    best_trainer = Trainer(
        model=AutoModelForSequenceClassification.from_pretrained(
            str(out / f"run_{best['run']:03d}")
        ),
        args=TrainingArguments(output_dir=str(out / "final"), report_to="none"),
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=metrics_fn
    )
    test_metrics = best_trainer.evaluate(dtest)
    with open(out / "final_test_results.json", "w") as f:
        json.dump({"best_validation_params": best["params"],
                   "test_metrics": test_metrics}, f, indent=2)

if __name__ == "__main__":
    main()
