"""
Multi-Dataset x Multi-Model ZERO-SHOT Evaluation for Fake Review Detection
--------------------------------------------------------------------------
Outer loop  : CSV datasets from data/datasets/
Inner loop  : Ollama LLM models
Output      : results/zeroshot/<dataset>.json + summary.json

    python3 src/zeroshot_classifier/zeroshot.py [--config src/config/zeroshot.yaml]

THE CONTROL CONDITION for src/rag_classifier/RAG.py. Same prompt, same judges, same
temperature, same test rows -- no retrieved examples. The difference between the two result
sets is therefore attributable to retrieval and to nothing else.

That comparison is the point. RAG's retrieval-only baseline already scores 1.000 on every
synthetic origin of d3.5_unsampled and 0.312 on human_real, which says the neighbours were
matching synthetic-to-synthetic rather than detecting fakeness. So when an LLM judge scores
well under RAG, it is unclear how much is the model's judgment and how much is the
retrieved neighbours handing it the answer. Subtracting this run from that one separates
them.

What differs from RAG.py:

* NO VECTOR STORE, so no faiss and no sentence-transformers. The train half is still
  loaded -- load_split is what refuses a dataset lacking a `split` column, which is the
  guarantee that these are the same test rows the SVM, XGBoost, BERT and RAG runs scored --
  but it is never shown to a model and never read for anything but its row count.
* NO RETRIEVAL-ONLY BASELINE is possible, so the anchor is the MAJORITY-CLASS FLOOR from
  run_provenance(): 0.500 on the five sampled datasets, but 0.667 / 0.707 / 0.773 on
  d3_unsampled / d2.5_unsampled / d3.5_unsampled. An 0.77 accuracy on d3.5_unsampled is the
  constant predictor. The summary table also prints the matching RAG accuracy wherever
  results/rag/<dataset>.json exists, since that is the comparison this run exists to make.
* --dataset and --skip-existing. RAG had neither, so scoping a job meant editing rag.yaml,
  which is why that file lists 3 datasets while results/rag/ holds 8 and its summary.json
  describes 1. summary.json here is rebuilt from the files on disk, not from this job's
  in-memory dict.
* Per-origin accuracy is kept, because the four judges also WROTE the fakes in d2/d3.

Metrics, UNKNOWN handling and the split loader come from src/llm_common.py, shared with
RAG.py so both runs are scored by the same code and stay subtractable.

Only `text` is ever shown to a model. `origin`, `source_dataset` and `cell_id_variation`
all leak the label; `origin` is kept for the test rows only, to break results down after
the fact.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import yaml

from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
from src.evaluation import run_provenance, save_json  # noqa: E402
from src.llm_common import (  # noqa: E402
    accuracy_by_origin,
    load_split,
    normalize_prediction,
    score,
)

# ─────────────────────────────────────────────
# PROMPT
# ─────────────────────────────────────────────

# VERBATIM from rag_prompt in src/rag_classifier/RAG.py, minus the
# "Here are some examples:\n    {examples}" block and one of its delimiting blank lines.
#
# THE AWKWARDNESS IS DELIBERATE AND MUST NOT BE TIDIED. "Carefully analyze the following
# review for signs such as:" appears twice, "review of a Hotel review" is ungrammatical,
# and the line after the first sentence holds nothing but four spaces. results/rag/ was
# produced with those defects, so cleaning them here would make the zero-shot/RAG gap a
# prompt-wording difference instead of the presence or absence of retrieved examples --
# which is the only thing this run is meant to measure.
#
# The honest framing for a write-up: BOTH arms use a defective prompt, so this measures
# retrieval's value GIVEN THIS PROMPT, not zero-shot's ceiling. A cleaned prompt belongs in
# a third arm, not as a replacement for this one.
zeroshot_prompt = PromptTemplate(
    input_variables=["review"],
    template="""
    You are an expert at detecting fake and real reviews of hotel. Carefully analyze the following review for signs such as:
    
    Task: Classify the following review of a Hotel review as either "real" or "fake". Carefully analyze the following review for signs such as:
    - Overly generic or vague language
    - Exaggerated praise or criticism
    - Repetitive or templated phrasing
    - Marketing-like wording or unnatural flow
    - Specific details vs. general statements

    Now classify this review:
    "{review}"

    Classify this review as either "real" or "fake" (respond with only one word):

"""
)

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────


def classify_review(review_text: str, llm) -> str:
    """One call. The retriever argument is the only thing RAG's version has that this does
    not -- kept structurally identical so the two call paths stay comparable."""
    prompt = zeroshot_prompt.format(review=review_text)
    try:
        response = llm.invoke(prompt)
    except Exception as e:
        # One dead call must not abort a twelve-hour run. Counted as UNKNOWN and reported.
        print(f"    [error] {type(e).__name__}: {e}")
        return "UNKNOWN"
    return normalize_prediction(response)


def build_llm(model_name: str, cfg: dict) -> OllamaLLM:
    """OllamaLLM with the request timeout actually wired up.

    rag.yaml:47 declares `request_timeout: 300` and RAG.py:277 never reads it -- OllamaLLM
    is constructed with model and temperature only. Over a 12h run one hung call would
    stall every dataset queued behind it, so it is passed through here.

    VIA client_kwargs, NOT `timeout=`. OllamaLLM is a pydantic model with extra="allow" on
    langchain-ollama 1.0.1: `OllamaLLM(model=..., timeout=300)` is accepted without error
    and then ignored, because `timeout` is not a declared field. That silently reproduces
    exactly the dead-setting bug this function exists to fix. `client_kwargs` IS declared
    (dict, default {}) and is forwarded to the underlying httpx client.

    Wrapped because the field is not guaranteed across versions, and a construction that
    fails here would kill the job at startup; falling back to RAG.py's exact call is the
    safe degradation. Not a bare `except TypeError`: pydantic raises ValidationError.
    """
    kwargs = {"model": model_name, "temperature": cfg["ollama"]["temperature"]}
    timeout = cfg["ollama"].get("request_timeout")
    if timeout:
        try:
            return OllamaLLM(**kwargs, client_kwargs={"timeout": float(timeout)})
        except Exception as e:
            print(f"    [warn] client_kwargs timeout rejected ({type(e).__name__}); "
                  f"running without it, as RAG.py does")
    return OllamaLLM(**kwargs)


# ─────────────────────────────────────────────
# MAIN EVALUATION LOOPS
# ─────────────────────────────────────────────


def evaluate_dataset(csv_path, cfg):
    print(f"\n{'#'*65}")
    print(f"  DATASET: {os.path.basename(csv_path)}")
    print(f"{'#'*65}")

    try:
        # load_split returns both halves. The train half is NOT used: zero-shot's whole
        # claim is that it never sees a labelled example. It is still loaded because
        # load_split is what refuses a dataset lacking a `split` column -- the guarantee
        # that this scored the same test rows as every other classifier -- and because
        # n_train lets these files be lined up against results/rag/<dataset>.json. It costs
        # nothing: load_split reads the CSV once either way.
        df_train, df_test = load_split(REPO / csv_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"  [SKIP] {csv_path}: {e}")
        return {"error": str(e)}

    limit = cfg.get("limit_test_rows")
    if limit:
        # reset_index is not optional: accuracy_by_origin indexes y_pred by the group's
        # index labels but reads Binary_label with .iloc, so a sliced index misaligns them.
        # See the precondition in src/llm_common.py's docstring.
        df_test = df_test.head(int(limit)).reset_index(drop=True)
        print(f"  [limit_test_rows={limit}] smoke run")

    # run_provenance wants train/val/test. There is no validation half here -- zero-shot
    # selects no hyperparameters -- so n_validation is honestly 0. An empty slice of
    # df_test keeps the schema and the dtypes. Spread at top level to match how
    # svm/xgboost/bert write it (src/svm_classifier/train.py:81) and where
    # slurm_common.sh:192 reads majority_class_accuracy from.
    provenance = run_provenance(Path(csv_path).stem, df_train, df_test.iloc[0:0], df_test)
    floor = provenance["majority_class_accuracy"]

    print(f"  Train: {len(df_train)} (unused) | Test: {len(df_test)} | "
          f"majority-class floor: {floor:.4f}")

    dataset_results = {
        **provenance,
        "csv_path": csv_path,
        "prompt": "zeroshot",
        "temperature": cfg["ollama"]["temperature"],
        "models": {},
    }

    y_true = df_test["Binary_label"].tolist()

    for model_name in cfg["models"]:
        print(f"\n  {'='*55}")
        print(f"    Model: {model_name}")
        print(f"  {'='*55}")

        llm = build_llm(model_name, cfg)
        y_pred = []
        for i, review in enumerate(df_test["text"], 1):
            y_pred.append(classify_review(review, llm))
            if i % 50 == 0 or i == len(df_test):
                print(f"    {i}/{len(df_test)}", flush=True)

        metrics = score(y_true, y_pred)
        metrics["by_origin"] = accuracy_by_origin(df_test, y_pred)
        dataset_results["models"][model_name] = metrics

        if "accuracy" in metrics:
            # Against the floor, not against a retrieval baseline -- there isn't one.
            print(f"\n    Accuracy      : {metrics['accuracy']:.4f}"
                  f"   (floor {floor:.4f}, {metrics['accuracy'] - floor:+.4f})")
            print(f"    F1            : {metrics['f1']:.4f}")
            print(f"    Unknown       : {metrics['unknown_predictions']}")
            print(f"    Confusion [[TN FP] [FN TP]]: {metrics['confusion_matrix']}")
        else:
            print(f"\n    No usable predictions "
                  f"({metrics['unknown_predictions']} unknown)")

    return dataset_results


def rag_accuracy(dataset_stem, model_name):
    """That model's accuracy on the same dataset under RAG, or None.

    Read for the PRINTED table only, never written into the JSON: a results file that
    embeds another run's numbers goes stale the next time RAG is re-run, silently.
    """
    path = REPO / "results" / "rag" / f"{dataset_stem}.json"
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)["models"][model_name].get("accuracy")
    except (KeyError, json.JSONDecodeError):
        return None


def main(argv=None):
    ap = argparse.ArgumentParser(description="Zero-shot evaluation for fake review detection")
    ap.add_argument("--config", default="src/config/zeroshot.yaml")
    ap.add_argument("--dataset", action="append",
                    help="dataset CSV path (repeatable). Default: every entry in the config")
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip datasets whose output JSON already exists (resume a "
                         "run that hit its wall clock)")
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(open(REPO / args.config))

    # Respect the environment: slurm_zeroshot.sh exports a per-job port, and an
    # unconditional os.environ["OLLAMA_HOST"] = ... would overwrite it.
    os.environ.setdefault("OLLAMA_HOST", cfg["ollama"]["host"])
    print(f"OLLAMA_HOST = {os.environ['OLLAMA_HOST']}")

    out_dir = REPO / cfg["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = args.dataset if args.dataset else cfg["datasets"]
    ran = []
    for csv_path in datasets:
        target = out_dir / f"{Path(csv_path).stem}.json"
        if args.skip_existing and target.exists():
            print(f"\n  [skip-existing] {target.name} already present")
            continue
        res = evaluate_dataset(csv_path, cfg)
        save_json(res, target)          # per dataset, as it finishes
        ran.append(csv_path)

    # ── SUMMARY, REBUILT FROM DISK ────────────
    # Not from this job's results. results/rag/summary.json describes one dataset because
    # the last RAG job ran one dataset, even though all 8 per-dataset files sit beside it;
    # a summary that under-reports the study is worse than no summary.
    on_disk = {}
    for path in sorted(out_dir.glob("*.json")):
        if path.name == "summary.json":
            continue
        with open(path, encoding="utf-8") as f:
            on_disk[path.stem] = json.load(f)

    save_json({
        "run_timestamp": datetime.now().isoformat(),
        "config": args.config,
        "prompt": "zeroshot",
        "models_evaluated": cfg["models"],
        "datasets_this_run": ran,
        "datasets": on_disk,
    }, out_dir / "summary.json")

    # ── FINAL SUMMARY TABLE ───────────────────
    print(f"\n{'='*88}")
    print("  FINAL SUMMARY   (zero-shot, no retrieved examples)")
    print(f"{'='*88}")
    for stem, ds in on_disk.items():
        if "error" in ds:
            print(f"\n  {stem}: ERROR — {ds['error']}")
            continue
        floor = ds["majority_class_accuracy"]
        print(f"\n  {stem}   n_test={ds['n_test']}  floor={floor:.4f}")
        print(f"  {'Model':<30} {'Accuracy':>10} {'vs floor':>10} {'F1':>8} "
              f"{'Unknown':>8} {'RAG':>8} {'delta':>8}")
        print(f"  {'-'*84}")
        for name, m in ds["models"].items():
            if "accuracy" not in m:
                print(f"  {name:<30} {'n/a':>10} {'':>10} {'':>8} "
                      f"{m.get('unknown_predictions', 0):>8}")
                continue
            rag = rag_accuracy(stem, name)
            rag_s = f"{rag:.4f}" if rag is not None else "    -"
            delta_s = f"{m['accuracy'] - rag:+.4f}" if rag is not None else "    -"
            print(f"  {name:<30} {m['accuracy']:>10.4f} "
                  f"{m['accuracy'] - floor:>+10.4f} {m['f1']:>8.4f} "
                  f"{m['unknown_predictions']:>8} {rag_s:>8} {delta_s:>8}")

    print(f"\n  Results in {cfg['output_dir']}/")
    print("  Read every accuracy against its floor, NOT against 0.5: on d3.5_unsampled a")
    print("  model scores 0.7733 by answering \"fake\" every time.")
    print("  The `delta` column is zero-shot minus RAG -- the value retrieval added.")
    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
