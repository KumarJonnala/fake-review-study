"""
Multi-Dataset × Multi-Model RAG Evaluation for Fake Review Detection
---------------------------------------------------------------------
Outer loop  : CSV datasets from data/datasets/
Inner loop  : Ollama LLM models
Output      : results/rag/<dataset>.json + summary.json

    python3 src/rag_classifier/RAG.py [--config src/config/rag.yaml]

Carried over from a previous project and adapted to this one. What changed:

* Datasets, models, k and the Ollama host come from src/config/rag.yaml instead of
  being hardcoded here.
* NO SPLITTING. The old version called train_test_split(test_size=0.5) itself. The
  datasets in data/datasets/ already carry a seeded 75-25 stratified `split` column from
  make_datasets_and_splits.py, so the vector store is built from `split == "train"` and
  every classified review comes from `split == "test"`. That is the only way RAG lands on
  the same partition as the SVM, XGBoost and BERT classifiers and stays comparable.
* A RETRIEVAL-ONLY BASELINE is reported beside every model. Majority label of the k
  retrieved neighbours, no LLM involved. On d2 that alone scores ~98%, because the
  generated reviews repeat each other enough that a synthetic test review retrieves
  near-identical synthetic training reviews already labelled `fake`. Without the baseline
  printed next to it, a high RAG score reads as an LLM result when it is a
  nearest-neighbour one.
* Per-origin accuracy, because the configured judges also WROTE the fakes in d2/d3 --
  a model recognising its own output is a confound worth being able to see.
* Metrics come from src/evaluation.py so the JSON keys match results/svm/<dataset>/*.json.
* Two bugs fixed; see normalize_prediction and `unknown_count` below.

Only `text` is ever shown to a model. `origin`, `source_dataset` and `cell_id_variation`
all leak the label; `origin` is kept for the test rows only, to break results down after
the fact.
"""

import argparse
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM

# langchain_community's HuggingFaceEmbeddings is deprecated in favour of the dedicated
# package; try that first so a current install does not warn, and fall back so an older
# environment still runs.
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
from src.evaluation import evaluate_binary, save_json  # noqa: E402

# ─────────────────────────────────────────────
# PROMPT
# ─────────────────────────────────────────────

rag_prompt = PromptTemplate(
    input_variables=["examples", "review"],
    template="""
    You are an expert at detecting fake and real reviews of hotel. Carefully analyze the following review for signs such as:
    
    Task: Classify the following review of a Hotel review as either "real" or "fake". Carefully analyze the following review for signs such as:
    - Overly generic or vague language
    - Exaggerated praise or criticism
    - Repetitive or templated phrasing
    - Marketing-like wording or unnatural flow
    - Specific details vs. general statements

    Here are some examples:
    {examples}

    Now classify this review:
    "{review}"

    Classify this review as either "real" or "fake" (respond with only one word):

"""
)

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def load_split(csv_path: Path):
    """The train and test halves of one dataset, taken from its `split` column.

    Nothing is re-split here. A dataset without the column is refused rather than
    silently split onto a different partition from the other classifiers.

    The old version did `df.rename(columns={label_col: "label"})`, which assumed the
    source had no `label` column. These datasets have BOTH `Binary_label` (real/fake) and
    `label` (0/1), so that rename produced two columns named `label` and `df["label"]`
    then returned a DataFrame instead of a Series. No rename now: `label` feeds the
    metrics and `Binary_label` feeds the prompt examples.
    """
    df = pd.read_csv(csv_path)
    for col in ("split", "text", "label", "Binary_label"):
        if col not in df.columns:
            raise ValueError(f"no `{col}` column")
    train = df[df["split"] == "train"].reset_index(drop=True)
    test = df[df["split"] == "test"].reset_index(drop=True)
    if train.empty or test.empty:
        raise ValueError("empty train or test half")
    return train, test


def build_vectorstore(df_train: pd.DataFrame, embeddings, store_dir: str) -> FAISS:
    """Build a fresh FAISS vectorstore from the training split.

    `Binary_label` is stored, not `label`: it is what gets pasted into the prompt as the
    example's answer, and "real"/"fake" is what the model is asked to reply with.
    """
    documents = [
        Document(page_content=row["text"], metadata={"label": row["Binary_label"]})
        for _, row in df_train.iterrows()
    ]
    vs = FAISS.from_documents(documents, embeddings)
    vs.save_local(store_dir)
    return FAISS.load_local(store_dir, embeddings, allow_dangerous_deserialization=True)


def format_examples(docs) -> str:
    return "\n\n".join(
        f'Review: "{d.page_content}"\nLabel: {d.metadata["label"]}'
        for d in docs
    )


def normalize_prediction(response: str) -> str:
    """"real", "fake", or "UNKNOWN".

    The old version matched "fake" before "real" anywhere in the string, so
    "this is not fake, it's real" scored as fake. The prompt asks for exactly one word,
    so check the first word first, then fall back to substring matching -- and return a
    genuine UNKNOWN when neither or BOTH appear, instead of silently picking one.
    """
    r = str(response).strip().lower().strip('"\'.` \n')
    words = r.split()
    first = words[0].strip('.,:;"\'') if words else ""
    if first in ("real", "truthful", "genuine"):
        return "real"
    if first in ("fake", "deceptive"):
        return "fake"
    has_fake = any(w in r for w in ("fake", "deceptive"))
    has_real = any(w in r for w in ("real", "truthful", "genuine"))
    if has_fake and not has_real:
        return "fake"
    if has_real and not has_fake:
        return "real"
    return "UNKNOWN"


def classify_review(review_text: str, llm, retriever) -> str:
    docs     = retriever.invoke(review_text)
    examples = format_examples(docs)
    prompt   = rag_prompt.format(examples=examples, review=review_text)
    try:
        response = llm.invoke(prompt)
    except Exception as e:
        # One dead call must not abort a two-hour run. Counted as UNKNOWN and reported.
        print(f"    [error] {type(e).__name__}: {e}")
        return "UNKNOWN"
    return normalize_prediction(response)


def retrieval_only_prediction(review_text: str, retriever) -> str:
    """Majority label of the k retrieved neighbours -- the no-LLM baseline.

    Every model number is only interpretable against this. On d2 it is near ceiling on its
    own, because the generated reviews repeat each other enough that retrieval alone
    answers the question.
    """
    labels = [d.metadata["label"] for d in retriever.invoke(review_text)]
    if not labels:
        return "UNKNOWN"
    return "fake" if labels.count("fake") * 2 >= len(labels) else "real"


def score(y_true_str, y_pred_str) -> dict:
    """Metrics over the rows that got a usable answer, plus a count of those that did not.

    UNKNOWNs are excluded from the metrics but REPORTED. The old version filtered them
    with no record of how many, which quietly inflates accuracy when a model refuses on
    exactly the hard cases.
    """
    keep = [i for i, p in enumerate(y_pred_str) if p in ("real", "fake")]
    n_unknown = len(y_pred_str) - len(keep)
    if not keep:
        return {"error": "no usable predictions", "unknown_predictions": n_unknown}
    yt = [1 if y_true_str[i] == "fake" else 0 for i in keep]
    yp = [1 if y_pred_str[i] == "fake" else 0 for i in keep]
    out = evaluate_binary(yt, yp)
    out["unknown_predictions"] = n_unknown
    out["scored_rows"] = len(keep)
    return out


def accuracy_by_origin(df_test: pd.DataFrame, y_pred_str) -> dict:
    """Accuracy per origin: the only way to see whether a model finds its OWN generations
    easier than another model's, which is the self-recognition confound in d2 and d3."""
    out = {}
    for origin, g in df_test.groupby("origin"):
        idx = g.index.tolist()
        keep = [i for i in idx if y_pred_str[i] in ("real", "fake")]
        correct = sum(1 for i in keep
                      if y_pred_str[i] == df_test["Binary_label"].iloc[i])
        out[origin] = {
            "n": len(idx),
            "accuracy": round(correct / len(keep), 4) if keep else None,
            "unknown_predictions": len(idx) - len(keep),
        }
    return out


# ─────────────────────────────────────────────
# MAIN EVALUATION LOOPS
# ─────────────────────────────────────────────

def evaluate_dataset(csv_path, cfg, embeddings, store_dir):
    k = cfg["retrieval"]["k"]
    print(f"\n{'#'*65}")
    print(f"  DATASET: {os.path.basename(csv_path)}")
    print(f"{'#'*65}")

    try:
        df_train, df_test = load_split(REPO / csv_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"  [SKIP] {csv_path}: {e}")
        return {"error": str(e)}

    limit = cfg.get("limit_test_rows")
    if limit:
        df_test = df_test.head(int(limit)).reset_index(drop=True)
        print(f"  [limit_test_rows={limit}] smoke run")
    print(f"  Train: {len(df_train)} | Test: {len(df_test)} | k={k}")

    print("  Building vector store...")
    vectorstore = build_vectorstore(df_train, embeddings, store_dir)
    retriever   = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )

    y_true = df_test["Binary_label"].tolist()

    # Baseline first, deliberately: it frames every model number that follows.
    print(f"\n  Retrieval-only {k}-NN baseline (no LLM)...")
    knn = [retrieval_only_prediction(t, retriever) for t in df_test["text"]]
    baseline = score(y_true, knn)
    print(f"    accuracy {baseline['accuracy']:.4f}  f1 {baseline['f1']:.4f}")

    dataset_results = {
        "csv_path":   csv_path,
        "train_rows": len(df_train),
        "test_rows":  len(df_test),
        "k":          k,
        "embedding_model": cfg["retrieval"]["embedding_model"],
        "retrieval_only_baseline":  baseline,
        "retrieval_only_by_origin": accuracy_by_origin(df_test, knn),
        "models": {},
    }

    for model_name in cfg["models"]:
        print(f"\n  {'='*55}")
        print(f"    Model: {model_name}")
        print(f"  {'='*55}")

        llm = OllamaLLM(model=model_name, temperature=cfg["ollama"]["temperature"])
        y_pred = []
        for i, review in enumerate(df_test["text"], 1):
            y_pred.append(classify_review(review, llm, retriever))
            if i % 50 == 0 or i == len(df_test):
                print(f"    {i}/{len(df_test)}", flush=True)

        metrics = score(y_true, y_pred)
        metrics["by_origin"] = accuracy_by_origin(df_test, y_pred)
        dataset_results["models"][model_name] = metrics

        if "accuracy" in metrics:
            print(f"\n    Accuracy      : {metrics['accuracy']:.4f}"
                  f"   (baseline {baseline['accuracy']:.4f},"
                  f" {metrics['accuracy'] - baseline['accuracy']:+.4f})")
            print(f"    F1            : {metrics['f1']:.4f}")
            print(f"    Unknown       : {metrics['unknown_predictions']}")
            print(f"    Confusion [[TN FP] [FN TP]]: {metrics['confusion_matrix']}")
        else:
            print(f"\n    No usable predictions "
                  f"({metrics['unknown_predictions']} unknown)")

    return dataset_results


def main(argv=None):
    ap = argparse.ArgumentParser(description="RAG evaluation for fake review detection")
    ap.add_argument("--config", default="src/config/rag.yaml")
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(open(REPO / args.config))

    # Respect the environment: slurm_RAG.sh exports a per-job port, and the old
    # unconditional os.environ["OLLAMA_HOST"] = ... would have overwritten it.
    os.environ.setdefault("OLLAMA_HOST", cfg["ollama"]["host"])
    print(f"OLLAMA_HOST = {os.environ['OLLAMA_HOST']}")

    print(f"Loading embedding model {cfg['retrieval']['embedding_model']}...")
    embeddings = HuggingFaceEmbeddings(model_name=cfg["retrieval"]["embedding_model"])

    final_output = {
        "run_timestamp": datetime.now().isoformat(),
        "config": args.config,
        "models_evaluated": cfg["models"],
        "datasets": {},
    }

    out_dir = REPO / cfg["output_dir"]
    # A temp dir, not the working directory: the old VECTORSTORE_PATH dropped a
    # `temp_vectorstore/` folder wherever the script happened to be run from.
    with tempfile.TemporaryDirectory(prefix="rag_faiss_") as store_dir:
        for csv_path in cfg["datasets"]:
            res = evaluate_dataset(csv_path, cfg, embeddings, store_dir)
            final_output["datasets"][csv_path] = res
            save_json(res, out_dir / f"{Path(csv_path).stem}.json")

    save_json(final_output, out_dir / "summary.json")

    # ── FINAL SUMMARY TABLE ───────────────────
    print(f"\n{'='*80}")
    print("  FINAL SUMMARY")
    print(f"{'='*80}")
    for csv_path, ds in final_output["datasets"].items():
        if "error" in ds:
            print(f"\n  {os.path.basename(csv_path)}: ERROR — {ds['error']}")
            continue
        print(f"\n  {os.path.basename(csv_path)}")
        print(f"  {'Model':<30} {'Accuracy':>10} {'F1':>10} {'Unknown':>9}")
        print(f"  {'-'*61}")
        rows = [(f"retrieval-only {ds['k']}-NN", ds["retrieval_only_baseline"])]
        rows += list(ds["models"].items())
        for name, m in rows:
            # `unknown_predictions`, not `unknown_count`: the old summary read a key that
            # was never written, so every run died here after saving the JSON.
            acc = f"{m['accuracy']:.4f}" if "accuracy" in m else "     n/a"
            f1 = f"{m['f1']:.4f}" if "f1" in m else "     n/a"
            print(f"  {name:<30} {acc:>10} {f1:>10} "
                  f"{m.get('unknown_predictions', 0):>9}")

    print(f"\n  Results in {cfg['output_dir']}/")
    print("  Read every model row against the retrieval-only baseline above it: on d2")
    print("  the baseline alone is near ceiling, so a high score there is a")
    print("  nearest-neighbour result, not evidence the LLM can judge.")
    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())