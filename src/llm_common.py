"""Helpers shared by every LLM-judge classifier -- RAG and zero-shot.

Moved verbatim out of src/rag_classifier/RAG.py when the zero-shot classifier was added.
Nothing changed in the move: results/rag/ was produced by these exact function bodies and
has to stay reproducible from them, so the bodies below are byte-identical to the ones at
RAG.py:93-113, 138-159, 188-204 and 207-221 before the split. Anything worth saying about
the move is said here, in the module docstring, rather than inside a function, so that an
inspect.getsource() diff against the pre-split file stays empty.

SHARED RATHER THAN COPIED, because two of these encode fixed bugs that a second copy would
drift away from -- and the whole point of the zero-shot run is that its numbers are
subtractable from the RAG ones, which they are not if the two are scored by different code:

  * normalize_prediction -- the old version substring-matched "fake" before "real", so
    "this is not fake, it's real" scored as fake; and it silently picked a label when both
    words or neither appeared, instead of returning UNKNOWN.
  * score -- the old version dropped UNKNOWNs with no record of how many, which inflates
    accuracy exactly when a model refuses on the hard cases.

PRECONDITION for accuracy_by_origin: df_test must carry a RangeIndex. It looks y_pred up by
the group's index labels but reads Binary_label positionally with .iloc, so a non-contiguous
index silently misaligns the two. load_split's reset_index(drop=True) provides that; any
caller that slices df_test afterwards -- a limit_test_rows smoke run, say -- must reset the
index again.

Retrieval-specific code deliberately stayed in RAG.py. build_vectorstore, format_examples
and retrieval_only_prediction each take a FAISS retriever or a langchain Document, and the
zero-shot job installs neither faiss-cpu nor sentence-transformers. Importing them here
would drag a multi-GB embedding stack into a venv that has no vector store to build.
"""
import sys
from pathlib import Path

import pandas as pd

# parent.parent, not RAG.py's parent.parent.parent: this module sits directly in src/.
# The guard makes it a no-op for callers that already inserted the repo root (RAG.py:59
# does, before importing this module).
REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from src.evaluation import evaluate_binary  # noqa: E402


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
