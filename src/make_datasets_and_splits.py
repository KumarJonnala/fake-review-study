"""Build the three study datasets and their 75-25 stratified train/test splits.

    python3 src/make_datasets_and_splits.py [--config src/config/datasets.yaml]

Writes one CSV per dataset to `output_dir`, each carrying a `split` column rather than
separate train/test files -- no row duplication, and one path for a training script to
load. See src/config/datasets.yaml for what goes into each and why.

The two corpus CSVs under data/ are read-only inputs. Nothing here writes to them.
"""

import argparse
import random
import sys
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

REPO = Path(__file__).resolve().parent.parent

# `Category` and `cell_id` are renamed on the way out -- the inputs keep their own names,
# so the sampling code below still addresses them as they really are.
RENAME_ON_WRITE = {"Category": "hotel", "cell_id": "cell_id_variation"}

# `origin`, `source_dataset` and `cell_id_variation` are written for stratification and
# error analysis. Each one determines or narrows the label -- a generated filename
# identifies a model, hence `fake` -- so none may be used as a feature.
#
# No separate `source` column: it would be `origin` with human_real/human_fake collapsed
# to "human", which is recoverable from `origin` alone. For a per-generator groupby, use
#     df.origin.where(~df.origin.str.startswith("human_"), "human")
OUTPUT_COLUMNS = [
    "text", "Binary_label", "label", "hotel",
    "origin", "source_dataset", "cell_id_variation", "split",
]


def largest_remainder(total, keys, avail, rng):
    """Split `total` as evenly as possible over `keys`, capped by `avail`.

    The remainder is handed out in a SEEDED SHUFFLED order, which is the one thing this
    does differently from `distribute_total` in generate_synthetic_reviews.py. That
    function gives every extra to the head of the list, which is why all 8 extra reviews
    in a 200-review run land on short cells. Reusing it here would stack the same bias on
    top of itself.
    """
    keys = list(keys)
    base, _ = divmod(total, len(keys))
    alloc = {k: min(base, avail.get(k, 0)) for k in keys}

    order = sorted(keys)
    rng.shuffle(order)
    while sum(alloc.values()) < total:
        room = [k for k in order if alloc[k] < avail.get(k, 0)]
        if not room:
            raise ValueError(
                f"cannot draw {total} rows: the pool holds only {sum(avail.values())}"
            )
        for k in room:
            if sum(alloc.values()) == total:
                break
            alloc[k] += 1
    return alloc


def make_train_test_split(df, seed, test_size, stratify_col):
    """Two-way split, written into the dataset's `split` column.

    Lives here rather than in src/data.py, which holds `split_from_column` -- the reader
    side that the three training scripts call to recover this boundary and carve a
    validation set out of the train half. This function decides the boundary; that one
    only ever reads it back.

    Stratifying on `origin` (human_real / human_fake / model name) rather than on `label`
    keeps every model proportionally represented in the test half; with the label alone a
    25% draw could take disproportionately from one model. `origin` determines `label` in
    all three datasets, so the label is stratified as a side effect.
    """
    train, test = train_test_split(
        df,
        test_size=test_size,
        stratify=df[stratify_col],
        random_state=seed,
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


def take(df, n, by, rng):
    """Take `n` rows from `df`, spread as evenly as possible over the values of `by`.

    `df` is expected to be pre-shuffled, so `head(k)` is already a random draw within a
    group and stays reproducible without a second seeded call per group.
    """
    if n == 0:
        return df.iloc[:0]
    avail = df.groupby(by).size().to_dict()
    alloc = largest_remainder(n, sorted(avail), avail, rng)
    return pd.concat([g.head(alloc[k]) for k, g in df.groupby(by)])


def sample_synthetic(pool, n, rng):
    """`n` synthetic rows, balanced over the 4 models and then the 16 cells within each.

    Balance is exact on the model axis and as even as the source permits on the cell
    axis. It cannot be exact on both: the generator writes 13 reviews to each short cell
    and 12 to each long one, so pooled over four models the cells offer 52 and 48. A
    perfectly cell-balanced draw would cap at 16 x 48 = 768, short of the 796 the study
    design calls for.
    """
    if n == 0:
        return pool.iloc[:0]
    per_model = largest_remainder(
        n, sorted(pool["model"].unique()), pool.groupby("model").size().to_dict(), rng
    )
    return pd.concat(
        [take(sub, per_model[m], "cell_id", rng) for m, sub in pool.groupby("model")]
    )


def load_pool(entries, rng_seed):
    """One named pool of synthetic reviews, shuffled once, deterministically."""
    frames = []
    for entry in entries:
        # A plain path keeps the file's own `model`; a {path, model} mapping overrides it.
        # The override is not cosmetic: gpt5.6-luna and gpt5.6-terra both ship
        # `model = "custom_review-luna-gpt"` despite being different runs with zero shared
        # texts, so without it sample_synthetic would group them into one 800-row model and
        # balance four generators three ways.
        rel = entry["path"] if isinstance(entry, dict) else entry
        path = REPO / rel
        if not path.exists():
            sys.exit(f"ERROR: synthetic input not found: {rel}")
        frame = pd.read_csv(path)
        if isinstance(entry, dict) and entry.get("model"):
            frame["model"] = entry["model"]

        # Two large-model files contain reruns of the same (cell_id, rep_index) from an
        # aborted start: deepseek repeats rep 0 of one cell, glm repeats reps 0-2. The
        # later row belongs to the run that completed, so keep="last". The small-model
        # files contain none, so this does not move d1/d2/d3.
        before = len(frame)
        frame = frame.drop_duplicates(subset=["cell_id", "rep_index"], keep="last")
        if len(frame) != before:
            print(f"  [dedup] {path.name}: {before} -> {len(frame)} rows")

        # Stamped per file, inside the loop: after the concat below the filename is gone.
        frame["source_dataset"] = path.name
        frames.append(frame)

    pool = pd.concat(frames, ignore_index=True)
    pool["origin"] = pool["model"]
    return pool.sample(frac=1, random_state=rng_seed)


def load_inputs(cfg, rng_seed):
    """The human corpus and every named synthetic pool, each shuffled deterministically."""
    corpus = REPO / cfg["inputs"]["human_corpus"]
    human = pd.read_csv(corpus)
    human["cell_id"] = pd.NA

    # `origin` is the stratification key and the only column separating a human real
    # review from a human fake. It replaces `model` and `is_synthetic`, and the corpus's
    # own `source` column (TripAdvisor/Web/MTurk) is dropped rather than carried, because
    # MTurk identifies every human fake.
    human["origin"] = "human_" + human["Binary_label"]
    human["source_dataset"] = corpus.name

    pools = {name: load_pool(entries, rng_seed)
             for name, entries in cfg["inputs"]["pools"].items()}
    return human.sample(frac=1, random_state=rng_seed), pools


def _count(value, pool):
    """`all` means every available row; anything else is a literal count.

    Asking for the whole pool needs no special path -- `largest_remainder` hands back
    exactly the available rows once the request equals what there is -- so an unsampled
    dataset is the sampled code with the numbers turned up to the ceiling.
    """
    return len(pool) if value == "all" else int(value)


def build(name, spec, human, pools, cfg, seed):
    # The dataset NAME is part of the seed, so each dataset draws its own stream and
    # adding or removing one does not shift the others. Renaming one does shift its own
    # draw, which is why d2/d3 moved by 2 and 10 rows when they gained the `_sampled` tag.
    rng = random.Random(f"{seed}:{name}")
    real = human[human["Binary_label"] == "real"]
    fake = human[human["Binary_label"] == "fake"]
    synth = pools[spec["pool"]]

    parts = [
        take(real, _count(spec["real"], real), "Category", rng),
        take(fake, _count(spec["human_fake"], fake), "Category", rng),
        sample_synthetic(synth, _count(spec["synthetic"], synth), rng),
    ]
    df = pd.concat([p for p in parts if len(p)], ignore_index=True)

    # Same 0/1 convention as load_data() in src/data.py, so a downstream trainer can use
    # either column without a second mapping.
    df["label"] = (df["Binary_label"].str.lower() == "fake").astype(int)

    train, test = make_train_test_split(
        df,
        seed=seed,
        test_size=cfg["split"]["test_size"],
        stratify_col=cfg["split"]["stratify_on"],
    )
    train["split"], test["split"] = "train", "test"
    out = pd.concat([train, test], ignore_index=True)
    out = out.rename(columns=RENAME_ON_WRITE)[OUTPUT_COLUMNS]

    path = REPO / cfg["output_dir"] / f"{name}.csv"
    out.to_csv(path, index=False)
    return out, path


def summarise(name, df, path):
    print(f"\n{name}  ->  {path.relative_to(REPO)}")
    print(f"  rows {len(df)}   " + "  ".join(
        f"{k} {v}" for k, v in df["Binary_label"].value_counts().sort_index().items()))
    print("  by origin:  " + "  ".join(
        f"{k} {v}" for k, v in df["origin"].value_counts().sort_index().items()))
    print("  by split:   " + "  ".join(
        f"{k} {v}" for k, v in df["split"].value_counts().items()))
    print("  from files: " + "  ".join(
        f"{k} {v}" for k, v in df["source_dataset"].value_counts().sort_index().items()))

    cells = df.dropna(subset=["cell_id_variation"])
    if len(cells):
        per = cells.groupby(["origin", "cell_id_variation"]).size()
        pooled = cells.groupby("cell_id_variation").size()
        print(f"  cells:      16/{cells['cell_id_variation'].nunique()} present   "
              f"per model-cell {per.min()}-{per.max()}   pooled {pooled.min()}-{pooled.max()}")
    print(f"  hotels:     {df['hotel'].nunique()}/20")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="src/config/datasets.yaml")
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(open(REPO / args.config))
    seed = cfg["seed"]
    (REPO / cfg["output_dir"]).mkdir(parents=True, exist_ok=True)

    human, pools = load_inputs(cfg, seed)
    print(f"human corpus: {len(human)} rows")
    for pool_name, pool in pools.items():
        print(f"  pool {pool_name:<14} {len(pool):>5} rows  "
              f"{pool['model'].nunique()} models, {pool['cell_id'].nunique()} cells  "
              f"({', '.join(f'{m} {n}' for m, n in sorted(pool['model'].value_counts().items()))})")

    for name, spec in cfg["datasets"].items():
        df, path = build(name, spec, human, pools, cfg, seed)
        summarise(name, df, path)

    print("\nReminder: origin, source_dataset and cell_id_variation are metadata — each "
          "leaks the label and must never be used as a feature.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
