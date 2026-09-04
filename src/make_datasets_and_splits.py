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

# Written to the output for stratification and error analysis; both perfectly determine
# the label, so neither may ever be used as a feature.
OUTPUT_COLUMNS = ["text", "Binary_label", "label", "Category", "origin", "cell_id", "split"]


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
    """Two-way split. Lives here rather than in src/data.py, which holds the fixed
    70/15/15 `make_split` that the three training scripts call.

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


def load_inputs(cfg, rng_seed):
    """The human corpus and the synthetic pool, both shuffled once, deterministically."""
    human = pd.read_csv(REPO / cfg["inputs"]["human_corpus"])
    human["cell_id"] = pd.NA

    frames = []
    for rel in cfg["inputs"]["synthetic_files"]:
        path = REPO / rel
        if not path.exists():
            sys.exit(f"ERROR: synthetic input not found: {rel}")
        frames.append(pd.read_csv(path))
    synth = pd.concat(frames, ignore_index=True)

    # `origin` replaces `model`/`is_synthetic` and, for the human rows, `source`.
    human["origin"] = "human_" + human["Binary_label"]
    synth["origin"] = synth["model"]

    shuffle = dict(frac=1, random_state=rng_seed)
    return human.sample(**shuffle), synth.sample(**shuffle)


def build(name, spec, human, synth, cfg, seed):
    rng = random.Random(f"{seed}:{name}")
    real = human[human["Binary_label"] == "real"]
    fake = human[human["Binary_label"] == "fake"]

    parts = [
        take(real, spec["real"], "Category", rng),
        take(fake, spec["human_fake"], "Category", rng),
        sample_synthetic(synth, spec["synthetic"], rng),
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
    out = pd.concat([train, test], ignore_index=True)[OUTPUT_COLUMNS]

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

    cells = df.dropna(subset=["cell_id"])
    if len(cells):
        per = cells.groupby(["origin", "cell_id"]).size()
        pooled = cells.groupby("cell_id").size()
        print(f"  cells:      16/{cells['cell_id'].nunique()} present   "
              f"per model-cell {per.min()}-{per.max()}   pooled {pooled.min()}-{pooled.max()}")
    print(f"  hotels:     {df['Category'].nunique()}/20")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="src/config/datasets.yaml")
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(open(REPO / args.config))
    seed = cfg["seed"]
    (REPO / cfg["output_dir"]).mkdir(parents=True, exist_ok=True)

    human, synth = load_inputs(cfg, seed)
    print(f"human corpus: {len(human)} rows   synthetic pool: {len(synth)} rows "
          f"({synth['model'].nunique()} models, {synth['cell_id'].nunique()} cells)")

    for name, spec in cfg["datasets"].items():
        df, path = build(name, spec, human, synth, cfg, seed)
        summarise(name, df, path)

    print("\nReminder: `origin` and `cell_id` are metadata — both leak the label and must "
          "never be used as features.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
