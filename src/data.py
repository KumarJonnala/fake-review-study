from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(path, text_column="text", label_column="Binary_label"):
    df = pd.read_csv(path)
    if df[text_column].isna().any() or df[label_column].isna().any():
        raise ValueError("Missing text or labels detected.")
    df = df.copy()
    df[text_column] = df[text_column].astype(str).str.strip()
    df["label"] = (df[label_column].str.lower() == "fake").astype(int)
    if df["label"].nunique() != 2:
        raise ValueError("Expected exactly two classes.")
    return df


def split_from_column(df, validation_size=0.15, seed=42, stratify_column="origin"):
    """Train / validation / test using the dataset's OWN `split` column.

    The train/test boundary is fixed in the CSV by make_datasets_and_splits.py and is
    never recomputed here. Re-deriving it -- which the old 70/15/15 `make_split` did --
    discards the origin-stratified 75/25 the datasets were built with, and because each
    training script drew its own, a row could sit in train for one model and in test for
    another. Nothing downstream could then compare the three models on equal footing.

    Validation is carved out of the TRAIN half only, so the test half stays untouched
    until the single final evaluation. `validation_size` is a fraction OF THAT TRAIN HALF,
    not of the whole dataset: 0.15 of 1194 gives 1015/179/398.

    Stratified on `origin` for the same reason the train/test split is (see
    make_datasets_and_splits.py:make_train_test_split): it keeps every generating model
    proportionally present in validation, which stratifying on the label alone does not.
    `origin` determines the label, so the label is stratified as a side effect.
    """
    for column in ("split", stratify_column):
        if column not in df.columns:
            raise ValueError(
                f"'{column}' column not found. This loader expects a dataset built by "
                f"src/make_datasets_and_splits.py; got columns {list(df.columns)}."
            )

    train_all = df[df["split"] == "train"]
    test = df[df["split"] == "test"]
    for name, part in (("train", train_all), ("test", test)):
        if part.empty:
            raise ValueError(
                f"no rows with split == '{name}'. Values present: "
                f"{sorted(df['split'].dropna().unique())}"
            )

    train, val = train_test_split(
        train_all,
        test_size=validation_size,
        stratify=train_all[stratify_column],
        random_state=seed,
    )
    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def prepare_dataset(base, name):
    """Load one study dataset by name and split it exactly as config.yaml specifies.

    Every trainer goes through here, so all three see identical train/validation/test
    rows for a given dataset -- which is the whole point of the datasets carrying their
    own `split` column.
    """
    path = Path(base["data"]["dataset_dir"]) / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"dataset not found: {path}. Build the datasets first with "
            f"`python3 src/make_datasets_and_splits.py`."
        )
    df = load_data(path, base["data"]["text_column"], base["data"]["label_column"])
    return split_from_column(
        df,
        validation_size=base["split"]["validation_size"],
        seed=base["split"]["random_seed"],
        stratify_column=base["split"]["stratify_on"],
    )
