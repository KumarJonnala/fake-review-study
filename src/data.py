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

def make_split(df, seed=42, train_size=0.70, validation_size=0.15):
    train, temp = train_test_split(
        df,
        train_size=train_size,
        stratify=df["label"],
        random_state=seed,
    )
    relative_val = validation_size / (1.0 - train_size)
    val, test = train_test_split(
        temp,
        train_size=relative_val,
        stratify=temp["label"],
        random_state=seed,
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)

def save_split_indices(train, val, test, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    train.index.to_series().to_csv(Path(output_dir) / "train_indices.csv", index=False)
    val.index.to_series().to_csv(Path(output_dir) / "validation_indices.csv", index=False)
    test.index.to_series().to_csv(Path(output_dir) / "test_indices.csv", index=False)
