"""Load, clean and balance the Sign Language MNIST gesture dataset."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.utils import resample

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OPENML_ID = 45082  # SignMNIST: first 27,455 rows are train, last 7,172 are test
N_TRAIN = 27455
IMG_SIZE = 28

# Labels 0-24 map to A-Y; 9 (J) and 25 (Z) need motion, so they are absent from the dataset.
LETTERS = {i: chr(ord("A") + i) for i in range(25) if i != 9}


def load_raw() -> pd.DataFrame:
    """Download once (cached in data/) and return label + pixel1..pixel784 columns."""
    df = fetch_openml(data_id=OPENML_ID, as_frame=True, parser="auto", data_home=str(DATA_DIR)).frame
    df["label"] = df["label"].astype(int)
    return df


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Fill missing pixels, clip out-of-range values and drop exact duplicates."""
    pixels = df.columns.drop("label")
    stats = {"rows_in": len(df), "missing_values": int(df[pixels].isna().sum().sum())}
    df = df.dropna(subset=["label"]).copy()
    df[pixels] = df[pixels].fillna(df[pixels].median())
    stats["out_of_range_values"] = int(((df[pixels] < 0) | (df[pixels] > 255)).sum().sum())
    df[pixels] = df[pixels].clip(0, 255)
    stats["duplicates"] = int(df.duplicated().sum())
    df = df.drop_duplicates()
    stats["rows_out"] = len(df)
    return df, stats


def split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use the dataset's official train/test split (row order is preserved by clean())."""
    return df[df.index < N_TRAIN], df[df.index >= N_TRAIN]


def balance(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Upsample every class to the size of the largest class."""
    target = df["label"].value_counts().max()
    parts = [resample(g, replace=True, n_samples=target, random_state=seed) if len(g) < target else g
             for _, g in df.groupby("label")]
    return pd.concat(parts).sample(frac=1, random_state=seed)


def to_xy(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Features scaled to 0-1 and integer labels."""
    return df.drop(columns="label").to_numpy(dtype=np.float32) / 255.0, df["label"].to_numpy()


def load_splits(balanced: bool = True):
    df, stats = clean(load_raw())
    train, test = split(df)
    if balanced:
        train = balance(train)
    return train, test, stats
