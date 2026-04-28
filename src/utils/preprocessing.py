"""
preprocessing.py

Text cleaning + data loading for the Sentiment140 dataset.
This file has NO sklearn -- all the cleaning logic is ours.

The Sentiment140 CSV has no header row. The columns are:
  label (0=negative, 4=positive), id, date, flag, user, text

We only care about label and text. We remap label 4 -> 1 so
it's a clean binary: 0 = negative, 1 = positive.
"""

import re
import pandas as pd

SENTIMENT140_COLUMNS = ["label", "id", "date", "flag", "user", "text"]


def clean_text(text: str) -> str:
    """Strip noise from a single tweet/post so TF-IDF has cleaner tokens."""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_sentiment140(path: str) -> pd.DataFrame:
    """
    Read the raw Sentiment140 CSV and return a two-column DataFrame [label, text].

    The file uses latin-1 encoding (not utf-8) -- will get decode errors otherwise.
    Labels in the raw file are 0 and 4; we map 4 -> 1 for binary classification.
    """
    df = pd.read_csv(path, encoding="latin-1", header=None, names=SENTIMENT140_COLUMNS)
    df = df[["label", "text"]].copy()
    df["label"] = df["label"].map({0: 0, 4: 1})
    df.dropna(subset=["label"], inplace=True)
    df["label"] = df["label"].astype(int)
    return df


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run clean_text on every row. Some tweets end up completely empty
    after cleaning (e.g. a tweet that was just a URL), so we drop those.
    """
    df = df.copy()
    df["text"] = df["text"].astype(str).apply(clean_text)
    df = df[df["text"].str.len() > 0].reset_index(drop=True)
    return df
