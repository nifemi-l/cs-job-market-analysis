"""
preprocess_reddit_text.py

Step 10 of the project:
Apply the same text preprocessing used for Sentiment140 to the Reddit posts.

Run:
    python src/pipeline/preprocess_reddit_text.py
"""

import argparse
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC_DIR))

import pandas as pd

from utils.preprocessing import clean_text

DEFAULT_INPUT = SRC_DIR / "data" / "reddit" / "processed" / "RS_2023-02_final_text.csv"
DEFAULT_OUTPUT = SRC_DIR / "data" / "reddit" / "processed" / "RS_2023-02_cleaned_text.csv"

REQUIRED_COLUMNS = [
    "id",
    "subreddit",
    "author",
    "created_utc",
    "title",
    "body",
    "final_text",
]


def normalize_text(value) -> str:
    """Convert missing values to empty string and strip outer whitespace."""
    if pd.isna(value):
        return ""
    return str(value).strip()


def main():
    parser = argparse.ArgumentParser(
        description="Step 10: apply Sentiment140 text preprocessing to Reddit final_text."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(DEFAULT_INPUT),
        help="Path to Step 9 CSV with final_text",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help="Path to output CSV with cleaned_text",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    print("=" * 70)
    print("STEP 10: APPLY SENTIMENT140 PREPROCESSING TO REDDIT POSTS")
    print("=" * 70)
    print(f"Input file:  {input_path}")
    print(f"Output file: {output_path}")
    print()

    if not input_path.exists():
        print(f"ERROR: Input file does not exist: {input_path}")
        sys.exit(1)

    print("Reading Step 9 CSV...")
    df = pd.read_csv(input_path)
    print(f"Input loaded successfully. Shape: {df.shape}")
    print()

    print("Checking required columns...")
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        print("ERROR: The input file is missing required columns:")
        for col in missing:
            print(f"  - {col}")
        print()
        print("Available columns are:")
        for col in df.columns.tolist():
            print(f"  - {col}")
        sys.exit(1)

    print("All required columns are present.")
    print()

    print("Keeping only the expected Step 9 columns...")
    df = df[REQUIRED_COLUMNS].copy()
    print(f"Working shape: {df.shape}")
    print()

    print("Normalizing final_text before cleaning...")
    df["final_text"] = df["final_text"].apply(normalize_text)

    empty_final_text_before = int((df["final_text"].str.len() == 0).sum())
    print(f"Rows with empty final_text before cleaning: {empty_final_text_before:,}")
    print()

    print("Applying clean_text() from preprocessing.py to final_text...")
    df["cleaned_text"] = df["final_text"].apply(clean_text)
    print("Cleaning completed.")
    print()

    empty_cleaned_text = int((df["cleaned_text"].str.len() == 0).sum())
    print(f"Rows with empty cleaned_text after cleaning: {empty_cleaned_text:,}")
    print()

    print("Dropping rows where cleaned_text is empty...")
    rows_before = len(df)
    df = df[df["cleaned_text"].str.len() > 0].reset_index(drop=True)
    rows_after = len(df)
    dropped_rows = rows_before - rows_after

    print(f"Rows before drop: {rows_before:,}")
    print(f"Rows dropped:     {dropped_rows:,}")
    print(f"Rows remaining:   {rows_after:,}")
    print()

    print("Preview of Step 10 output:")
    preview_cols = ["id", "subreddit", "final_text", "cleaned_text"]
    print(df[preview_cols].head(5).to_string(index=False))
    print()

    print("Creating output folder if needed...")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Saving Step 10 CSV...")
    df.to_csv(output_path, index=False, encoding="utf-8")
    print("File saved successfully.")
    print()

    print("Final summary")
    print("-" * 70)
    print(f"Rows written:    {len(df):,}")
    print(f"Columns written: {len(df.columns):,}")
    print("Final columns:")
    for col in df.columns:
        print(f"  - {col}")
    print()
    print(f"Saved output file to: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
