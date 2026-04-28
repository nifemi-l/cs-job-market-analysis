"""
build_reddit_final_text.py

Step 9 of the project:
Create one final text field for analysis by combining title and body when appropriate.

Rules:
- If both title and body exist, final_text = title + " " + body
- If only title exists, final_text = title
- If only body exists, final_text = body
- If both are empty, drop the row

Run:
    python src/pipeline/build_reddit_final_text.py
"""

import argparse
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent

import pandas as pd

DEFAULT_INPUT = SRC_DIR / "data" / "reddit" / "processed" / "all_subreddits_selected_fields.csv"
DEFAULT_OUTPUT = SRC_DIR / "data" / "reddit" / "processed" / "all_subreddits_final_text.csv"

REQUIRED_COLUMNS = [
    "id",
    "subreddit",
    "author",
    "created_utc",
    "title",
    "body",
]

PLACEHOLDER_VALUES = {
    "",
    "[deleted]",
    "[removed]",
    "nan",
    "none",
    "null",
}


def normalize_text(value) -> str:
    """
    Convert missing values to empty string, strip whitespace,
    and collapse repeated internal whitespace.
    """
    if pd.isna(value):
        return ""

    text = str(value).strip()
    text = " ".join(text.split())

    if text.lower() in PLACEHOLDER_VALUES:
        return ""

    return text


def build_final_text(title: str, body: str) -> str:
    """
    Combine title and body into one final text field.
    """
    if title and body:
        return f"{title} {body}"
    if title:
        return title
    if body:
        return body
    return ""


def main():
    parser = argparse.ArgumentParser(
        description="Step 9: create one final Reddit text field by combining title and body."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(DEFAULT_INPUT),
        help="Path to Step 8 selected-fields CSV",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help="Path to output CSV with final_text column",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    print("=" * 70)
    print("STEP 9: CREATE ONE FINAL TEXT FIELD FOR ANALYSIS")
    print("=" * 70)
    print(f"Input file:  {input_path}")
    print(f"Output file: {output_path}")
    print()

    if not input_path.exists():
        print(f"ERROR: Input file does not exist: {input_path}")
        sys.exit(1)

    print("Reading Step 8 CSV...")
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

    print("Selecting required columns only...")
    df = df[REQUIRED_COLUMNS].copy()
    print(f"Working shape: {df.shape}")
    print()

    print("Normalizing title and body fields...")
    df["title"] = df["title"].apply(normalize_text)
    df["body"] = df["body"].apply(normalize_text)

    print("Normalizing non-text fields...")
    df["id"] = df["id"].apply(normalize_text)
    df["subreddit"] = df["subreddit"].apply(normalize_text)
    df["author"] = df["author"].apply(normalize_text)
    df["created_utc"] = pd.to_numeric(df["created_utc"], errors="coerce")
    print()

    print("Computing title/body availability...")
    has_title = df["title"].str.len() > 0
    has_body = df["body"].str.len() > 0

    both_present_count = int((has_title & has_body).sum())
    title_only_count = int((has_title & ~has_body).sum())
    body_only_count = int((~has_title & has_body).sum())
    both_empty_count = int((~has_title & ~has_body).sum())

    print(f"Rows with both title and body: {both_present_count:,}")
    print(f"Rows with title only:          {title_only_count:,}")
    print(f"Rows with body only:           {body_only_count:,}")
    print(f"Rows with both empty:          {both_empty_count:,}")
    print()

    print("Building final_text column...")
    df["final_text"] = [
        build_final_text(title, body)
        for title, body in zip(df["title"], df["body"])
    ]

    print("Dropping rows where final_text is empty...")
    before_drop = len(df)
    df = df[df["final_text"].str.len() > 0].reset_index(drop=True)
    dropped_rows = before_drop - len(df)

    print(f"Rows before drop: {before_drop:,}")
    print(f"Rows dropped:     {dropped_rows:,}")
    print(f"Rows remaining:   {len(df):,}")
    print()

    print("Checking for missing created_utc after numeric conversion...")
    missing_created_utc = int(df["created_utc"].isna().sum())
    print(f"Missing created_utc values: {missing_created_utc:,}")
    print()

    print("Preview of Step 9 output:")
    preview_cols = ["id", "subreddit", "title", "body", "final_text"]
    print(df[preview_cols].head(5).to_string(index=False))
    print()

    print("Creating output folder if needed...")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Saving Step 9 CSV...")
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
