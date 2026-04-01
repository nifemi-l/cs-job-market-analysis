"""
select_reddit_fields.py

Step 8 of the project:
Keep only the important fields from each Reddit post.

Input:
- Step 7 filtered Reddit CSV

Output:
- A smaller CSV containing only the fields needed for later steps

Fields kept:
- id
- subreddit
- author
- created_utc
- title
- body

Run from src/:
    python .\select_reddit_fields.py

Optional:
    python .\select_reddit_fields.py --input .\data\reddit\filtered\RS_2023-02_filtered.csv --output .\data\reddit\processed\RS_2023-02_selected_fields.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


DEFAULT_INPUT = Path("data/reddit/filtered/RS_2023-02_filtered.csv")
DEFAULT_OUTPUT = Path("data/reddit/processed/RS_2023-02_selected_fields.csv")

KEEP_COLUMNS = [
    "id",
    "subreddit",
    "author",
    "created_utc",
    "title",
    "body",
]


def normalize_text(value):
    """Convert missing values to empty string and strip whitespace."""
    if pd.isna(value):
        return ""
    return str(value).strip()


def main():
    parser = argparse.ArgumentParser(
        description="Step 8: keep only the important Reddit fields for later sentiment analysis."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(DEFAULT_INPUT),
        help="Path to Step 7 filtered Reddit CSV",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help="Path to output CSV for selected fields",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    print("=" * 70)
    print("STEP 8: KEEP ONLY IMPORTANT REDDIT FIELDS")
    print("=" * 70)
    print(f"Input file:  {input_path}")
    print(f"Output file: {output_path}")
    print(f"Columns to keep: {KEEP_COLUMNS}")
    print()

    if not input_path.exists():
        print(f"ERROR: Input file does not exist: {input_path}")
        sys.exit(1)

    print("Reading input CSV...")
    df = pd.read_csv(input_path)
    print(f"Input loaded successfully. Shape: {df.shape}")
    print()

    print("Checking required columns...")
    missing = [col for col in KEEP_COLUMNS if col not in df.columns]
    if missing:
        print("ERROR: The input file is missing required columns:")
        for col in missing:
            print(f"  - {col}")
        print()
        print("Available columns are:")
        for col in df.columns.tolist():
            print(f"  - {col}")
        sys.exit(1)

    print("Selecting only the important columns...")
    selected = df[KEEP_COLUMNS].copy()
    print(f"Selected shape: {selected.shape}")
    print()

    print("Normalizing text fields...")
    for col in ["id", "subreddit", "author", "title", "body"]:
        selected[col] = selected[col].apply(normalize_text)

    print("Normalizing created_utc...")
    selected["created_utc"] = pd.to_numeric(selected["created_utc"], errors="coerce")

    print("Checking for missing values after normalization...")
    print(f"Missing id values:          {selected['id'].eq('').sum():,}")
    print(f"Missing subreddit values:   {selected['subreddit'].eq('').sum():,}")
    print(f"Missing author values:      {selected['author'].eq('').sum():,}")
    print(f"Missing title values:       {selected['title'].eq('').sum():,}")
    print(f"Missing body values:        {selected['body'].eq('').sum():,}")
    print(f"Missing created_utc values: {selected['created_utc'].isna().sum():,}")
    print()

    print("Preview of cleaned Step 8 data:")
    print(selected.head(5).to_string(index=False))
    print()

    print("Creating output folder if needed...")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Saving selected fields CSV...")
    selected.to_csv(output_path, index=False, encoding="utf-8")
    print("File saved successfully.")
    print()

    print("Final summary")
    print("-" * 70)
    print(f"Rows written:    {len(selected):,}")
    print(f"Columns written: {len(selected.columns):,}")
    print("Final columns:")
    for col in selected.columns:
        print(f"  - {col}")
    print()
    print(f"Saved output file to: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()