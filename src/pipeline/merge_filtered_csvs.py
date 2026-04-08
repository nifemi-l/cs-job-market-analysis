"""
merge_filtered_csvs.py

Merge per-subreddit filtered CSV files into one dataset.

Command:
    python src/pipeline/merge_filtered_csvs.py
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

SRC_DIR = Path(__file__).resolve().parent.parent

DEFAULT_INPUT_DIR = SRC_DIR / "data" / "reddit" / "filtered"
DEFAULT_OUTPUT = DEFAULT_INPUT_DIR / "all_subreddits_filtered.csv"

def main():
    parser = argparse.ArgumentParser(
        description="Merge per-subreddit filtered CSVs into one combined file."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=str(DEFAULT_INPUT_DIR),
        help="Directory containing the per-subreddit filtered CSVs.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help="Path for the merged output CSV.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_path = Path(args.output)

    print("=" * 70)
    print("MERGE FILTERED SUBREDDIT CSVs")
    print("=" * 70)
    print(f"Input directory: {input_dir}")
    print(f"Output file:     {output_path}")
    print()

    csv_files = sorted(input_dir.glob("*_filtered.csv"))
    if not csv_files:
        print(f"ERROR: No *_filtered.csv files found in {input_dir}")
        sys.exit(1)

    frames = []
    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        print(f"  {csv_path.name}: {len(df):,} rows")
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined.drop_duplicates(subset=["id"], keep="first", inplace=True)
    combined.sort_values("created_utc", inplace=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False, encoding="utf-8")

    print()
    print(f"Combined rows (after dedup): {len(combined):,}")
    print(f"Subreddits present: {sorted(combined['subreddit'].unique())}")
    print(f"Saved to: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
