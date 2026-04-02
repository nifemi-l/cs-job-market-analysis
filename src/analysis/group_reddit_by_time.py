"""
group_reddit_by_time.py

Step 14 of the project:
Split or group the Reddit posts by time period, such as pre-COVID,
COVID, and post-COVID, or by month.

Run:
    python src/analysis/group_reddit_by_time.py
"""

import argparse
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent

import pandas as pd

DEFAULT_INPUT = SRC_DIR / "data" / "reddit" / "predictions" / "RS_2023-02_sentiment_predictions.csv"
DEFAULT_OUTPUT = SRC_DIR / "data" / "reddit" / "analysis" / "RS_2023-02_time_grouped.csv"

REQUIRED_COLUMNS = [
    "id",
    "subreddit",
    "author",
    "created_utc",
    "title",
    "body",
    "final_text",
    "cleaned_text",
    "predicted_label",
    "predicted_sentiment",
]


def assign_time_period(year):
    """
    Map a calendar year to one of the project time periods.
    """
    if pd.isna(year):
        return "unknown"
    year = int(year)

    if 2017 <= year <= 2019:
        return "pre-COVID"
    elif 2020 <= year <= 2021:
        return "COVID"
    elif year >= 2022:
        return "post-COVID"
    else:
        return "outside-study-range"


def main():
    parser = argparse.ArgumentParser(
        description="Step 14: group Reddit posts by month and project time period."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(DEFAULT_INPUT),
        help="Path to Step 13 sentiment prediction CSV",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help="Path to output CSV with time grouping columns",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    print("=" * 70)
    print("STEP 14: GROUP REDDIT POSTS BY TIME PERIOD")
    print("=" * 70)
    print(f"Input file:  {input_path}")
    print(f"Output file: {output_path}")
    print()

    if not input_path.exists():
        print(f"ERROR: Input file does not exist: {input_path}")
        sys.exit(1)

    print("Reading Step 13 prediction CSV...")
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

    print("Converting created_utc to numeric...")
    df["created_utc"] = pd.to_numeric(df["created_utc"], errors="coerce")
    missing_created_utc = int(df["created_utc"].isna().sum())
    print(f"Rows with invalid created_utc after numeric conversion: {missing_created_utc:,}")
    print()

    print("Converting created_utc to UTC datetime...")
    df["created_datetime_utc"] = pd.to_datetime(
        df["created_utc"],
        unit="s",
        utc=True,
        errors="coerce",
    )

    invalid_datetime = int(df["created_datetime_utc"].isna().sum())
    print(f"Rows with invalid UTC datetime: {invalid_datetime:,}")
    print()

    print("Creating time grouping columns...")
    df["year"] = df["created_datetime_utc"].dt.year
    df["month"] = df["created_datetime_utc"].dt.month
    df["year_month"] = df["created_datetime_utc"].dt.strftime("%Y-%m")
    df["time_period"] = df["year"].apply(assign_time_period)
    print("Time grouping columns created.")
    print()

    print("Checking distribution by year_month...")
    year_month_counts = df["year_month"].value_counts(dropna=False).sort_index()
    print(year_month_counts.to_string())
    print()

    print("Checking distribution by time_period...")
    time_period_counts = df["time_period"].value_counts(dropna=False)
    print(time_period_counts.to_string())
    print()

    print("Checking predicted sentiment by time_period...")
    time_period_sentiment = (
        df.groupby(["time_period", "predicted_sentiment"])
          .size()
          .unstack(fill_value=0)
    )
    print(time_period_sentiment.to_string())
    print()

    print("Preview of Step 14 output:")
    preview_cols = [
        "id",
        "subreddit",
        "created_utc",
        "created_datetime_utc",
        "year_month",
        "time_period",
        "predicted_sentiment",
    ]
    print(df[preview_cols].head(5).to_string(index=False))
    print()

    print("Creating output folder if needed...")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Saving Step 14 CSV...")
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
