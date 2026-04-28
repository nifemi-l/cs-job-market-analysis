"""
measure_topic_trends_over_time.py

Step 18 of the project:
Measure how often each topic appears over time so you can track trend changes.

Run:
    python src/analysis/measure_topic_trends_over_time.py
"""

import argparse
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC_DIR))

import pandas as pd

from utils.topic_buckets import get_topic_buckets

DEFAULT_INPUT = SRC_DIR / "data" / "reddit" / "analysis" / "all_subreddits_topic_tagged.csv"
DEFAULT_OUTPUT_DIR = SRC_DIR / "data" / "reddit" / "analysis" / "step18_topic_trends"

REQUIRED_BASE_COLUMNS = [
    "id",
    "subreddit",
    "created_utc",
    "created_datetime_utc",
    "year",
    "month",
    "year_month",
    "time_period",
    "has_any_topic",
]


def safe_divide(numerator, denominator):
    """Return numerator / denominator, but avoid division-by-zero errors."""
    if denominator == 0:
        return 0.0
    return numerator / denominator


def build_full_month_index(year_month_values):
    """
    Build a complete month index from min to max observed month.
    """
    cleaned = [
        str(x).strip()
        for x in year_month_values
        if pd.notna(x) and str(x).strip() != ""
    ]

    if not cleaned:
        return []

    period_index = pd.PeriodIndex(cleaned, freq="M")
    full_range = pd.period_range(start=period_index.min(), end=period_index.max(), freq="M")
    return [str(p) for p in full_range]


def main():
    parser = argparse.ArgumentParser(
        description="Step 18: measure how often each topic appears over time."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(DEFAULT_INPUT),
        help="Path to Step 16 topic-tagged CSV",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where Step 18 output files will be saved",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    print("=" * 70)
    print("STEP 18: MEASURE TOPIC TRENDS OVER TIME")
    print("=" * 70)
    print(f"Input file:  {input_path}")
    print(f"Output dir:  {output_dir}")
    print()

    if not input_path.exists():
        print(f"ERROR: Input file does not exist: {input_path}")
        sys.exit(1)

    print("Reading Step 16 topic-tagged CSV...")
    df = pd.read_csv(input_path)
    print(f"Input loaded successfully. Shape: {df.shape}")
    print()

    print("Loading topic bucket definitions...")
    topic_buckets = get_topic_buckets()
    bucket_names = list(topic_buckets.keys())
    print(f"Total topic buckets loaded: {len(bucket_names)}")
    print(f"Bucket names: {bucket_names}")
    print()

    print("Checking required columns...")
    required_topic_columns = [f"topic_{bucket}" for bucket in bucket_names]
    required_columns = REQUIRED_BASE_COLUMNS + required_topic_columns

    missing = [col for col in required_columns if col not in df.columns]
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

    print("Normalizing year_month, time_period, and topic columns...")
    df["year_month"] = df["year_month"].fillna("").astype(str).str.strip()
    df["time_period"] = df["time_period"].fillna("").astype(str).str.strip()
    df["subreddit"] = df["subreddit"].fillna("").astype(str).str.strip().str.lower()
    df["has_any_topic"] = pd.to_numeric(df["has_any_topic"], errors="coerce").fillna(0).astype(int)

    for bucket in bucket_names:
        topic_col = f"topic_{bucket}"
        df[topic_col] = pd.to_numeric(df[topic_col], errors="coerce").fillna(0).astype(int)

    invalid_year_month = int((df["year_month"] == "").sum())
    invalid_time_period = int((df["time_period"] == "").sum())
    print(f"Rows with empty year_month:  {invalid_year_month:,}")
    print(f"Rows with empty time_period: {invalid_time_period:,}")
    print()

    if len(df) == 0:
        print("ERROR: Input file has no rows.")
        sys.exit(1)

    print("Building complete month index...")
    full_months = build_full_month_index(df["year_month"].unique())
    if not full_months:
        print("ERROR: Could not build a valid month index from year_month.")
        sys.exit(1)

    print(f"Month range: {full_months[0]} to {full_months[-1]}")
    print(f"Number of months in range: {len(full_months)}")
    print()

    print("Computing monthly totals...")
    monthly_total_posts = (
        df.groupby("year_month")
          .size()
          .reindex(full_months, fill_value=0)
          .rename("total_posts_in_month")
    )

    monthly_topic_tagged_posts = (
        df.groupby("year_month")["has_any_topic"]
          .sum()
          .reindex(full_months, fill_value=0)
          .rename("topic_tagged_posts_in_month")
    )

    monthly_base_df = pd.concat([monthly_total_posts, monthly_topic_tagged_posts], axis=1).reset_index()
    monthly_base_df = monthly_base_df.rename(columns={"index": "year_month"})

    print("Monthly totals computed.")
    print(monthly_base_df.head(12).to_string(index=False))
    print()

    print("Computing monthly topic trends...")
    monthly_long_rows = []

    for bucket in bucket_names:
        topic_col = f"topic_{bucket}"

        topic_counts = (
            df.groupby("year_month")[topic_col]
              .sum()
              .reindex(full_months, fill_value=0)
        )

        for ym in full_months:
            total_posts = int(monthly_total_posts.loc[ym])
            topic_tagged_posts = int(monthly_topic_tagged_posts.loc[ym])
            topic_post_count = int(topic_counts.loc[ym])

            monthly_long_rows.append({
                "year_month": ym,
                "topic_bucket": bucket,
                "topic_description": topic_buckets[bucket]["description"],
                "total_posts_in_month": total_posts,
                "topic_tagged_posts_in_month": topic_tagged_posts,
                "topic_post_count": topic_post_count,
                "topic_share_of_all_posts": safe_divide(topic_post_count, total_posts),
                "topic_share_of_topic_tagged_posts": safe_divide(topic_post_count, topic_tagged_posts),
            })

    monthly_long_df = pd.DataFrame(monthly_long_rows)

    print("Monthly topic trends computed.")
    print()

    print("Creating monthly wide-format tables...")
    monthly_counts_wide = (
        monthly_long_df.pivot(index="year_month", columns="topic_bucket", values="topic_post_count")
                      .fillna(0)
                      .astype(int)
                      .reset_index()
    )

    monthly_share_all_wide = (
        monthly_long_df.pivot(index="year_month", columns="topic_bucket", values="topic_share_of_all_posts")
                      .fillna(0.0)
                      .reset_index()
    )

    monthly_share_topic_tagged_wide = (
        monthly_long_df.pivot(index="year_month", columns="topic_bucket", values="topic_share_of_topic_tagged_posts")
                      .fillna(0.0)
                      .reset_index()
    )

    print("Monthly wide-format tables created.")
    print()

    print("Computing time-period topic summary...")
    valid_time_period_df = df[df["time_period"] != ""].copy()

    time_period_total_posts = (
        valid_time_period_df.groupby("time_period")
                            .size()
                            .rename("total_posts_in_period")
    )

    time_period_topic_tagged_posts = (
        valid_time_period_df.groupby("time_period")["has_any_topic"]
                            .sum()
                            .rename("topic_tagged_posts_in_period")
    )

    time_period_rows = []
    for bucket in bucket_names:
        topic_col = f"topic_{bucket}"
        grouped_counts = (
            valid_time_period_df.groupby("time_period")[topic_col]
                                .sum()
        )

        for period_name in time_period_total_posts.index:
            total_posts = int(time_period_total_posts.loc[period_name])
            topic_tagged_posts = int(time_period_topic_tagged_posts.loc[period_name])
            topic_post_count = int(grouped_counts.get(period_name, 0))

            time_period_rows.append({
                "time_period": period_name,
                "topic_bucket": bucket,
                "topic_description": topic_buckets[bucket]["description"],
                "total_posts_in_period": total_posts,
                "topic_tagged_posts_in_period": topic_tagged_posts,
                "topic_post_count": topic_post_count,
                "topic_share_of_all_posts": safe_divide(topic_post_count, total_posts),
                "topic_share_of_topic_tagged_posts": safe_divide(topic_post_count, topic_tagged_posts),
            })

    time_period_summary_df = pd.DataFrame(time_period_rows)

    print("Time-period topic summary computed.")
    print()

    print("Computing subreddit-by-month topic counts...")
    subreddit_month_rows = []

    grouped_total = (
        df.groupby(["year_month", "subreddit"])
          .size()
          .rename("total_posts_in_month_subreddit")
    )

    for bucket in bucket_names:
        topic_col = f"topic_{bucket}"
        grouped_topic = (
            df.groupby(["year_month", "subreddit"])[topic_col]
              .sum()
              .rename("topic_post_count")
        )

        merged = pd.concat([grouped_total, grouped_topic], axis=1).fillna(0).reset_index()
        merged["topic_bucket"] = bucket
        merged["topic_description"] = topic_buckets[bucket]["description"]
        merged["topic_post_count"] = merged["topic_post_count"].astype(int)
        merged["total_posts_in_month_subreddit"] = merged["total_posts_in_month_subreddit"].astype(int)
        merged["topic_share_of_all_posts"] = merged.apply(
            lambda row: safe_divide(row["topic_post_count"], row["total_posts_in_month_subreddit"]),
            axis=1,
        )
        subreddit_month_rows.append(merged)

    subreddit_month_df = pd.concat(subreddit_month_rows, ignore_index=True)

    print("Subreddit-by-month topic counts computed.")
    print()

    print("Overall topic totals across the whole dataset:")
    overall_topic_totals = []
    for bucket in bucket_names:
        topic_col = f"topic_{bucket}"
        count = int(df[topic_col].sum())
        pct = safe_divide(count, len(df))
        overall_topic_totals.append({
            "topic_bucket": bucket,
            "topic_post_count": count,
            "topic_share_of_all_posts": pct,
        })

    overall_topic_totals_df = pd.DataFrame(overall_topic_totals).sort_values(
        by=["topic_post_count", "topic_bucket"],
        ascending=[False, True]
    )
    print(overall_topic_totals_df.to_string(index=False))
    print()

    print("Preview of monthly long-format topic trends:")
    print(monthly_long_df.head(12).to_string(index=False))
    print()

    print("Preview of time-period topic summary:")
    print(time_period_summary_df.head(12).to_string(index=False))
    print()

    print("Creating output folder if needed...")
    output_dir.mkdir(parents=True, exist_ok=True)

    monthly_base_output = output_dir / "monthly_post_totals.csv"
    monthly_long_output = output_dir / "topic_monthly_trends_long.csv"
    monthly_counts_wide_output = output_dir / "topic_monthly_trends_wide_counts.csv"
    monthly_share_all_output = output_dir / "topic_monthly_trends_wide_share_all_posts.csv"
    monthly_share_tagged_output = output_dir / "topic_monthly_trends_wide_share_topic_tagged_posts.csv"
    time_period_output = output_dir / "topic_time_period_summary.csv"
    subreddit_month_output = output_dir / "topic_subreddit_monthly_trends.csv"
    overall_topic_output = output_dir / "topic_overall_frequency_summary.csv"

    print("Saving Step 18 output files...")
    monthly_base_df.to_csv(monthly_base_output, index=False, encoding="utf-8")
    monthly_long_df.to_csv(monthly_long_output, index=False, encoding="utf-8")
    monthly_counts_wide.to_csv(monthly_counts_wide_output, index=False, encoding="utf-8")
    monthly_share_all_wide.to_csv(monthly_share_all_output, index=False, encoding="utf-8")
    monthly_share_topic_tagged_wide.to_csv(monthly_share_tagged_output, index=False, encoding="utf-8")
    time_period_summary_df.to_csv(time_period_output, index=False, encoding="utf-8")
    subreddit_month_df.to_csv(subreddit_month_output, index=False, encoding="utf-8")
    overall_topic_totals_df.to_csv(overall_topic_output, index=False, encoding="utf-8")
    print("Files saved successfully.")
    print()

    print("Final summary")
    print("-" * 70)
    print(f"Saved: {monthly_base_output}")
    print(f"Saved: {monthly_long_output}")
    print(f"Saved: {monthly_counts_wide_output}")
    print(f"Saved: {monthly_share_all_output}")
    print(f"Saved: {monthly_share_tagged_output}")
    print(f"Saved: {time_period_output}")
    print(f"Saved: {subreddit_month_output}")
    print(f"Saved: {overall_topic_output}")
    print()
    print("Step 18 is complete when:")
    print("1. the script finishes without errors")
    print("2. these output files are created")
    print("3. monthly topic counts and shares are saved")
    print("4. time-period topic summaries are saved")
    print("5. you can see how topic frequency changes by month in the long/wide files")
    print("=" * 70)


if __name__ == "__main__":
    main()
