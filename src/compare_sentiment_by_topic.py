"""
compare_sentiment_by_topic.py

Step 17 of the project:
Compare sentiment across different topics to see which themes are associated
with more negative or more positive sentiment.

What this script does:
1. Reads the Step 16 topic-tagged Reddit CSV
2. Uses the Step 15 topic bucket definitions
3. For each topic bucket, computes:
   - number of posts
   - number of positive posts
   - number of negative posts
   - positive rate
   - negative rate
   - average predicted label
   - average positive probability
   - average negative probability
   - difference from the overall positive rate
4. Saves summary CSV files
5. Prints ranked summaries so we can quickly see which topics are more positive
   or more negative

Run from src/:
    python .\compare_sentiment_by_topic.py

Optional:
    python .\compare_sentiment_by_topic.py ^
        --input .\data\reddit\analysis\RS_2023-02_topic_tagged.csv ^
        --output_dir .\data\reddit\analysis\step17_topic_sentiment
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from topic_buckets import get_topic_buckets


DEFAULT_INPUT = Path("data/reddit/analysis/RS_2023-02_topic_tagged.csv")
DEFAULT_OUTPUT_DIR = Path("data/reddit/analysis/step17_topic_sentiment")

REQUIRED_BASE_COLUMNS = [
    "id",
    "subreddit",
    "year_month",
    "time_period",
    "predicted_label",
    "predicted_sentiment",
    "has_any_topic",
]


def main():
    parser = argparse.ArgumentParser(
        description="Step 17: compare sentiment across topic buckets."
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
        help="Directory where Step 17 outputs will be saved",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    print("=" * 70)
    print("STEP 17: COMPARE SENTIMENT ACROSS TOPICS")
    print("=" * 70)
    print(f"Input file:   {input_path}")
    print(f"Output dir:   {output_dir}")
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

    print("Normalizing prediction columns...")
    df["predicted_label"] = pd.to_numeric(df["predicted_label"], errors="coerce")
    df["predicted_sentiment"] = df["predicted_sentiment"].fillna("").astype(str).str.strip().str.lower()

    if "prob_positive" in df.columns:
        df["prob_positive"] = pd.to_numeric(df["prob_positive"], errors="coerce")
    else:
        df["prob_positive"] = pd.NA

    if "prob_negative" in df.columns:
        df["prob_negative"] = pd.to_numeric(df["prob_negative"], errors="coerce")
    else:
        df["prob_negative"] = pd.NA

    invalid_labels = int(df["predicted_label"].isna().sum())
    print(f"Rows with invalid predicted_label: {invalid_labels:,}")

    before_drop = len(df)
    df = df[df["predicted_label"].isin([0, 1])].reset_index(drop=True)
    dropped = before_drop - len(df)
    print(f"Rows dropped due to invalid predicted_label: {dropped:,}")
    print(f"Rows remaining for analysis: {len(df):,}")
    print()

    if len(df) == 0:
        print("ERROR: No rows remain after validating predicted_label.")
        sys.exit(1)

    print("Computing overall sentiment baseline...")
    overall_total = len(df)
    overall_positive = int((df["predicted_label"] == 1).sum())
    overall_negative = int((df["predicted_label"] == 0).sum())
    overall_positive_rate = overall_positive / overall_total
    overall_negative_rate = overall_negative / overall_total

    print(f"Overall total posts:    {overall_total:,}")
    print(f"Overall positive posts: {overall_positive:,}")
    print(f"Overall negative posts: {overall_negative:,}")
    print(f"Overall positive rate:  {overall_positive_rate:.4f}")
    print(f"Overall negative rate:  {overall_negative_rate:.4f}")
    print()

    print("Computing sentiment summary for each topic...")
    summary_rows = []

    for bucket in bucket_names:
        topic_col = f"topic_{bucket}"
        topic_df = df[df[topic_col] == 1].copy()

        total_posts = len(topic_df)
        positive_posts = int((topic_df["predicted_label"] == 1).sum())
        negative_posts = int((topic_df["predicted_label"] == 0).sum())

        if total_posts > 0:
            positive_rate = positive_posts / total_posts
            negative_rate = negative_posts / total_posts
            avg_predicted_label = float(topic_df["predicted_label"].mean())

            avg_prob_positive = (
                float(topic_df["prob_positive"].dropna().mean())
                if topic_df["prob_positive"].notna().any()
                else pd.NA
            )
            avg_prob_negative = (
                float(topic_df["prob_negative"].dropna().mean())
                if topic_df["prob_negative"].notna().any()
                else pd.NA
            )
        else:
            positive_rate = 0.0
            negative_rate = 0.0
            avg_predicted_label = pd.NA
            avg_prob_positive = pd.NA
            avg_prob_negative = pd.NA

        summary_rows.append({
            "topic_bucket": bucket,
            "topic_description": topic_buckets[bucket]["description"],
            "total_posts": total_posts,
            "positive_posts": positive_posts,
            "negative_posts": negative_posts,
            "positive_rate": positive_rate,
            "negative_rate": negative_rate,
            "avg_predicted_label": avg_predicted_label,
            "avg_prob_positive": avg_prob_positive,
            "avg_prob_negative": avg_prob_negative,
            "overall_positive_rate": overall_positive_rate,
            "overall_negative_rate": overall_negative_rate,
            "positive_rate_minus_overall": positive_rate - overall_positive_rate,
            "negative_rate_minus_overall": negative_rate - overall_negative_rate,
        })

    summary_df = pd.DataFrame(summary_rows)

    print("Ranking topics from most positive to least positive...")
    positive_rank_df = summary_df.sort_values(
        by=["positive_rate", "total_posts"],
        ascending=[False, False]
    ).reset_index(drop=True)
    positive_rank_df.insert(0, "positive_rank", range(1, len(positive_rank_df) + 1))

    print(positive_rank_df[
        ["positive_rank", "topic_bucket", "total_posts", "positive_rate", "negative_rate", "positive_rate_minus_overall"]
    ].to_string(index=False))
    print()

    print("Ranking topics from most negative to least negative...")
    negative_rank_df = summary_df.sort_values(
        by=["negative_rate", "total_posts"],
        ascending=[False, False]
    ).reset_index(drop=True)
    negative_rank_df.insert(0, "negative_rank", range(1, len(negative_rank_df) + 1))

    print(negative_rank_df[
        ["negative_rank", "topic_bucket", "total_posts", "negative_rate", "positive_rate", "negative_rate_minus_overall"]
    ].to_string(index=False))
    print()

    print("Creating a long-format topic-post table for future analysis...")
    long_rows = []
    for bucket in bucket_names:
        topic_col = f"topic_{bucket}"
        topic_df = df[df[topic_col] == 1].copy()
        if len(topic_df) == 0:
            continue

        topic_df["topic_bucket"] = bucket
        topic_df["topic_description"] = topic_buckets[bucket]["description"]

        keep_cols = [
            "id",
            "subreddit",
            "year_month",
            "time_period",
            "predicted_label",
            "predicted_sentiment",
            "prob_positive",
            "prob_negative",
            "topic_bucket",
            "topic_description",
        ]
        long_rows.append(topic_df[keep_cols])

    if long_rows:
        long_df = pd.concat(long_rows, ignore_index=True)
    else:
        long_df = pd.DataFrame(columns=[
            "id",
            "subreddit",
            "year_month",
            "time_period",
            "predicted_label",
            "predicted_sentiment",
            "prob_positive",
            "prob_negative",
            "topic_bucket",
            "topic_description",
        ])

    print(f"Long-format topic-post rows: {len(long_df):,}")
    print()

    print("Creating output folder if needed...")
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_output = output_dir / "topic_sentiment_summary.csv"
    positive_rank_output = output_dir / "topic_sentiment_ranked_positive.csv"
    negative_rank_output = output_dir / "topic_sentiment_ranked_negative.csv"
    long_output = output_dir / "topic_post_long_format.csv"

    print("Saving Step 17 outputs...")
    summary_df.to_csv(summary_output, index=False, encoding="utf-8")
    positive_rank_df.to_csv(positive_rank_output, index=False, encoding="utf-8")
    negative_rank_df.to_csv(negative_rank_output, index=False, encoding="utf-8")
    long_df.to_csv(long_output, index=False, encoding="utf-8")
    print("Files saved successfully.")
    print()

    print("Final summary")
    print("-" * 70)
    print(f"Main summary file:          {summary_output}")
    print(f"Positive ranking file:      {positive_rank_output}")
    print(f"Negative ranking file:      {negative_rank_output}")
    print(f"Long-format topic file:     {long_output}")
    print()
    print("Step 17 is complete when:")
    print("1. the script finishes without errors")
    print("2. the four output files are created")
    print("3. the summary files show per-topic positive and negative rates")
    print("4. the rankings show which topics are relatively more positive or negative")
    print("=" * 70)


if __name__ == "__main__":
    main()