"""
tag_reddit_topics.py

Step 16 of the project:
Label or tag Reddit posts into the topic buckets using keyword-based rules.

Run:
    python src/analysis/tag_reddit_topics.py
"""

import argparse
import re
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC_DIR))

import pandas as pd

from utils.preprocessing import clean_text
from utils.topic_buckets import get_topic_buckets

DEFAULT_INPUT = SRC_DIR / "data" / "reddit" / "analysis" / "all_subreddits_time_grouped.csv"
DEFAULT_OUTPUT = SRC_DIR / "data" / "reddit" / "analysis" / "all_subreddits_topic_tagged.csv"

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
    "created_datetime_utc",
    "year",
    "month",
    "year_month",
    "time_period",
]


def normalize_text(value) -> str:
    """Convert missing values to empty string and strip whitespace."""
    if pd.isna(value):
        return ""
    return str(value).strip()


def build_keyword_pattern(cleaned_keyword: str):
    """
    Build a regex that matches a cleaned keyword or phrase as a whole token/phrase.
    """
    if not cleaned_keyword:
        return None

    parts = cleaned_keyword.split()
    if not parts:
        return None

    escaped_parts = [re.escape(part) for part in parts]
    phrase_pattern = r"\s+".join(escaped_parts)
    full_pattern = rf"(?<![a-z]){phrase_pattern}(?![a-z])"
    return re.compile(full_pattern)


def prepare_topic_patterns(topic_buckets):
    """
    Convert raw keyword lists into cleaned keywords + compiled regex patterns.
    """
    prepared = {}

    for bucket_name, info in topic_buckets.items():
        prepared[bucket_name] = []
        seen_cleaned_keywords = set()

        for raw_keyword in info["keywords"]:
            cleaned_keyword = clean_text(raw_keyword)
            if not cleaned_keyword:
                continue
            if cleaned_keyword in seen_cleaned_keywords:
                continue

            pattern = build_keyword_pattern(cleaned_keyword)
            if pattern is None:
                continue

            prepared[bucket_name].append({
                "raw_keyword": raw_keyword,
                "cleaned_keyword": cleaned_keyword,
                "pattern": pattern,
            })
            seen_cleaned_keywords.add(cleaned_keyword)

    return prepared


def find_matches_in_text(text: str, bucket_keyword_info):
    """
    Return a sorted list of raw keywords whose cleaned patterns match the text.
    """
    matches = []

    for item in bucket_keyword_info:
        if item["pattern"].search(text):
            matches.append(item["raw_keyword"])

    return sorted(set(matches), key=str.lower)


def main():
    parser = argparse.ArgumentParser(
        description="Step 16: tag Reddit posts into topic buckets using keyword-based rules."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(DEFAULT_INPUT),
        help="Path to Step 14 time-grouped CSV",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help="Path to output CSV with topic tags",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    print("=" * 70)
    print("STEP 16: TAG REDDIT POSTS INTO TOPIC BUCKETS")
    print("=" * 70)
    print(f"Input file:  {input_path}")
    print(f"Output file: {output_path}")
    print()

    if not input_path.exists():
        print(f"ERROR: Input file does not exist: {input_path}")
        sys.exit(1)

    print("Reading Step 14 CSV...")
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

    print("Normalizing cleaned_text...")
    df["cleaned_text"] = df["cleaned_text"].fillna("").astype(str).apply(normalize_text)
    empty_cleaned_text = int((df["cleaned_text"].str.len() == 0).sum())
    print(f"Rows with empty cleaned_text: {empty_cleaned_text:,}")
    print()

    print("Loading topic bucket definitions from Step 15...")
    topic_buckets = get_topic_buckets()
    print(f"Total topic buckets loaded: {len(topic_buckets)}")
    print()

    print("Preparing keyword patterns...")
    prepared_patterns = prepare_topic_patterns(topic_buckets)
    for bucket_name, items in prepared_patterns.items():
        print(f"  {bucket_name}: {len(items)} usable keyword patterns")
    print()

    print("Applying topic-tag rules...")
    topic_binary_columns = []
    matched_keyword_columns = []

    for bucket_name in topic_buckets.keys():
        topic_col = f"topic_{bucket_name}"
        match_col = f"matched_keywords_{bucket_name}"
        topic_binary_columns.append(topic_col)
        matched_keyword_columns.append(match_col)

        df[topic_col] = 0
        df[match_col] = ""

    for idx, text in enumerate(df["cleaned_text"]):
        if idx > 0 and idx % 5000 == 0:
            print(f"  Progress: tagged {idx:,} rows")

        matched_topics_for_row = []

        for bucket_name, bucket_info in prepared_patterns.items():
            topic_col = f"topic_{bucket_name}"
            match_col = f"matched_keywords_{bucket_name}"

            matches = find_matches_in_text(text, bucket_info)
            if matches:
                df.at[idx, topic_col] = 1
                df.at[idx, match_col] = "; ".join(matches)
                matched_topics_for_row.append(bucket_name)

        df.at[idx, "topic_list"] = "; ".join(matched_topics_for_row)
        df.at[idx, "num_topics"] = len(matched_topics_for_row)
        df.at[idx, "has_any_topic"] = 1 if matched_topics_for_row else 0

    print("Topic tagging complete.")
    print()

    print("Topic coverage summary:")
    for bucket_name in topic_buckets.keys():
        topic_col = f"topic_{bucket_name}"
        count = int(df[topic_col].sum())
        pct = (count / len(df)) * 100 if len(df) > 0 else 0.0
        print(f"  {topic_col}: {count:,} rows ({pct:.2f}%)")
    print()

    print("Rows with at least one topic:")
    has_any_topic_count = int(df["has_any_topic"].sum())
    has_any_topic_pct = (has_any_topic_count / len(df)) * 100 if len(df) > 0 else 0.0
    print(f"  {has_any_topic_count:,} rows ({has_any_topic_pct:.2f}%)")
    print()

    print("Distribution of number of topics per post:")
    num_topics_dist = df["num_topics"].value_counts().sort_index()
    print(num_topics_dist.to_string())
    print()

    print("Topic counts by subreddit:")
    summary_cols = [f"topic_{bucket_name}" for bucket_name in topic_buckets.keys()]
    subreddit_topic_summary = df.groupby("subreddit")[summary_cols].sum()
    print(subreddit_topic_summary.to_string())
    print()

    print("Preview of Step 16 output:")
    preview_cols = [
        "id",
        "subreddit",
        "predicted_sentiment",
        "year_month",
        "topic_list",
        "num_topics",
        "has_any_topic",
    ] + summary_cols[:3]
    print(df[preview_cols].head(10).to_string(index=False))
    print()

    print("Creating output folder if needed...")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Saving Step 16 CSV...")
    df.to_csv(output_path, index=False, encoding="utf-8")
    print("File saved successfully.")
    print()

    print("Final summary")
    print("-" * 70)
    print(f"Rows written:    {len(df):,}")
    print(f"Columns written: {len(df.columns):,}")
    print("New topic columns added:")
    for col in ["topic_list", "num_topics", "has_any_topic"] + topic_binary_columns + matched_keyword_columns:
        print(f"  - {col}")
    print()
    print(f"Saved output file to: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
