"""
create_final_report_outputs.py

Step 20 of the project:
Create the final plots, tables, and written summary for the report and presentation.

What this script does:
1. Reads the Step 16 topic-tagged Reddit CSV
2. Optionally uses Step 17/18/19 outputs if available
3. Creates final summary tables
4. Creates final plots
5. Creates a written summary in TXT and Markdown
6. Saves everything into one final output folder

This script is designed to work both:
- on the current sample dataset
- later on the full real multi-year dataset

Run from src/:
    python .\create_final_report_outputs.py

Optional:
    python .\create_final_report_outputs.py ^
        --input .\data\reddit\analysis\RS_2023-02_topic_tagged.csv ^
        --output_dir .\data\reddit\final_outputs
"""

import argparse
from pathlib import Path
import sys
import math

import pandas as pd
import matplotlib.pyplot as plt

from topic_buckets import get_topic_buckets


DEFAULT_INPUT = Path("data/reddit/analysis/RS_2023-02_topic_tagged.csv")
DEFAULT_OUTPUT_DIR = Path("data/reddit/final_outputs")

REQUIRED_COLUMNS = [
    "id",
    "subreddit",
    "created_utc",
    "year_month",
    "time_period",
    "predicted_label",
    "predicted_sentiment",
    "has_any_topic",
]

TIME_PERIOD_ORDER = ["pre-COVID", "COVID", "post-COVID"]


def safe_divide(numerator, denominator):
    if denominator == 0:
        return 0.0
    return numerator / denominator


def sort_year_month_strings(values):
    cleaned = [str(v).strip() for v in values if pd.notna(v) and str(v).strip() != ""]
    if not cleaned:
        return []
    periods = pd.PeriodIndex(cleaned, freq="M")
    return [str(p) for p in periods.sort_values()]


def ensure_columns(df, columns):
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def normalize_dataframe(df, bucket_names):
    df = df.copy()

    df["predicted_label"] = pd.to_numeric(df["predicted_label"], errors="coerce")
    df["year_month"] = df["year_month"].fillna("").astype(str).str.strip()
    df["time_period"] = df["time_period"].fillna("").astype(str).str.strip()
    df["subreddit"] = df["subreddit"].fillna("").astype(str).str.strip().str.lower()
    df["predicted_sentiment"] = df["predicted_sentiment"].fillna("").astype(str).str.strip().str.lower()
    df["has_any_topic"] = pd.to_numeric(df["has_any_topic"], errors="coerce").fillna(0).astype(int)

    if "prob_positive" in df.columns:
        df["prob_positive"] = pd.to_numeric(df["prob_positive"], errors="coerce")
    else:
        df["prob_positive"] = pd.NA

    if "prob_negative" in df.columns:
        df["prob_negative"] = pd.to_numeric(df["prob_negative"], errors="coerce")
    else:
        df["prob_negative"] = pd.NA

    for bucket in bucket_names:
        topic_col = f"topic_{bucket}"
        if topic_col in df.columns:
            df[topic_col] = pd.to_numeric(df[topic_col], errors="coerce").fillna(0).astype(int)
        else:
            df[topic_col] = 0

    df = df[df["predicted_label"].isin([0, 1])].reset_index(drop=True)
    return df


def build_overall_summary(df):
    total_posts = len(df)
    positive_posts = int((df["predicted_label"] == 1).sum())
    negative_posts = int((df["predicted_label"] == 0).sum())

    row = {
        "total_posts": total_posts,
        "positive_posts": positive_posts,
        "negative_posts": negative_posts,
        "positive_rate": safe_divide(positive_posts, total_posts),
        "negative_rate": safe_divide(negative_posts, total_posts),
        "num_subreddits": int(df["subreddit"].nunique()),
        "num_months": int(df["year_month"].replace("", pd.NA).dropna().nunique()),
        "num_time_periods": int(df["time_period"].replace("", pd.NA).dropna().nunique()),
        "posts_with_any_topic": int(df["has_any_topic"].sum()),
        "share_posts_with_any_topic": safe_divide(int(df["has_any_topic"].sum()), total_posts),
    }

    if "prob_positive" in df.columns and df["prob_positive"].notna().any():
        row["avg_prob_positive"] = float(df["prob_positive"].dropna().mean())
    else:
        row["avg_prob_positive"] = pd.NA

    if "prob_negative" in df.columns and df["prob_negative"].notna().any():
        row["avg_prob_negative"] = float(df["prob_negative"].dropna().mean())
    else:
        row["avg_prob_negative"] = pd.NA

    return pd.DataFrame([row])


def build_time_period_sentiment_table(df):
    rows = []
    for period in TIME_PERIOD_ORDER:
        period_df = df[df["time_period"] == period].copy()
        total_posts = len(period_df)
        positive_posts = int((period_df["predicted_label"] == 1).sum())
        negative_posts = int((period_df["predicted_label"] == 0).sum())

        rows.append({
            "time_period": period,
            "total_posts": total_posts,
            "positive_posts": positive_posts,
            "negative_posts": negative_posts,
            "positive_rate": safe_divide(positive_posts, total_posts),
            "negative_rate": safe_divide(negative_posts, total_posts),
            "posts_with_any_topic": int(period_df["has_any_topic"].sum()),
            "share_posts_with_any_topic": safe_divide(int(period_df["has_any_topic"].sum()), total_posts),
        })

    return pd.DataFrame(rows)


def build_subreddit_sentiment_table(df):
    grouped = (
        df.groupby("subreddit")
          .agg(
              total_posts=("id", "size"),
              positive_posts=("predicted_label", lambda s: int((s == 1).sum())),
              negative_posts=("predicted_label", lambda s: int((s == 0).sum())),
          )
          .reset_index()
    )
    grouped["positive_rate"] = grouped.apply(
        lambda row: safe_divide(row["positive_posts"], row["total_posts"]), axis=1
    )
    grouped["negative_rate"] = grouped.apply(
        lambda row: safe_divide(row["negative_posts"], row["total_posts"]), axis=1
    )
    grouped = grouped.sort_values(by=["total_posts", "subreddit"], ascending=[False, True]).reset_index(drop=True)
    return grouped


def build_topic_sentiment_table(df, topic_buckets):
    overall_positive_rate = safe_divide(int((df["predicted_label"] == 1).sum()), len(df))

    rows = []
    for bucket, info in topic_buckets.items():
        topic_col = f"topic_{bucket}"
        topic_df = df[df[topic_col] == 1].copy()
        total_posts = len(topic_df)
        positive_posts = int((topic_df["predicted_label"] == 1).sum())
        negative_posts = int((topic_df["predicted_label"] == 0).sum())

        row = {
            "topic_bucket": bucket,
            "topic_description": info["description"],
            "total_posts": total_posts,
            "positive_posts": positive_posts,
            "negative_posts": negative_posts,
            "positive_rate": safe_divide(positive_posts, total_posts),
            "negative_rate": safe_divide(negative_posts, total_posts),
            "share_of_all_posts": safe_divide(total_posts, len(df)),
            "positive_rate_minus_overall": safe_divide(positive_posts, total_posts) - overall_positive_rate if total_posts > 0 else pd.NA,
        }

        if "prob_positive" in topic_df.columns and topic_df["prob_positive"].notna().any():
            row["avg_prob_positive"] = float(topic_df["prob_positive"].dropna().mean())
        else:
            row["avg_prob_positive"] = pd.NA

        rows.append(row)

    out = pd.DataFrame(rows)
    out = out.sort_values(by=["total_posts", "topic_bucket"], ascending=[False, True]).reset_index(drop=True)
    return out


def build_monthly_sentiment_table(df):
    months = sort_year_month_strings(df["year_month"].unique())
    rows = []

    for ym in months:
        month_df = df[df["year_month"] == ym].copy()
        total_posts = len(month_df)
        positive_posts = int((month_df["predicted_label"] == 1).sum())
        negative_posts = int((month_df["predicted_label"] == 0).sum())

        rows.append({
            "year_month": ym,
            "total_posts": total_posts,
            "positive_posts": positive_posts,
            "negative_posts": negative_posts,
            "positive_rate": safe_divide(positive_posts, total_posts),
            "negative_rate": safe_divide(negative_posts, total_posts),
            "posts_with_any_topic": int(month_df["has_any_topic"].sum()),
            "share_posts_with_any_topic": safe_divide(int(month_df["has_any_topic"].sum()), total_posts),
        })

    return pd.DataFrame(rows)


def build_topic_monthly_table(df, topic_buckets):
    months = sort_year_month_strings(df["year_month"].unique())
    rows = []

    for bucket, info in topic_buckets.items():
        topic_col = f"topic_{bucket}"
        for ym in months:
            month_df = df[df["year_month"] == ym].copy()
            topic_df = month_df[month_df[topic_col] == 1].copy()

            rows.append({
                "year_month": ym,
                "topic_bucket": bucket,
                "topic_description": info["description"],
                "total_posts_in_month": len(month_df),
                "topic_posts_in_month": len(topic_df),
                "topic_share_of_all_posts": safe_divide(len(topic_df), len(month_df)),
                "topic_positive_rate": safe_divide(int((topic_df["predicted_label"] == 1).sum()), len(topic_df)),
                "topic_negative_rate": safe_divide(int((topic_df["predicted_label"] == 0).sum()), len(topic_df)),
            })

    return pd.DataFrame(rows)


def build_hypothesis_overview_table(df, topic_buckets):
    rows = []

    # H1
    recruiting_df = df[df["topic_recruiting_pipeline"] == 1].copy()
    pre_df = recruiting_df[recruiting_df["time_period"] == "pre-COVID"].copy()
    post_df = recruiting_df[recruiting_df["time_period"] == "post-COVID"].copy()

    rows.append({
        "hypothesis": "H1",
        "question": "Recruiting sentiment is more negative in post-COVID than pre-COVID.",
        "status_with_current_data": (
            "ready_for_real_test" if len(pre_df) > 0 and len(post_df) > 0 else "not_fully_testable_with_current_sample"
        ),
        "current_result_note": (
            f"pre-COVID recruiting posts={len(pre_df)}, post-COVID recruiting posts={len(post_df)}"
        ),
    })

    # H2
    layoffs_df = df[df["topic_layoffs_market"] == 1].copy()
    rows.append({
        "hypothesis": "H2",
        "question": "Layoff-related posts increase after 2022 and have lower sentiment.",
        "status_with_current_data": (
            "partially_testable" if len(layoffs_df) > 0 else "not_observable_in_current_sample"
        ),
        "current_result_note": (
            f"layoffs-topic posts in current dataset={len(layoffs_df)}"
        ),
    })

    # H3
    author_counts = (
        df[df["author"].fillna("").astype(str).str.strip() != ""]
        .groupby("author")
        .size()
    )
    eligible_users = int((author_counts >= 3).sum())
    rows.append({
        "hypothesis": "H3",
        "question": "User-level sentiment differs by topic and produces cross-topic correlations.",
        "status_with_current_data": (
            "partially_testable" if eligible_users > 0 else "not_fully_testable_with_current_sample"
        ),
        "current_result_note": f"eligible users with >=3 posts={eligible_users}",
    })

    # H4
    num_months = int(df["year_month"].replace("", pd.NA).dropna().nunique())
    rows.append({
        "hypothesis": "H4",
        "question": "AI-topic prevalence over time is associated with overall sentiment shifts.",
        "status_with_current_data": (
            "ready_for_real_test" if num_months >= 2 else "not_fully_testable_with_current_sample"
        ),
        "current_result_note": f"distinct months in dataset={num_months}",
    })

    return pd.DataFrame(rows)


def save_plot_overall_sentiment_by_time_period(table_df, output_path):
    plot_df = table_df[table_df["total_posts"] > 0].copy()
    if len(plot_df) == 0:
        return False

    plt.figure(figsize=(8, 5))
    plt.bar(plot_df["time_period"], plot_df["positive_rate"])
    plt.title("Positive Sentiment Rate by Time Period")
    plt.xlabel("Time Period")
    plt.ylabel("Positive Rate")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    return True


def save_plot_sentiment_by_subreddit(table_df, output_path):
    plot_df = table_df.copy()
    if len(plot_df) == 0:
        return False

    plt.figure(figsize=(8, 5))
    plt.bar(plot_df["subreddit"], plot_df["positive_rate"])
    plt.title("Positive Sentiment Rate by Subreddit")
    plt.xlabel("Subreddit")
    plt.ylabel("Positive Rate")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    return True


def save_plot_topic_positive_rate(table_df, output_path):
    plot_df = table_df.sort_values(by="positive_rate", ascending=False).copy()
    plot_df = plot_df[plot_df["total_posts"] > 0].copy()
    if len(plot_df) == 0:
        return False

    plt.figure(figsize=(9, 5))
    plt.bar(plot_df["topic_bucket"], plot_df["positive_rate"])
    plt.title("Positive Sentiment Rate by Topic")
    plt.xlabel("Topic Bucket")
    plt.ylabel("Positive Rate")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    return True


def save_plot_topic_frequency(table_df, output_path):
    plot_df = table_df.sort_values(by="total_posts", ascending=False).copy()
    plot_df = plot_df[plot_df["total_posts"] > 0].copy()
    if len(plot_df) == 0:
        return False

    plt.figure(figsize=(9, 5))
    plt.bar(plot_df["topic_bucket"], plot_df["total_posts"])
    plt.title("Topic Frequency in Dataset")
    plt.xlabel("Topic Bucket")
    plt.ylabel("Number of Posts")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    return True


def save_plot_monthly_sentiment(monthly_df, output_path):
    plot_df = monthly_df.copy()
    if len(plot_df) == 0:
        return False

    plt.figure(figsize=(10, 5))
    plt.plot(plot_df["year_month"], plot_df["positive_rate"], marker="o")
    plt.title("Monthly Positive Sentiment Rate")
    plt.xlabel("Month")
    plt.ylabel("Positive Rate")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    return True


def save_plot_monthly_topic_trends(topic_monthly_df, output_path):
    plot_df = topic_monthly_df.copy()
    if len(plot_df) == 0:
        return False

    pivot_df = (
        plot_df.pivot(index="year_month", columns="topic_bucket", values="topic_posts_in_month")
               .fillna(0)
    )
    if pivot_df.empty:
        return False

    plt.figure(figsize=(11, 6))
    for col in pivot_df.columns:
        plt.plot(pivot_df.index, pivot_df[col], marker="o", label=col)

    plt.title("Topic Frequency Over Time")
    plt.xlabel("Month")
    plt.ylabel("Topic Post Count")
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    return True


def build_written_summary(overall_df, time_df, subreddit_df, topic_df, monthly_df, hypothesis_df):
    overall = overall_df.iloc[0].to_dict()

    summary_lines = []
    summary_lines.append("FINAL PROJECT SUMMARY")
    summary_lines.append("=" * 70)
    summary_lines.append("")
    summary_lines.append("Dataset overview")
    summary_lines.append(
        f"- Total posts analyzed: {int(overall['total_posts']):,}"
    )
    summary_lines.append(
        f"- Positive posts: {int(overall['positive_posts']):,} ({overall['positive_rate']:.2%})"
    )
    summary_lines.append(
        f"- Negative posts: {int(overall['negative_posts']):,} ({overall['negative_rate']:.2%})"
    )
    summary_lines.append(
        f"- Posts with at least one topic tag: {int(overall['posts_with_any_topic']):,} ({overall['share_posts_with_any_topic']:.2%})"
    )
    summary_lines.append(
        f"- Distinct subreddits: {int(overall['num_subreddits'])}"
    )
    summary_lines.append(
        f"- Distinct months: {int(overall['num_months'])}"
    )
    summary_lines.append(
        f"- Distinct time periods: {int(overall['num_time_periods'])}"
    )
    summary_lines.append("")

    if len(time_df[time_df["total_posts"] > 0]) > 0:
        summary_lines.append("Sentiment by time period")
        for _, row in time_df.iterrows():
            if row["total_posts"] <= 0:
                continue
            summary_lines.append(
                f"- {row['time_period']}: {int(row['total_posts']):,} posts, positive rate {row['positive_rate']:.2%}, negative rate {row['negative_rate']:.2%}"
            )
        summary_lines.append("")

    if len(subreddit_df) > 0:
        summary_lines.append("Sentiment by subreddit")
        for _, row in subreddit_df.iterrows():
            summary_lines.append(
                f"- {row['subreddit']}: {int(row['total_posts']):,} posts, positive rate {row['positive_rate']:.2%}, negative rate {row['negative_rate']:.2%}"
            )
        summary_lines.append("")

    if len(topic_df) > 0:
        summary_lines.append("Topic summary")
        top_freq = topic_df.sort_values(by="total_posts", ascending=False).head(3)
        top_pos = topic_df.sort_values(by="positive_rate", ascending=False).head(3)
        top_neg = topic_df.sort_values(by="negative_rate", ascending=False).head(3)

        summary_lines.append("Most frequent topics:")
        for _, row in top_freq.iterrows():
            summary_lines.append(
                f"- {row['topic_bucket']}: {int(row['total_posts']):,} posts ({row['share_of_all_posts']:.2%} of all posts)"
            )

        summary_lines.append("")
        summary_lines.append("Most positive topics:")
        for _, row in top_pos.iterrows():
            if row["total_posts"] > 0:
                summary_lines.append(
                    f"- {row['topic_bucket']}: positive rate {row['positive_rate']:.2%} across {int(row['total_posts']):,} posts"
                )

        summary_lines.append("")
        summary_lines.append("Most negative topics:")
        for _, row in top_neg.iterrows():
            if row["total_posts"] > 0:
                summary_lines.append(
                    f"- {row['topic_bucket']}: negative rate {row['negative_rate']:.2%} across {int(row['total_posts']):,} posts"
                )
        summary_lines.append("")

    if len(monthly_df) > 0:
        summary_lines.append("Monthly trend overview")
        for _, row in monthly_df.iterrows():
            summary_lines.append(
                f"- {row['year_month']}: {int(row['total_posts']):,} posts, positive rate {row['positive_rate']:.2%}, topic-tag share {row['share_posts_with_any_topic']:.2%}"
            )
        summary_lines.append("")

    if len(hypothesis_df) > 0:
        summary_lines.append("Hypothesis-testing readiness")
        for _, row in hypothesis_df.iterrows():
            summary_lines.append(
                f"- {row['hypothesis']}: {row['status_with_current_data']} ({row['current_result_note']})"
            )
        summary_lines.append("")

    if int(overall["num_months"]) < 2:
        summary_lines.append("Important note")
        summary_lines.append(
            "- This current run is based on a sample month, so time-based comparisons and multi-era hypothesis testing are not yet final."
        )
        summary_lines.append(
            "- When the full dataset is available, rerun the same pipeline to generate the final report and presentation outputs."
        )
        summary_lines.append("")

    return "\n".join(summary_lines)


def build_markdown_summary(overall_df, time_df, subreddit_df, topic_df, monthly_df, hypothesis_df):
    overall = overall_df.iloc[0].to_dict()

    lines = []
    lines.append("# Final Project Summary")
    lines.append("")
    lines.append("## Dataset overview")
    lines.append(f"- Total posts analyzed: {int(overall['total_posts']):,}")
    lines.append(f"- Positive posts: {int(overall['positive_posts']):,} ({overall['positive_rate']:.2%})")
    lines.append(f"- Negative posts: {int(overall['negative_posts']):,} ({overall['negative_rate']:.2%})")
    lines.append(f"- Posts with at least one topic tag: {int(overall['posts_with_any_topic']):,} ({overall['share_posts_with_any_topic']:.2%})")
    lines.append(f"- Distinct subreddits: {int(overall['num_subreddits'])}")
    lines.append(f"- Distinct months: {int(overall['num_months'])}")
    lines.append(f"- Distinct time periods: {int(overall['num_time_periods'])}")
    lines.append("")

    lines.append("## Sentiment by time period")
    for _, row in time_df.iterrows():
        if row["total_posts"] <= 0:
            continue
        lines.append(f"- **{row['time_period']}**: {int(row['total_posts']):,} posts, positive rate {row['positive_rate']:.2%}, negative rate {row['negative_rate']:.2%}")
    lines.append("")

    lines.append("## Sentiment by subreddit")
    for _, row in subreddit_df.iterrows():
        lines.append(f"- **{row['subreddit']}**: {int(row['total_posts']):,} posts, positive rate {row['positive_rate']:.2%}, negative rate {row['negative_rate']:.2%}")
    lines.append("")

    lines.append("## Topic summary")
    for _, row in topic_df.sort_values(by='total_posts', ascending=False).iterrows():
        lines.append(f"- **{row['topic_bucket']}**: {int(row['total_posts']):,} posts, positive rate {row['positive_rate']:.2%}, negative rate {row['negative_rate']:.2%}")
    lines.append("")

    lines.append("## Hypothesis-testing readiness")
    for _, row in hypothesis_df.iterrows():
        lines.append(f"- **{row['hypothesis']}**: {row['status_with_current_data']} ({row['current_result_note']})")
    lines.append("")

    if int(overall["num_months"]) < 2:
        lines.append("## Important note")
        lines.append("- This run uses a sample dataset, so the final comparative conclusions must be generated again on the full real dataset.")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Step 20: create final plots, tables, and written summary for report/presentation."
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
        help="Directory where final outputs will be saved",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    print("=" * 70)
    print("STEP 20: CREATE FINAL PLOTS, TABLES, AND WRITTEN SUMMARY")
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
    print(f"Topic buckets loaded: {bucket_names}")
    print()

    print("Checking required columns...")
    required_topic_columns = [f'topic_{bucket}' for bucket in bucket_names]
    ensure_columns(df, REQUIRED_COLUMNS + required_topic_columns)
    print("All required columns are present.")
    print()

    print("Normalizing analysis dataframe...")
    df = normalize_dataframe(df, bucket_names)
    print(f"Rows remaining after normalization: {len(df):,}")
    print()

    if len(df) == 0:
        print("ERROR: No usable rows remain after normalization.")
        sys.exit(1)

    print("Creating output folders...")
    tables_dir = output_dir / "tables"
    plots_dir = output_dir / "plots"
    summaries_dir = output_dir / "written_summary"

    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    summaries_dir.mkdir(parents=True, exist_ok=True)
    print("Folders created.")
    print()

    print("Building final tables...")
    overall_summary_df = build_overall_summary(df)
    time_period_sentiment_df = build_time_period_sentiment_table(df)
    subreddit_sentiment_df = build_subreddit_sentiment_table(df)
    topic_sentiment_df = build_topic_sentiment_table(df, topic_buckets)
    monthly_sentiment_df = build_monthly_sentiment_table(df)
    topic_monthly_df = build_topic_monthly_table(df, topic_buckets)
    hypothesis_overview_df = build_hypothesis_overview_table(df, topic_buckets)
    print("Final tables created.")
    print()

    print("Saving final tables...")
    overall_summary_path = tables_dir / "overall_summary.csv"
    time_period_sentiment_path = tables_dir / "time_period_sentiment_table.csv"
    subreddit_sentiment_path = tables_dir / "subreddit_sentiment_table.csv"
    topic_sentiment_path = tables_dir / "topic_sentiment_table.csv"
    monthly_sentiment_path = tables_dir / "monthly_sentiment_table.csv"
    topic_monthly_path = tables_dir / "topic_monthly_table.csv"
    hypothesis_overview_path = tables_dir / "hypothesis_overview_table.csv"

    overall_summary_df.to_csv(overall_summary_path, index=False, encoding="utf-8")
    time_period_sentiment_df.to_csv(time_period_sentiment_path, index=False, encoding="utf-8")
    subreddit_sentiment_df.to_csv(subreddit_sentiment_path, index=False, encoding="utf-8")
    topic_sentiment_df.to_csv(topic_sentiment_path, index=False, encoding="utf-8")
    monthly_sentiment_df.to_csv(monthly_sentiment_path, index=False, encoding="utf-8")
    topic_monthly_df.to_csv(topic_monthly_path, index=False, encoding="utf-8")
    hypothesis_overview_df.to_csv(hypothesis_overview_path, index=False, encoding="utf-8")
    print("Tables saved successfully.")
    print()

    print("Creating final plots...")
    created_plots = []

    if save_plot_overall_sentiment_by_time_period(
        time_period_sentiment_df,
        plots_dir / "plot_positive_sentiment_by_time_period.png"
    ):
        created_plots.append("plot_positive_sentiment_by_time_period.png")

    if save_plot_sentiment_by_subreddit(
        subreddit_sentiment_df,
        plots_dir / "plot_positive_sentiment_by_subreddit.png"
    ):
        created_plots.append("plot_positive_sentiment_by_subreddit.png")

    if save_plot_topic_positive_rate(
        topic_sentiment_df,
        plots_dir / "plot_positive_sentiment_by_topic.png"
    ):
        created_plots.append("plot_positive_sentiment_by_topic.png")

    if save_plot_topic_frequency(
        topic_sentiment_df,
        plots_dir / "plot_topic_frequency.png"
    ):
        created_plots.append("plot_topic_frequency.png")

    if save_plot_monthly_sentiment(
        monthly_sentiment_df,
        plots_dir / "plot_monthly_positive_sentiment.png"
    ):
        created_plots.append("plot_monthly_positive_sentiment.png")

    if save_plot_monthly_topic_trends(
        topic_monthly_df,
        plots_dir / "plot_monthly_topic_trends.png"
    ):
        created_plots.append("plot_monthly_topic_trends.png")

    print(f"Plots created: {len(created_plots)}")
    for name in created_plots:
        print(f"  - {name}")
    print()

    print("Creating written summaries...")
    txt_summary = build_written_summary(
        overall_summary_df,
        time_period_sentiment_df,
        subreddit_sentiment_df,
        topic_sentiment_df,
        monthly_sentiment_df,
        hypothesis_overview_df,
    )

    md_summary = build_markdown_summary(
        overall_summary_df,
        time_period_sentiment_df,
        subreddit_sentiment_df,
        topic_sentiment_df,
        monthly_sentiment_df,
        hypothesis_overview_df,
    )

    txt_summary_path = summaries_dir / "final_summary.txt"
    md_summary_path = summaries_dir / "final_summary.md"

    with open(txt_summary_path, "w", encoding="utf-8") as f:
        f.write(txt_summary)

    with open(md_summary_path, "w", encoding="utf-8") as f:
        f.write(md_summary)

    print("Written summaries created.")
    print()

    print("Preview of overall summary:")
    print(overall_summary_df.to_string(index=False))
    print()

    print("Preview of topic sentiment table:")
    print(topic_sentiment_df.head(10).to_string(index=False))
    print()

    print("Final summary")
    print("-" * 70)
    print(f"Tables folder:   {tables_dir}")
    print(f"Plots folder:    {plots_dir}")
    print(f"Summary folder:  {summaries_dir}")
    print()
    print("Saved table files:")
    print(f"  - {overall_summary_path}")
    print(f"  - {time_period_sentiment_path}")
    print(f"  - {subreddit_sentiment_path}")
    print(f"  - {topic_sentiment_path}")
    print(f"  - {monthly_sentiment_path}")
    print(f"  - {topic_monthly_path}")
    print(f"  - {hypothesis_overview_path}")
    print()
    print("Saved summary files:")
    print(f"  - {txt_summary_path}")
    print(f"  - {md_summary_path}")
    print()
    print("Step 20 is complete when:")
    print("1. the script finishes without errors")
    print("2. the tables, plots, and written_summary folders are created")
    print("3. the final CSV summary tables are saved")
    print("4. the PNG plots are saved")
    print("5. final_summary.txt and final_summary.md are saved")
    print("=" * 70)


if __name__ == "__main__":
    main()