"""
test_project_hypotheses.py

Step 19 of the project:
Test the project hypotheses one by one using the sentiment results and topic trends.

Hypotheses from the project:
H1: Recruiting sentiment becomes more negative in the post-COVID era (2022+),
    compared to pre-COVID.
H2: Posts mentioning layoffs or hiring freezes increase after 2022 and co-occur
    with lower sentiment.
H3: User-level sentiment differs by topic, producing measurable cross-topic
    correlations.
H4: Increases in AI-related discussion are associated with shifts in overall
    job-market sentiment over time.

What this script does:
1. Reads the Step 16 topic-tagged Reddit CSV
2. Uses the Step 15 topic bucket definitions
3. Tests H1, H2, H3, and H4
4. Saves multiple CSV outputs plus a human-readable text report

Run from src/:
    python .\test_project_hypotheses.py

Optional:
    python .\test_project_hypotheses.py ^
        --input .\data\reddit\analysis\RS_2023-02_topic_tagged.csv ^
        --output_dir .\data\reddit\analysis\step19_hypothesis_tests ^
        --min_user_posts 3
"""

import argparse
import math
import sys
from pathlib import Path

import pandas as pd

from topic_buckets import get_topic_buckets


DEFAULT_INPUT = Path("data/reddit/analysis/RS_2023-02_topic_tagged.csv")
DEFAULT_OUTPUT_DIR = Path("data/reddit/analysis/step19_hypothesis_tests")

REQUIRED_BASE_COLUMNS = [
    "id",
    "subreddit",
    "author",
    "year_month",
    "time_period",
    "predicted_label",
    "predicted_sentiment",
    "has_any_topic",
]

TIME_PERIOD_ORDER = ["pre-COVID", "COVID", "post-COVID"]


def safe_divide(numerator, denominator):
    """Avoid division-by-zero errors."""
    if denominator == 0:
        return 0.0
    return numerator / denominator


def normal_cdf(x):
    """Standard normal CDF using math.erf, so we do not need scipy."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def two_proportion_z_test(success_a, total_a, success_b, total_b):
    """
    Two-sided two-proportion z-test for proportions.

    Returns a dictionary with:
    - rate_a
    - rate_b
    - diff_a_minus_b
    - z_stat
    - p_value_two_sided

    If the test cannot be computed, values are set to pd.NA.
    """
    if total_a <= 0 or total_b <= 0:
        return {
            "rate_a": pd.NA,
            "rate_b": pd.NA,
            "diff_a_minus_b": pd.NA,
            "z_stat": pd.NA,
            "p_value_two_sided": pd.NA,
        }

    rate_a = success_a / total_a
    rate_b = success_b / total_b

    pooled = (success_a + success_b) / (total_a + total_b)
    variance = pooled * (1 - pooled) * ((1 / total_a) + (1 / total_b))

    if variance <= 0:
        return {
            "rate_a": rate_a,
            "rate_b": rate_b,
            "diff_a_minus_b": rate_a - rate_b,
            "z_stat": pd.NA,
            "p_value_two_sided": pd.NA,
        }

    z_stat = (rate_a - rate_b) / math.sqrt(variance)
    p_value = 2 * (1 - normal_cdf(abs(z_stat)))

    return {
        "rate_a": rate_a,
        "rate_b": rate_b,
        "diff_a_minus_b": rate_a - rate_b,
        "z_stat": z_stat,
        "p_value_two_sided": p_value,
    }


def normalize_author(value):
    """Normalize author names and drop unusable placeholders."""
    if pd.isna(value):
        return ""
    text = str(value).strip().lower()
    if text in {"", "[deleted]", "[removed]", "automoderator", "nan", "none"}:
        return ""
    return text


def summarize_sentiment(df_subset, group_name):
    """Create a standard sentiment summary row for any subset of posts."""
    total_posts = len(df_subset)
    positive_posts = int((df_subset["predicted_label"] == 1).sum())
    negative_posts = int((df_subset["predicted_label"] == 0).sum())

    row = {
        "group_name": group_name,
        "total_posts": total_posts,
        "positive_posts": positive_posts,
        "negative_posts": negative_posts,
        "positive_rate": safe_divide(positive_posts, total_posts),
        "negative_rate": safe_divide(negative_posts, total_posts),
        "avg_predicted_label": (
            float(df_subset["predicted_label"].mean()) if total_posts > 0 else pd.NA
        ),
        "avg_prob_positive": (
            float(df_subset["prob_positive"].dropna().mean())
            if total_posts > 0 and "prob_positive" in df_subset.columns and df_subset["prob_positive"].notna().any()
            else pd.NA
        ),
        "avg_prob_negative": (
            float(df_subset["prob_negative"].dropna().mean())
            if total_posts > 0 and "prob_negative" in df_subset.columns and df_subset["prob_negative"].notna().any()
            else pd.NA
        ),
    }
    return row


def sort_year_month_strings(values):
    """Sort YYYY-MM strings chronologically."""
    cleaned = [str(v).strip() for v in values if pd.notna(v) and str(v).strip() != ""]
    if not cleaned:
        return []
    periods = pd.PeriodIndex(cleaned, freq="M")
    return [str(p) for p in periods.sort_values()]


def main():
    parser = argparse.ArgumentParser(
        description="Step 19: test project hypotheses using sentiment, topics, and time."
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
        help="Directory where Step 19 outputs will be saved",
    )
    parser.add_argument(
        "--min_user_posts",
        type=int,
        default=3,
        help="Minimum total posts per user for H3 user-level analysis",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    min_user_posts = int(args.min_user_posts)

    print("=" * 70)
    print("STEP 19: TEST PROJECT HYPOTHESES")
    print("=" * 70)
    print(f"Input file:      {input_path}")
    print(f"Output dir:      {output_dir}")
    print(f"Min user posts:  {min_user_posts}")
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
    required_topic_columns = [f"topic_{bucket}" for bucket in bucket_names]
    required_columns = REQUIRED_BASE_COLUMNS + required_topic_columns

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        print("ERROR: The input file is missing required columns:")
        for col in missing:
            print(f"  - {col}")
        sys.exit(1)

    print("All required columns are present.")
    print()

    print("Normalizing analysis columns...")
    df["predicted_label"] = pd.to_numeric(df["predicted_label"], errors="coerce")
    df["year_month"] = df["year_month"].fillna("").astype(str).str.strip()
    df["time_period"] = df["time_period"].fillna("").astype(str).str.strip()
    df["subreddit"] = df["subreddit"].fillna("").astype(str).str.strip().str.lower()
    df["author_norm"] = df["author"].apply(normalize_author)

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
        df[topic_col] = pd.to_numeric(df[topic_col], errors="coerce").fillna(0).astype(int)

    before_drop = len(df)
    df = df[df["predicted_label"].isin([0, 1])].reset_index(drop=True)
    dropped_invalid_label = before_drop - len(df)

    print(f"Rows dropped for invalid predicted_label: {dropped_invalid_label:,}")
    print(f"Rows available for hypothesis testing:    {len(df):,}")
    print()

    if len(df) == 0:
        print("ERROR: No usable rows remain after validating predicted_label.")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    report_lines = []
    report_lines.append("STEP 19: PROJECT HYPOTHESIS TEST REPORT")
    report_lines.append("=" * 70)
    report_lines.append(f"Input file: {input_path}")
    report_lines.append(f"Rows analyzed: {len(df):,}")
    report_lines.append(f"Minimum user posts for H3: {min_user_posts}")
    report_lines.append("")

    # ==============================================================
    # H1
    # Recruiting sentiment becomes more negative post-COVID vs pre-COVID
    # ==============================================================
    print("Testing H1...")
    recruiting_col = "topic_recruiting_pipeline"

    h1_df = df[
        (df[recruiting_col] == 1) &
        (df["time_period"].isin(["pre-COVID", "post-COVID"]))
    ].copy()

    h1_period_summary_rows = []
    for period_name in ["pre-COVID", "post-COVID"]:
        period_df = h1_df[h1_df["time_period"] == period_name].copy()
        row = summarize_sentiment(period_df, period_name)
        row["time_period"] = period_name
        h1_period_summary_rows.append(row)

    h1_period_summary_df = pd.DataFrame(h1_period_summary_rows)

    pre_df = h1_df[h1_df["time_period"] == "pre-COVID"].copy()
    post_df = h1_df[h1_df["time_period"] == "post-COVID"].copy()

    pre_total = len(pre_df)
    post_total = len(post_df)
    pre_positive = int((pre_df["predicted_label"] == 1).sum())
    post_positive = int((post_df["predicted_label"] == 1).sum())

    h1_pos_test = two_proportion_z_test(
        success_a=post_positive,
        total_a=post_total,
        success_b=pre_positive,
        total_b=pre_total,
    )

    pre_negative = int((pre_df["predicted_label"] == 0).sum())
    post_negative = int((post_df["predicted_label"] == 0).sum())

    h1_neg_test = two_proportion_z_test(
        success_a=post_negative,
        total_a=post_total,
        success_b=pre_negative,
        total_b=pre_total,
    )

    h1_test_df = pd.DataFrame([{
        "hypothesis": "H1",
        "topic_bucket": "recruiting_pipeline",
        "pre_total_posts": pre_total,
        "post_total_posts": post_total,
        "pre_positive_rate": safe_divide(pre_positive, pre_total),
        "post_positive_rate": safe_divide(post_positive, post_total),
        "post_minus_pre_positive_rate": h1_pos_test["diff_a_minus_b"],
        "positive_rate_z_stat": h1_pos_test["z_stat"],
        "positive_rate_p_value_two_sided": h1_pos_test["p_value_two_sided"],
        "pre_negative_rate": safe_divide(pre_negative, pre_total),
        "post_negative_rate": safe_divide(post_negative, post_total),
        "post_minus_pre_negative_rate": h1_neg_test["diff_a_minus_b"],
        "negative_rate_z_stat": h1_neg_test["z_stat"],
        "negative_rate_p_value_two_sided": h1_neg_test["p_value_two_sided"],
        "supports_h1_directionally": (
            safe_divide(post_positive, post_total) < safe_divide(pre_positive, pre_total)
            and safe_divide(post_negative, post_total) > safe_divide(pre_negative, pre_total)
            if pre_total > 0 and post_total > 0 else False
        ),
        "note": (
            "H1 direction means post-COVID recruiting sentiment should be more negative than pre-COVID."
            if pre_total > 0 and post_total > 0
            else "Insufficient pre-COVID or post-COVID recruiting data for direct comparison."
        ),
    }])

    report_lines.append("H1: Recruiting sentiment post-COVID vs pre-COVID")
    report_lines.append(h1_test_df.to_string(index=False))
    report_lines.append("")

    # ==============================================================
    # H2
    # Layoffs/hiring freeze posts increase after 2022 and have lower sentiment
    # ==============================================================
    print("Testing H2...")
    layoffs_col = "topic_layoffs_market"

    # H2A: frequency by time period
    h2_frequency_rows = []
    for period_name in TIME_PERIOD_ORDER:
        period_df = df[df["time_period"] == period_name].copy()
        total_posts = len(period_df)
        layoffs_posts = int(period_df[layoffs_col].sum())

        h2_frequency_rows.append({
            "time_period": period_name,
            "total_posts": total_posts,
            "layoffs_posts": layoffs_posts,
            "layoffs_share_of_all_posts": safe_divide(layoffs_posts, total_posts),
        })

    h2_frequency_by_period_df = pd.DataFrame(h2_frequency_rows)

    pre_period_df = df[df["time_period"] == "pre-COVID"].copy()
    covid_period_df = df[df["time_period"] == "COVID"].copy()
    post_period_df = df[df["time_period"] == "post-COVID"].copy()

    earlier_df = df[df["time_period"].isin(["pre-COVID", "COVID"])].copy()

    post_layoffs = int(post_period_df[layoffs_col].sum())
    earlier_layoffs = int(earlier_df[layoffs_col].sum())

    h2_frequency_test = two_proportion_z_test(
        success_a=post_layoffs,
        total_a=len(post_period_df),
        success_b=earlier_layoffs,
        total_b=len(earlier_df),
    )

    h2_frequency_test_df = pd.DataFrame([{
        "hypothesis": "H2_frequency",
        "comparison": "post-COVID vs earlier (pre-COVID + COVID)",
        "post_total_posts": len(post_period_df),
        "post_layoffs_posts": post_layoffs,
        "post_layoffs_share": safe_divide(post_layoffs, len(post_period_df)),
        "earlier_total_posts": len(earlier_df),
        "earlier_layoffs_posts": earlier_layoffs,
        "earlier_layoffs_share": safe_divide(earlier_layoffs, len(earlier_df)),
        "post_minus_earlier_layoffs_share": h2_frequency_test["diff_a_minus_b"],
        "z_stat": h2_frequency_test["z_stat"],
        "p_value_two_sided": h2_frequency_test["p_value_two_sided"],
        "supports_h2_frequency_directionally": (
            safe_divide(post_layoffs, len(post_period_df)) >
            safe_divide(earlier_layoffs, len(earlier_df))
            if len(post_period_df) > 0 and len(earlier_df) > 0 else False
        ),
    }])

    # H2B: layoffs topic has lower sentiment than non-layoffs
    layoffs_posts_df = df[df[layoffs_col] == 1].copy()
    non_layoffs_posts_df = df[df[layoffs_col] == 0].copy()

    layoffs_positive = int((layoffs_posts_df["predicted_label"] == 1).sum())
    non_layoffs_positive = int((non_layoffs_posts_df["predicted_label"] == 1).sum())

    h2_sentiment_test_overall = two_proportion_z_test(
        success_a=layoffs_positive,
        total_a=len(layoffs_posts_df),
        success_b=non_layoffs_positive,
        total_b=len(non_layoffs_posts_df),
    )

    post_layoffs_posts_df = post_period_df[post_period_df[layoffs_col] == 1].copy()
    post_non_layoffs_posts_df = post_period_df[post_period_df[layoffs_col] == 0].copy()

    post_layoffs_positive = int((post_layoffs_posts_df["predicted_label"] == 1).sum())
    post_non_layoffs_positive = int((post_non_layoffs_posts_df["predicted_label"] == 1).sum())

    h2_sentiment_test_post_only = two_proportion_z_test(
        success_a=post_layoffs_positive,
        total_a=len(post_layoffs_posts_df),
        success_b=post_non_layoffs_positive,
        total_b=len(post_non_layoffs_posts_df),
    )

    h2_sentiment_tests_df = pd.DataFrame([
        {
            "hypothesis": "H2_sentiment",
            "comparison_scope": "overall",
            "layoffs_total_posts": len(layoffs_posts_df),
            "layoffs_positive_rate": safe_divide(layoffs_positive, len(layoffs_posts_df)),
            "non_layoffs_total_posts": len(non_layoffs_posts_df),
            "non_layoffs_positive_rate": safe_divide(non_layoffs_positive, len(non_layoffs_posts_df)),
            "layoffs_minus_non_layoffs_positive_rate": h2_sentiment_test_overall["diff_a_minus_b"],
            "z_stat": h2_sentiment_test_overall["z_stat"],
            "p_value_two_sided": h2_sentiment_test_overall["p_value_two_sided"],
            "supports_h2_lower_sentiment_directionally": (
                safe_divide(layoffs_positive, len(layoffs_posts_df)) <
                safe_divide(non_layoffs_positive, len(non_layoffs_posts_df))
                if len(layoffs_posts_df) > 0 and len(non_layoffs_posts_df) > 0 else False
            ),
        },
        {
            "hypothesis": "H2_sentiment",
            "comparison_scope": "post-COVID only",
            "layoffs_total_posts": len(post_layoffs_posts_df),
            "layoffs_positive_rate": safe_divide(post_layoffs_positive, len(post_layoffs_posts_df)),
            "non_layoffs_total_posts": len(post_non_layoffs_posts_df),
            "non_layoffs_positive_rate": safe_divide(post_non_layoffs_positive, len(post_non_layoffs_posts_df)),
            "layoffs_minus_non_layoffs_positive_rate": h2_sentiment_test_post_only["diff_a_minus_b"],
            "z_stat": h2_sentiment_test_post_only["z_stat"],
            "p_value_two_sided": h2_sentiment_test_post_only["p_value_two_sided"],
            "supports_h2_lower_sentiment_directionally": (
                safe_divide(post_layoffs_positive, len(post_layoffs_posts_df)) <
                safe_divide(post_non_layoffs_positive, len(post_non_layoffs_posts_df))
                if len(post_layoffs_posts_df) > 0 and len(post_non_layoffs_posts_df) > 0 else False
            ),
        },
    ])

    report_lines.append("H2: Layoffs frequency and lower-sentiment checks")
    report_lines.append(h2_frequency_test_df.to_string(index=False))
    report_lines.append("")
    report_lines.append(h2_sentiment_tests_df.to_string(index=False))
    report_lines.append("")

    # ==============================================================
    # H3
    # User-level sentiment differs by topic, producing cross-topic correlations
    # ==============================================================
    print("Testing H3...")
    user_df = df[df["author_norm"] != ""].copy()

    author_post_counts = (
        user_df.groupby("author_norm")
               .size()
               .rename("total_posts_by_user")
               .reset_index()
    )

    eligible_authors = author_post_counts[
        author_post_counts["total_posts_by_user"] >= min_user_posts
    ]["author_norm"].tolist()

    eligible_user_df = user_df[user_df["author_norm"].isin(eligible_authors)].copy()

    h3_long_rows = []
    for bucket in bucket_names:
        topic_col = f"topic_{bucket}"
        topic_user_df = eligible_user_df[eligible_user_df[topic_col] == 1].copy()

        grouped = (
            topic_user_df.groupby("author_norm")
                         .agg(
                             posts_in_topic=("id", "size"),
                             avg_predicted_label=("predicted_label", "mean"),
                             avg_prob_positive=("prob_positive", "mean"),
                             avg_prob_negative=("prob_negative", "mean"),
                         )
                         .reset_index()
        )

        if len(grouped) == 0:
            continue

        grouped["topic_bucket"] = bucket
        grouped["topic_description"] = topic_buckets[bucket]["description"]
        h3_long_rows.append(grouped)

    if h3_long_rows:
        h3_user_topic_long_df = pd.concat(h3_long_rows, ignore_index=True)
    else:
        h3_user_topic_long_df = pd.DataFrame(columns=[
            "author_norm",
            "posts_in_topic",
            "avg_predicted_label",
            "avg_prob_positive",
            "avg_prob_negative",
            "topic_bucket",
            "topic_description",
        ])

    h3_topic_user_coverage_df = (
        h3_user_topic_long_df.groupby("topic_bucket")
                             .agg(
                                 users_with_topic=("author_norm", "nunique"),
                                 total_topic_posts_from_eligible_users=("posts_in_topic", "sum"),
                                 mean_user_topic_sentiment=("avg_predicted_label", "mean"),
                             )
                             .reset_index()
        if len(h3_user_topic_long_df) > 0
        else pd.DataFrame(columns=[
            "topic_bucket",
            "users_with_topic",
            "total_topic_posts_from_eligible_users",
            "mean_user_topic_sentiment",
        ])
    )

    if len(h3_user_topic_long_df) > 0:
        h3_user_topic_wide_df = (
            h3_user_topic_long_df.pivot(
                index="author_norm",
                columns="topic_bucket",
                values="avg_predicted_label"
            )
            .reset_index()
        )

        h3_corr_df = (
            h3_user_topic_long_df.pivot(
                index="author_norm",
                columns="topic_bucket",
                values="avg_predicted_label"
            )
            .corr(method="pearson", min_periods=2)
        )
    else:
        h3_user_topic_wide_df = pd.DataFrame()
        h3_corr_df = pd.DataFrame()

    if not h3_corr_df.empty:
        corr_long = h3_corr_df.copy()

        # Give the index and columns different names so they do not collide
        corr_long.index.name = "topic_a"
        corr_long.columns.name = "topic_b"

        # Convert matrix to long format without stack/reset_index collision
        corr_long = (
            corr_long.reset_index()
                    .melt(id_vars="topic_a", var_name="topic_b", value_name="correlation")
        )

        # Remove self-correlations like recruiting_pipeline vs recruiting_pipeline
        corr_long = corr_long[corr_long["topic_a"] != corr_long["topic_b"]].copy()

        if len(corr_long) > 0:
            max_abs_corr = corr_long["correlation"].abs().max()
        else:
            max_abs_corr = pd.NA
    else:
        max_abs_corr = pd.NA

    h3_summary_df = pd.DataFrame([{
        "hypothesis": "H3",
        "min_user_posts": min_user_posts,
        "eligible_users": len(eligible_authors),
        "user_topic_rows": len(h3_user_topic_long_df),
        "num_topic_buckets_with_user_data": (
            h3_user_topic_long_df["topic_bucket"].nunique()
            if len(h3_user_topic_long_df) > 0 else 0
        ),
        "max_abs_topic_correlation": max_abs_corr,
        "supports_h3_measurable_correlations": (
            pd.notna(max_abs_corr) and max_abs_corr > 0
        ),
        "note": (
            "Correlation matrix based on user-level average predicted sentiment by topic."
            if len(h3_user_topic_long_df) > 0
            else "Insufficient eligible user-topic data."
        ),
    }])

    report_lines.append("H3: User-level topic sentiment correlations")
    report_lines.append(h3_summary_df.to_string(index=False))
    report_lines.append("")

    # ==============================================================
    # H4
    # AI discussion increases are associated with overall sentiment shifts over time
    # ==============================================================
    print("Testing H4...")
    ai_col = "topic_ai_llm"

    valid_month_df = df[df["year_month"] != ""].copy()

    monthly_rows = []
    sorted_months = sort_year_month_strings(valid_month_df["year_month"].unique())

    for ym in sorted_months:
        month_df = valid_month_df[valid_month_df["year_month"] == ym].copy()
        total_posts = len(month_df)
        ai_posts = int(month_df[ai_col].sum())
        positive_posts = int((month_df["predicted_label"] == 1).sum())
        negative_posts = int((month_df["predicted_label"] == 0).sum())

        monthly_rows.append({
            "year_month": ym,
            "total_posts": total_posts,
            "ai_posts": ai_posts,
            "ai_share_of_all_posts": safe_divide(ai_posts, total_posts),
            "overall_positive_posts": positive_posts,
            "overall_negative_posts": negative_posts,
            "overall_positive_rate": safe_divide(positive_posts, total_posts),
            "overall_negative_rate": safe_divide(negative_posts, total_posts),
            "overall_avg_prob_positive": (
                float(month_df["prob_positive"].dropna().mean())
                if month_df["prob_positive"].notna().any() else pd.NA
            ),
            "overall_avg_prob_negative": (
                float(month_df["prob_negative"].dropna().mean())
                if month_df["prob_negative"].notna().any() else pd.NA
            ),
        })

    h4_monthly_df = pd.DataFrame(monthly_rows)

    if len(h4_monthly_df) >= 2:
        ai_positive_corr = h4_monthly_df["ai_share_of_all_posts"].corr(
            h4_monthly_df["overall_positive_rate"],
            method="pearson",
        )
        ai_negative_corr = h4_monthly_df["ai_share_of_all_posts"].corr(
            h4_monthly_df["overall_negative_rate"],
            method="pearson",
        )
    else:
        ai_positive_corr = pd.NA
        ai_negative_corr = pd.NA

    h4_corr_df = pd.DataFrame([{
        "hypothesis": "H4",
        "num_months_available": len(h4_monthly_df),
        "ai_share_vs_overall_positive_rate_corr": ai_positive_corr,
        "ai_share_vs_overall_negative_rate_corr": ai_negative_corr,
        "supports_h4_association_testable": len(h4_monthly_df) >= 2,
        "note": (
            "Correlation requires at least 2 months of data."
            if len(h4_monthly_df) < 2
            else "Correlation computed across monthly AI share and overall sentiment."
        ),
    }])

    report_lines.append("H4: AI discussion vs overall sentiment over time")
    report_lines.append(h4_corr_df.to_string(index=False))
    report_lines.append("")

    # ==============================================================
    # Save outputs
    # ==============================================================
    print("Saving Step 19 output files...")

    h1_period_output = output_dir / "h1_recruiting_pre_vs_post_period_summary.csv"
    h1_test_output = output_dir / "h1_recruiting_pre_vs_post_test.csv"

    h2_freq_period_output = output_dir / "h2_layoffs_frequency_by_period.csv"
    h2_freq_test_output = output_dir / "h2_layoffs_frequency_tests.csv"
    h2_sentiment_output = output_dir / "h2_layoffs_sentiment_tests.csv"

    h3_long_output = output_dir / "h3_user_topic_sentiment_long.csv"
    h3_wide_output = output_dir / "h3_user_topic_sentiment_wide.csv"
    h3_corr_output = output_dir / "h3_user_topic_correlation_matrix.csv"
    h3_coverage_output = output_dir / "h3_topic_user_coverage.csv"
    h3_summary_output = output_dir / "h3_summary.csv"

    h4_monthly_output = output_dir / "h4_monthly_ai_sentiment_trends.csv"
    h4_corr_output = output_dir / "h4_ai_sentiment_correlation.csv"

    report_output = output_dir / "hypothesis_test_report.txt"

    h1_period_summary_df.to_csv(h1_period_output, index=False, encoding="utf-8")
    h1_test_df.to_csv(h1_test_output, index=False, encoding="utf-8")

    h2_frequency_by_period_df.to_csv(h2_freq_period_output, index=False, encoding="utf-8")
    h2_frequency_test_df.to_csv(h2_freq_test_output, index=False, encoding="utf-8")
    h2_sentiment_tests_df.to_csv(h2_sentiment_output, index=False, encoding="utf-8")

    h3_user_topic_long_df.to_csv(h3_long_output, index=False, encoding="utf-8")
    h3_user_topic_wide_df.to_csv(h3_wide_output, index=False, encoding="utf-8")
    h3_corr_df.to_csv(h3_corr_output, encoding="utf-8")
    h3_topic_user_coverage_df.to_csv(h3_coverage_output, index=False, encoding="utf-8")
    h3_summary_df.to_csv(h3_summary_output, index=False, encoding="utf-8")

    h4_monthly_df.to_csv(h4_monthly_output, index=False, encoding="utf-8")
    h4_corr_df.to_csv(h4_corr_output, index=False, encoding="utf-8")

    with open(report_output, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print("Files saved successfully.")
    print()

    print("Quick results preview:")
    print("-" * 70)
    print("H1 summary:")
    print(h1_test_df.to_string(index=False))
    print()
    print("H2 frequency test:")
    print(h2_frequency_test_df.to_string(index=False))
    print()
    print("H2 sentiment test:")
    print(h2_sentiment_tests_df.to_string(index=False))
    print()
    print("H3 summary:")
    print(h3_summary_df.to_string(index=False))
    print()
    print("H4 summary:")
    print(h4_corr_df.to_string(index=False))
    print()

    print("Final summary")
    print("-" * 70)
    print(f"Saved: {h1_period_output}")
    print(f"Saved: {h1_test_output}")
    print(f"Saved: {h2_freq_period_output}")
    print(f"Saved: {h2_freq_test_output}")
    print(f"Saved: {h2_sentiment_output}")
    print(f"Saved: {h3_long_output}")
    print(f"Saved: {h3_wide_output}")
    print(f"Saved: {h3_corr_output}")
    print(f"Saved: {h3_coverage_output}")
    print(f"Saved: {h3_summary_output}")
    print(f"Saved: {h4_monthly_output}")
    print(f"Saved: {h4_corr_output}")
    print(f"Saved: {report_output}")
    print()
    print("Step 19 is complete when:")
    print("1. the script finishes without errors")
    print("2. these hypothesis output files are created")
    print("3. each hypothesis has a saved summary/test file")
    print("4. the text report is created")
    print("5. on the full dataset, H1-H4 can be interpreted from these outputs")
    print("=" * 70)


if __name__ == "__main__":
    main()