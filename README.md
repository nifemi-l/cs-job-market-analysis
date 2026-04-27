# EECS 767 Project

CS job-market sentiment analysis using:
- **Sentiment140** for model training
- **Reddit** data for downstream analysis, hypothesis tests, and report outputs

## Current State

The pipeline is implemented end-to-end.

Generated artifacts exist under:
- `src/data/reddit/analysis/`
- `src/data/reddit/final_outputs/`

`src/data/reddit/final_outputs/written_summary/final_summary.md` is the latest
human-readable summary output from the reporting step.

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

## Required Inputs

### 1) Sentiment140 training file

Place the Sentiment140 CSV at:

```text
src/data/sentiment140/training.1600000.processed.noemoticon.csv
```

### 2) Reddit input file(s)

Put Reddit input files (`.zst`, `.jsonl`, `.ndjson`) in:

```text
src/data/reddit/
```

`filter_reddit_submissions.py` can auto-detect a default file in that folder,
or you can pass `--input`.

## Project Structure

```text
src/
├── training/
│   └── train_sentiment.py
├── pipeline/
│   ├── filter_reddit_submissions.py
│   ├── merge_filtered_csvs.py
│   ├── select_reddit_fields.py
│   ├── build_reddit_final_text.py
│   ├── preprocess_reddit_text.py
│   ├── predict_reddit_sentiment.py
│   ├── upload_to_spaces.py
│   └── download_from_spaces.py
├── analysis/
│   ├── group_reddit_by_time.py
│   ├── tag_reddit_topics.py
│   ├── compare_sentiment_by_topic.py
│   └── measure_topic_trends_over_time.py
├── reporting/
│   ├── test_project_hypotheses.py
│   └── create_final_report_outputs.py
├── utils/
│   ├── preprocessing.py
│   └── topic_buckets.py
└── data/
    ├── sentiment140/
    └── reddit/
```

## End-to-End Run Order

Run from project root:

```bash
# 1) Train sentiment artifacts (creates src/models/*.joblib)
python src/training/train_sentiment.py

# 2) Filter Reddit data to target subreddits
# (pass --input explicitly if needed)
python src/pipeline/filter_reddit_submissions.py

# 3) Merge per-subreddit filtered CSVs into one file
python src/pipeline/merge_filtered_csvs.py

# 4) Keep required columns
python src/pipeline/select_reddit_fields.py

# 5) Build final_text
python src/pipeline/build_reddit_final_text.py

# 6) Clean final_text
python src/pipeline/preprocess_reddit_text.py

# 7) Predict sentiment
python src/pipeline/predict_reddit_sentiment.py

# 8) Add time features / period labels
python src/analysis/group_reddit_by_time.py

# 9) Save topic bucket definitions
python src/utils/topic_buckets.py

# 10) Topic tagging
python src/analysis/tag_reddit_topics.py

# 11) Topic sentiment comparisons
python src/analysis/compare_sentiment_by_topic.py

# 12) Topic trends over time
python src/analysis/measure_topic_trends_over_time.py

# 13) Hypothesis tests
python src/reporting/test_project_hypotheses.py

# 14) Final tables, plots, and summaries
python src/reporting/create_final_report_outputs.py
```

## Key Pipeline Outputs

Primary intermediate files:
- `src/data/reddit/filtered/all_subreddits_filtered.csv`
- `src/data/reddit/processed/all_subreddits_selected_fields.csv`
- `src/data/reddit/processed/all_subreddits_final_text.csv`
- `src/data/reddit/processed/all_subreddits_cleaned_text.csv`
- `src/data/reddit/predictions/all_subreddits_sentiment_predictions.csv`
- `src/data/reddit/analysis/all_subreddits_time_grouped.csv`
- `src/data/reddit/analysis/all_subreddits_topic_tagged.csv`

Analysis/report outputs:
- `src/data/reddit/analysis/step17_topic_sentiment/`
- `src/data/reddit/analysis/step18_topic_trends/`
- `src/data/reddit/analysis/step19_hypothesis_tests/`
- `src/data/reddit/final_outputs/tables/`
- `src/data/reddit/final_outputs/plots/`
- `src/data/reddit/final_outputs/written_summary/`

## Optional: DigitalOcean Spaces Sync

If you use Spaces sync scripts, create a root `.env`:

```text
DO_SPACES_KEY=your-access-key-id
DO_SPACES_SECRET=your-secret-access-key
DO_SPACES_REGION=sfo3
DO_SPACES_BUCKET=eecs767-reddit
```

Upload:

```bash
python src/pipeline/upload_to_spaces.py
```

Download:

```bash
python src/pipeline/download_from_spaces.py
```

## Notes

- `filter_reddit_submissions.py` reads `.zst` directly with `zstandard`.
- Prediction depends on `src/models/tfidf_vectorizer.joblib` and
  `src/models/logistic_regression.joblib`.