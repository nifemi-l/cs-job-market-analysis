# EECS 767 Final Project

Authors: Anahita Memar, Nifemi Lawal

CS job-market sentiment analysis across market shifts (pre-COVID, COVID, post-COVID).
We built an end-to-end text pipeline using Reddit data and a sentiment model trained
on Sentiment140.

## Project Highlights

- **1.6M** labeled tweets used for sentiment model training (Sentiment140)
- **1.54M** Reddit posts analyzed
- **108** months of data (2017-2025)
- **4** core subreddits: `jobs`, `cscareerquestions`, `csmajors`, `recruitinghell`
- **807k** topic-tagged posts
- **48.26%** overall positive sentiment (51.74% negative)

## What We Did

- Trained and selected a sentiment classifier (Logistic Regression outperformed Naive Bayes).
- Built a reproducible pipeline for Reddit filtering, preprocessing, inference, and analysis.
- Grouped results by month and by era (pre-COVID, COVID, post-COVID).
- Added topic tagging (recruiting pipeline, layoffs/market, compensation, AI/LLM, and more).
- Generated summary tables/plots and hypothesis-support checks.

## Key Results (Simple Overview)

- Sentiment became more negative over time:  
pre-COVID **52.03%** positive -> COVID **49.00%** -> post-COVID **46.51%**.
- Subreddit mood differs a lot:
  - `csmajors`: **59.07%** positive
  - `cscareerquestions`: **56.48%**
  - `jobs`: **42.14%**
  - `recruitinghell`: **39.55%**
- Most negative themes include layoffs/market and compensation.
- AI/LLM discussion increased strongly in recent years.

## Tech Stack

- Python
- pandas, numpy
- scikit-learn (TF-IDF, Logistic Regression, Naive Bayes)
- matplotlib, seaborn
- zstandard (for compressed Reddit input handling)

## Final Report

The final written report is available in:

- `docs/EECS_767_Project_Report`

## Data Inputs

- Sentiment140 CSV:
`src/data/sentiment140/training.1600000.processed.noemoticon.csv`
- Reddit input files (`.zst`, `.jsonl`, `.ndjson`):
`src/data/reddit/`

## Minimal Reproduction

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the full workflow from project root:

```bash
python src/training/train_sentiment.py
python src/pipeline/filter_reddit_submissions.py
python src/pipeline/merge_filtered_csvs.py
python src/pipeline/select_reddit_fields.py
python src/pipeline/build_reddit_final_text.py
python src/pipeline/preprocess_reddit_text.py
python src/pipeline/predict_reddit_sentiment.py
python src/analysis/group_reddit_by_time.py
python src/utils/topic_buckets.py
python src/analysis/tag_reddit_topics.py
python src/analysis/compare_sentiment_by_topic.py
python src/analysis/measure_topic_trends_over_time.py
python src/reporting/test_project_hypotheses.py
python src/reporting/create_final_report_outputs.py
```

## Limitations

- Training labels come from Sentiment140 (Twitter), so domain mismatch with Reddit can affect sentiment accuracy.
- Topic tagging is keyword-based and may miss nuanced phrasing or implicit context.
- Sentiment classification does not fully capture sarcasm and complex discourse.
- Reddit reflects community discussion, not the full CS job market population.
- Results are primarily correlational and should not be interpreted as causal.

## Next Steps

- Fine-tune sentiment models on Reddit-native labeled data for better domain fit.
- Replace keyword tagging with semantic topic classification.
- Extend analysis to comment threads and user-level trajectories.
- Add external labor-market indicators to contextualize trend changes.

