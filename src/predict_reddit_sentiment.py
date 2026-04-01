"""
predict_reddit_sentiment.py

Steps 11 and 12 of the project:
- Step 11: Load the saved TF-IDF vectorizer and the selected trained sentiment model
- Step 12: Run sentiment prediction on the Reddit posts

Input:
- Step 10 CSV with cleaned_text

Model files:
- models/tfidf_vectorizer.joblib
- models/logistic_regression.joblib

Output:
- A new CSV with predicted sentiment labels

Run from src/:
    python .\predict_reddit_sentiment.py

Optional:
    python .\predict_reddit_sentiment.py ^
        --input .\data\reddit\processed\RS_2023-02_cleaned_text.csv ^
        --vectorizer .\models\tfidf_vectorizer.joblib ^
        --model .\models\logistic_regression.joblib ^
        --output .\data\reddit\predictions\RS_2023-02_sentiment_predictions.csv
"""

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd


DEFAULT_INPUT = Path("data/reddit/processed/RS_2023-02_cleaned_text.csv")
DEFAULT_VECTORIZER = Path("models/tfidf_vectorizer.joblib")
DEFAULT_MODEL = Path("models/logistic_regression.joblib")
DEFAULT_OUTPUT = Path("data/reddit/predictions/RS_2023-02_sentiment_predictions.csv")

REQUIRED_COLUMNS = [
    "id",
    "subreddit",
    "author",
    "created_utc",
    "title",
    "body",
    "final_text",
    "cleaned_text",
]


def main():
    parser = argparse.ArgumentParser(
        description="Steps 11 and 12: load saved TF-IDF + sentiment model and predict Reddit sentiment."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(DEFAULT_INPUT),
        help="Path to Step 10 CSV with cleaned_text",
    )
    parser.add_argument(
        "--vectorizer",
        type=str,
        default=str(DEFAULT_VECTORIZER),
        help="Path to saved TF-IDF vectorizer joblib file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=str(DEFAULT_MODEL),
        help="Path to saved sentiment model joblib file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help="Path to output CSV with sentiment predictions",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    vectorizer_path = Path(args.vectorizer)
    model_path = Path(args.model)
    output_path = Path(args.output)

    print("=" * 70)
    print("STEPS 11 AND 12: LOAD SAVED MODEL OBJECTS AND PREDICT SENTIMENT")
    print("=" * 70)
    print(f"Input file:      {input_path}")
    print(f"Vectorizer file: {vectorizer_path}")
    print(f"Model file:      {model_path}")
    print(f"Output file:     {output_path}")
    print()

    # ------------------------------------------------------------------
    # Step 11: load the saved TF-IDF vectorizer and trained sentiment model
    # ------------------------------------------------------------------
    print("Checking required files...")
    if not input_path.exists():
        print(f"ERROR: Input file does not exist: {input_path}")
        sys.exit(1)

    if not vectorizer_path.exists():
        print(f"ERROR: Vectorizer file does not exist: {vectorizer_path}")
        sys.exit(1)

    if not model_path.exists():
        print(f"ERROR: Model file does not exist: {model_path}")
        sys.exit(1)

    print("All required files are present.")
    print()

    print("Reading Step 10 CSV...")
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

    print("Dropping rows with missing or empty cleaned_text...")
    before_drop = len(df)
    df["cleaned_text"] = df["cleaned_text"].fillna("").astype(str).str.strip()
    df = df[df["cleaned_text"].str.len() > 0].reset_index(drop=True)
    dropped_rows = before_drop - len(df)

    print(f"Rows before drop: {before_drop:,}")
    print(f"Rows dropped:     {dropped_rows:,}")
    print(f"Rows remaining:   {len(df):,}")
    print()

    if len(df) == 0:
        print("ERROR: No rows remain after checking cleaned_text.")
        sys.exit(1)

    print("Loading saved TF-IDF vectorizer...")
    vectorizer = joblib.load(vectorizer_path)
    print(f"Vectorizer loaded: {type(vectorizer).__name__}")

    print("Loading saved sentiment model...")
    model = joblib.load(model_path)
    print(f"Model loaded: {type(model).__name__}")
    print()

    # ------------------------------------------------------------------
    # Step 12: transform Reddit text and predict sentiment
    # ------------------------------------------------------------------
    print("Transforming cleaned_text into TF-IDF features...")
    X_reddit = vectorizer.transform(df["cleaned_text"])
    print(f"TF-IDF matrix shape: {X_reddit.shape}")
    print()

    print("Predicting sentiment labels...")
    df["predicted_label"] = model.predict(X_reddit)
    df["predicted_sentiment"] = df["predicted_label"].map({
        0: "negative",
        1: "positive",
    })
    print("Label prediction complete.")
    print()

    if hasattr(model, "predict_proba"):
        print("Computing sentiment probabilities...")
        probs = model.predict_proba(X_reddit)
        df["prob_negative"] = probs[:, 0]
        df["prob_positive"] = probs[:, 1]
        print("Probability computation complete.")
        print()
    else:
        print("Model does not support predict_proba(); skipping probability columns.")
        print()

    print("Prediction summary:")
    label_counts = df["predicted_sentiment"].value_counts(dropna=False)
    for label, count in label_counts.items():
        print(f"  {label}: {count:,}")
    print()

    print("Prediction percentages:")
    label_percentages = df["predicted_sentiment"].value_counts(normalize=True, dropna=False) * 100
    for label, pct in label_percentages.items():
        print(f"  {label}: {pct:.2f}%")
    print()

    print("Predictions by subreddit:")
    subreddit_summary = (
        df.groupby(["subreddit", "predicted_sentiment"])
          .size()
          .unstack(fill_value=0)
    )
    print(subreddit_summary.to_string())
    print()

    preview_cols = [
        "id",
        "subreddit",
        "cleaned_text",
        "predicted_label",
        "predicted_sentiment",
    ]
    if "prob_negative" in df.columns and "prob_positive" in df.columns:
        preview_cols.extend(["prob_negative", "prob_positive"])

    print("Preview of prediction output:")
    print(df[preview_cols].head(5).to_string(index=False))
    print()

    print("Creating output folder if needed...")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Saving prediction CSV...")
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