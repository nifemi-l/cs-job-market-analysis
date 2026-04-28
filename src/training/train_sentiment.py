"""
train_sentiment.py

Trains a sentiment classifier on Sentiment140 (1.6M tweets).
80/20 split, TF-IDF, then Logistic Regression + Naive Bayes.

Saves the vectorizer and both models to src/models/ so we can
just load them later for the Reddit stuff instead of retraining.

Run:
    python src/training/train_sentiment.py
"""

import os
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC_DIR))

import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from utils.preprocessing import load_sentiment140, preprocess_dataframe

DATA_PATH = SRC_DIR / "data" / "sentiment140" / "training.1600000.processed.noemoticon.csv"
MODEL_DIR = SRC_DIR / "models"


def main():
    print("Loading Sentiment140...")
    df = load_sentiment140(str(DATA_PATH))
    print(f"  Loaded {len(df)} rows  (neg={sum(df['label']==0)}, pos={sum(df['label']==1)})")

    print("Preprocessing text...")
    df = preprocess_dataframe(df)
    print(f"  {len(df)} rows after cleaning")

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"],
        test_size=0.2, random_state=42, stratify=df["label"],
    )
    print(f"  Train: {len(X_train)}  Test: {len(X_test)}")

    print("Fitting TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(max_features=50_000, sublinear_tf=True)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print("\nTraining Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, solver="lbfgs")
    lr.fit(X_train_tfidf, y_train)

    y_pred_lr = lr.predict(X_test_tfidf)
    print("=== Logistic Regression ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}")
    print(f"F1:       {f1_score(y_test, y_pred_lr):.4f}")
    print(classification_report(y_test, y_pred_lr, target_names=["negative", "positive"]))
    print(confusion_matrix(y_test, y_pred_lr))

    print("\nTraining Naive Bayes...")
    nb = MultinomialNB()
    nb.fit(X_train_tfidf, y_train)

    y_pred_nb = nb.predict(X_test_tfidf)
    print("=== Naive Bayes ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_nb):.4f}")
    print(f"F1:       {f1_score(y_test, y_pred_nb):.4f}")
    print(classification_report(y_test, y_pred_nb, target_names=["negative", "positive"]))
    print(confusion_matrix(y_test, y_pred_nb))

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(vectorizer, str(MODEL_DIR / "tfidf_vectorizer.joblib"))
    joblib.dump(lr, str(MODEL_DIR / "logistic_regression.joblib"))
    joblib.dump(nb, str(MODEL_DIR / "naive_bayes.joblib"))
    print(f"\nModels saved to {MODEL_DIR}/")


if __name__ == "__main__":
    main()
