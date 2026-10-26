
from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "SMSSpamCollection"
MODEL_PATH = BASE_DIR / "spam_model.pkl"

# Load the tab-separated dataset: label, message
if not DATASET_PATH.exists():
    raise FileNotFoundError(
        "SMSSpamCollection file not found. "
        "Download and extract it into the project folder."
    )

data = pd.read_csv(
    DATASET_PATH,
    sep="\t",
    header=None,
    names=["label", "message"],
    encoding="utf-8",
)

data = data.dropna(subset=["label", "message"])
data["label"] = data["label"].str.strip().str.lower()
data["message"] = data["message"].astype(str)

data = data[data["label"].isin(["ham", "spam"])]
data = data.drop_duplicates(subset=["message"])

if data["label"].nunique() != 2:
    raise ValueError("Dataset must contain both ham and spam labels.")

X = data["message"]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

# Text processing + classifier in one pipeline
model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
        ),
    ),
    ("classifier", MultinomialNB()),
])

print("Training model...")
model.fit(X_train, y_train)

# Evaluate on held-out test messages
predictions = model.predict(X_test)

print("\nDataset size:", len(data))
print("Training messages:", len(X_train))
print("Test messages:", len(X_test))
print(f"Test accuracy: {accuracy_score(y_test, predictions):.4f}")

print("\nClassification report:")
print(classification_report(y_test, predictions, zero_division=0))

print("Confusion matrix (labels: ham, spam):")
print(confusion_matrix(y_test, predictions, labels=["ham", "spam"]))

# Save the complete pipeline
joblib.dump(model, MODEL_PATH)

print(f"\nModel saved successfully: {MODEL_PATH}")
