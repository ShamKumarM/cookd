import json
import pickle

from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

root = Path(__file__).resolve().parents[2]

data_path = root / "data" / "intent_training.json"

model_path = root / "models" / "intent_classifier.pkl"


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

rows = json.loads(
    data_path.read_text()
)

texts = [
    row["text"]
    for row in rows
]

labels = [
    row["intent"]
    for row in rows
]


# ---------------------------------------------------------
# TRAIN / VALIDATION SPLIT
# ---------------------------------------------------------

X_train, X_val, y_train, y_val = train_test_split(
    texts,
    labels,
    test_size=0.20,
    random_state=42,
    stratify=labels
)


print()
print("Dataset:")
print("Total:", len(rows))
print("Training:", len(X_train))
print("Validation:", len(X_val))


# ---------------------------------------------------------
# CREATE MODEL
# ---------------------------------------------------------

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            sublinear_tf=True
        )
    ),
    (
        "clf",
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced"
        )
    )
])


# ---------------------------------------------------------
# TRAIN ON TRAINING SET
# ---------------------------------------------------------

model.fit(
    X_train,
    y_train
)


# ---------------------------------------------------------
# VALIDATION
# ---------------------------------------------------------

predictions = model.predict(X_val)

accuracy = accuracy_score(
    y_val,
    predictions
)


print()
print("=" * 60)
print("VALIDATION RESULTS")
print("=" * 60)

print()
print(f"Accuracy: {accuracy:.4f}")

print()
print(
    classification_report(
        y_val,
        predictions
    )
)


# ---------------------------------------------------------
# FINAL TRAINING
# ---------------------------------------------------------

print()
print("Training final model on full dataset...")

model.fit(
    texts,
    labels
)


# ---------------------------------------------------------
# SAVE MODEL
# ---------------------------------------------------------

model_path.parent.mkdir(
    exist_ok=True
)

with open(
    model_path,
    "wb"
) as f:

    pickle.dump(
        model,
        f
    )


print()
print("=" * 60)
print("MODEL SAVED")
print("=" * 60)

print(
    model_path
)