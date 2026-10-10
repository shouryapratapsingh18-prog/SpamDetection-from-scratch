
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# Load cleaned dataset
df = pd.read_csv("data/cleaned_spamham.csv")

X = df["Message"]
y = df["Label"]

# Use the same test split as the model comparison
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Build the Linear SVM pipeline
model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            max_features=10000
        )
    ),
    ("classifier", LinearSVC())
])

# Train only on the training split
model.fit(X_train, y_train)

# Predict unseen test messages
y_pred = model.predict(X_test)

print("\n--- LINEAR SVM EVALUATION ---")
print(f"Test messages: {len(y_test)}")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

print("\n--- CONFUSION MATRIX ---")
print("Labels: ham, spam")
print(confusion_matrix(y_test, y_pred, labels=["ham", "spam"]))

print("\n--- CLASSIFICATION REPORT ---")
print(classification_report(y_test, y_pred, labels=["ham", "spam"]))
