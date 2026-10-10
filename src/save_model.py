
import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

# Load the cleaned dataset
df = pd.read_csv("data/cleaned_spamham.csv")

X = df["Message"]
y = df["Label"]

# Create the final spam detection pipeline
spam_pipeline = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            max_features=10000
        )
    ),
    (
        "classifier",
        LinearSVC()
    )
])

# Train the model on the complete cleaned dataset
spam_pipeline.fit(X, y)

# Save the trained model
joblib.dump(spam_pipeline, "models/spam_detector.joblib")

print("Linear SVM model trained and saved successfully!")
print(f"Training messages: {len(X)}")
print("Model path: models/spam_detector.joblib")
