
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Load cleaned dataset
df = pd.read_csv("data/cleaned_spamham.csv")

X = df["Message"]
y = df["Label"]

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Build a pipeline: text -> TF-IDF -> classifier
spam_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=10000
    )),
    ("classifier", MultinomialNB())
])

# Train the pipeline
spam_pipeline.fit(X_train, y_train)

# Create a folder to store the trained model
import os
os.makedirs("models", exist_ok=True)

# Save the complete pipeline
joblib.dump(spam_pipeline, "models/spam_detector.joblib")

print("Model saved successfully!")
print("Location: models/spam_detector.joblib")
