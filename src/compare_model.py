
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the cleaned dataset
df = pd.read_csv("data/cleaned_spamham.csv")

X = df["Message"]
y = df["Label"]

# Use the same test split for a fair comparison
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Define the models to compare
models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Linear SVM": LinearSVC()
}

results = []

for name, classifier in models.items():

    # Each model gets its own TF-IDF pipeline
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            max_features=10000
        )),
        ("classifier", classifier)
    ])

    # Train
    pipeline.fit(X_train, y_train)

    # Predict unseen test messages
    predictions = pipeline.predict(X_test)

    # Calculate evaluation metrics
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Spam Precision": precision_score(
            y_test, predictions, pos_label="spam", zero_division=0
        ),
        "Spam Recall": recall_score(
            y_test, predictions, pos_label="spam", zero_division=0
        ),
        "Spam F1": f1_score(
            y_test, predictions, pos_label="spam", zero_division=0
        )
    })

# Display comparison
results_df = pd.DataFrame(results)

print("\n--- MODEL COMPARISON ---")
print(results_df.round(4).to_string(index=False))

# Save results for documentation
results_df.to_csv("model_comparison.csv", index=False)
print("\nResults saved to model_comparison.csv")
