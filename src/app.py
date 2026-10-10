import os
import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Page configuration
st.set_page_config(
    page_title="Spam Detection AI",
    page_icon="📧",
    layout="wide"
)

# Determine absolute paths relative to src/app.py
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "../models/spam_detector.joblib")
data_path = os.path.join(current_dir, "../data/cleaned_spamham.csv")

# Load saved model with caching
@st.cache_resource
def load_saved_model():
    return joblib.load(model_path)

try:
    model = load_saved_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

st.title("📧 Spam Detection AI")
st.write("An NLP project using TF-IDF and Linear Support Vector Machine.")

page = st.sidebar.radio(
    "Navigate",
    ["Spam Detector", "Model Performance"]
)

# --------------------------------------------------
# PAGE 1: SPAM DETECTOR
# --------------------------------------------------
if page == "Spam Detector":

    st.header("Check a Message")

    message = st.text_area(
        "Enter an email or message:",
        placeholder="Paste your message here...",
        height=180
    )

    if st.button("Check Message", type="primary"):
        if not message.strip():
            st.warning("Please enter a message first.")
        else:
            prediction = model.predict([message])[0]
            score = model.decision_function([message])[0]

            st.subheader("Analysis Result")

            if prediction.lower() == "spam":
                st.error("🚨 SPAM DETECTED")
                st.write("The model classifies this message as spam.")
            else:
                st.success("✅ LEGITIMATE MESSAGE (HAM)")
                st.write("The model classifies this message as legitimate.")

            st.metric("SVM Decision Score", f"{score:.4f}")

            st.caption(
                "The SVM decision score is not a probability "
                "or confidence percentage."
            )

# --------------------------------------------------
# PAGE 2: MODEL PERFORMANCE
# --------------------------------------------------
else:

    st.header("Model Performance Dashboard")

    @st.cache_data
    def load_data():
        df = pd.read_csv(data_path)
        return df

    @st.cache_resource
    def evaluate_model():
        df = load_data()

        X = df["Message"]
        y = df["Label"]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )

        evaluation_model = Pipeline([
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

        evaluation_model.fit(X_train, y_train)
        predictions = evaluation_model.predict(X_test)

        return y_test, predictions

    try:
        y_test, predictions = evaluate_model()

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(
            y_test, predictions, pos_label="spam", zero_division=0
        )
        recall = recall_score(
            y_test, predictions, pos_label="spam", zero_division=0
        )
        f1 = f1_score(
            y_test, predictions, pos_label="spam", zero_division=0
        )

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Accuracy", f"{accuracy:.2%}")
        col2.metric("Spam Precision", f"{precision:.2%}")
        col3.metric("Spam Recall", f"{recall:.2%}")
        col4.metric("Spam F1-score", f"{f1:.2%}")

        st.caption(
            "Metrics are computed on the held-out test split. "
            "They may differ from results on other datasets."
        )

        st.subheader("Confusion Matrix")

        matrix = confusion_matrix(
            y_test, predictions, labels=["ham", "spam"]
        )

        fig, ax = plt.subplots()
        image = ax.imshow(matrix)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Ham", "Spam"])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Ham", "Spam"])

        ax.set_xlabel("Predicted label")
        ax.set_ylabel("Actual label")
        ax.set_title("Linear SVM Confusion Matrix")

        for row in range(2):
            for col in range(2):
                ax.text(
                    col, row, str(matrix[row, col]),
                    ha="center", va="center"
                )

        fig.colorbar(image, ax=ax, label="Number of messages")
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Classification Report")

        report = classification_report(
            y_test,
            predictions,
            labels=["ham", "spam"],
            output_dict=True,
            zero_division=0
        )

        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.round(3), width="stretch")

    except FileNotFoundError as error:
        st.error(
            f"Required file not found: {error}. "
            "Check that data/cleaned_spamham.csv exists."
        )

st.divider()
st.caption(
    "Educational machine learning project. "
    "Predictions may be incorrect."
)