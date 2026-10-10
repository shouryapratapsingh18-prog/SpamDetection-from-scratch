
import streamlit as st
import joblib

# Configure the browser page
st.set_page_config(
    page_title="Spam Detection System",
    page_icon="📧",
    layout="centered"
)

# Load the saved machine learning pipeline
@st.cache_resource
def load_model():
    return joblib.load("models/spam_detector.joblib")

model = load_model()

# Page heading
st.title("📧 Spam Detection System")
st.write(
    "Enter an email or message below to check whether "
    "it is likely to be spam or legitimate."
)

# Message input
message = st.text_area(
    "Enter your message",
    placeholder="Example: Congratulations! Claim your prize now...",
    height=180
)

# Prediction button
if st.button("Check Message", type="primary"):
    if not message.strip():
        st.warning("Please enter a message first.")
    else:
        prediction = model.predict([message])[0]
        probabilities = model.predict_proba([message])[0]

        # Get probability for the predicted class
        class_index = list(model.classes_).index(prediction)
        confidence = probabilities[class_index]

        st.subheader("Prediction Result")

        if prediction == "spam":
            st.error("🚨 This message is classified as SPAM")
        else:
            st.success("✅ This message is classified as HAM")

        st.metric(
            "Model confidence",
            f"{confidence:.2%}"
        )

        st.caption(
            "Confidence is a model-estimated probability, "
            "not a guarantee of correctness."
        )

st.divider()
st.caption("Educational machine learning project • TF-IDF + Naive Bayes")
