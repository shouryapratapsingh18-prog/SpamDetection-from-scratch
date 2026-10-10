
import streamlit as st
import joblib

# Load the trained Linear SVM pipeline
model = joblib.load("models/spam_detector.joblib")

# Page configuration
st.set_page_config(
    page_title="Spam Detection AI",
    page_icon="📧",
    layout="centered"
)

# App heading
st.title("📧 Spam Detection AI")
st.write(
    "Enter an email or message below to check "
    "whether it is spam or legitimate."
)

# Message input
message = st.text_area(
    "Enter your message:",
    placeholder="Paste your email or message here...",
    height=180
)

# Predict button
if st.button("Check Message", type="primary"):
    if not message.strip():
        st.warning("Please enter a message first.")
    else:
        prediction = model.predict([message])[0]
        score = model.decision_function([message])[0]

        st.subheader("Analysis Result")

        if prediction == "spam":
            st.error("🚨 SPAM DETECTED")
            st.write("The model classifies this message as spam.")
        else:
            st.success("✅ LEGITIMATE MESSAGE (HAM)")
            st.write("The model classifies this message as legitimate.")

        st.metric("SVM Decision Score", f"{score:.4f}")

        st.caption(
            "The decision score is not a probability or confidence "
            "percentage. Scores on opposite sides of the decision "
            "boundary correspond to different predicted classes."
        )

st.divider()
st.caption("Educational machine learning project. Predictions may be incorrect.")
