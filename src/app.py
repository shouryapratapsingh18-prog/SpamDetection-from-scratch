import os
import streamlit as st
import joblib

# Page configuration
st.set_page_config(page_title="Spam Detection AI", page_icon="✉️", layout="centered")

# Build the correct path to the model file
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "../models/spam_detector.joblib")

# Load the trained model with caching for better performance
@st.cache_resource
def load_model():
    return joblib.load(model_path)

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# App UI Header
st.title("✉️ Spam Detection AI")
st.write("An NLP project using TF-IDF and Linear Support Vector Machine.")

st.markdown("### Check a Message")
message = st.text_area("Enter an email or message:", placeholder="Paste your message here...")

if st.button("Check Message"):
    if not message.strip():
        st.warning("Please enter a non-empty message.")
    else:
        # Predict the message
        prediction = model.predict([message])[0]
        
        # Display Results
        st.markdown("---")
        st.markdown("### Result")
        
        if prediction.lower() == "spam":
            st.error(f"**Prediction:** {prediction.upper()} 🚨")
        else:
            st.success(f"**Prediction:** {prediction.upper()} ✅")

st.markdown("---")
st.caption("Educational machine learning project. Predictions may be incorrect.")