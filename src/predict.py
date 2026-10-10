
import joblib

# Load the saved Linear SVM pipeline
model = joblib.load("models/spam_detector.joblib")

# Messages to test
messages = [
    "Congratulations! You have won a free prize. Click here now!",
    "URGENT! Claim your cash reward now!",
    "You have won 10000000000000 dollars. Claim now!",
    "FREE entry in our prize draw. Reply now!",
    "Hey, are we meeting for lunch today?",
    "Please send me the assignment notes.",
    "Can we meet tomorrow at 10 AM?",
    "Your electricity bill is due tomorrow."
]
# Predict spam or ham
predictions = model.predict(messages)

# Get SVM decision scores
scores = model.decision_function(messages)

print("\n--- SPAM DETECTION RESULTS ---")

for message, prediction, score in zip(messages, predictions, scores):
    print(f"\nMessage: {message}")
    print(f"Prediction: {prediction.upper()}")
    print(f"SVM decision score: {score:.4f}")

    if prediction == "spam":
        print("Result: This message is classified as spam.")
    else:
        print("Result: This message is classified as legitimate (ham).")
