
import joblib

# Load the trained model
model = joblib.load("models/spam_detector.joblib")

print("=" * 45)
print("       SPAM DETECTION SYSTEM")
print("=" * 45)
print("Enter an email or message to classify.")
print("Type 'exit' to quit.")

while True:
    message = input("\nEnter your message: ").strip()

    # Stop the program when requested
    if message.lower() == "exit":
        print("Exiting Spam Detection System. Goodbye!")
        break

    # Handle empty input
    if not message:
        print("Please enter a non-empty message.")
        continue

    # Predict the message
    prediction = model.predict([message])[0]
    probabilities = model.predict_proba([message])[0]

    # Find the probability for the predicted class
    class_index = list(model.classes_).index(prediction)
    confidence = probabilities[class_index]

    print("\n--- RESULT ---")
    print("Prediction:", prediction.upper())
    print(f"Model confidence: {confidence:.2%}")
