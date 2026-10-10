
import joblib

# Load the saved pipeline
model = joblib.load("models/spam_detector.joblib")

# Messages to test
messages = [
    "Congratulations! You have won a free prize. Claim now!",
    "Hi, are we meeting for lunch today?",
    "URGENT! You have won money. Click here now!",
    "Please send me the assignment before class."
]

# Predict labels
predictions = model.predict(messages)

# Get estimated probabilities for each class
probabilities = model.predict_proba(messages)

# Get the class names in the model's actual order
classes = model.classes_

for message, prediction, probability in zip(
    messages, predictions, probabilities
):
    print("\nMessage:", message)
    print("Prediction:", prediction.upper())

    # Display the probability of each class
    for class_name, score in zip(classes, probability):
        print(f"{class_name.upper()} probability: {score:.2%}")

    # Highest class probability (not a guarantee of correctness)
    print(f"Model confidence: {max(probability):.2%}")
