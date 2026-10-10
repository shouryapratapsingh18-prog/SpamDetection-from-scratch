
import joblib

MODEL_PATH = "models/spam_detector.joblib"


def test_model_predictions():
    model = joblib.load(MODEL_PATH)

    messages = [
        "Congratulations! You won a free prize. Claim now!",
        "Hey, are we meeting for lunch today?"
    ]

    predictions = model.predict(messages)

    assert predictions[0] == "spam", (
        f"Expected spam, got {predictions[0]}"
    )
    assert predictions[1] == "ham", (
        f"Expected ham, got {predictions[1]}"
    )


def test_model_output_labels():
    model = joblib.load(MODEL_PATH)

    predictions = model.predict([
        "Your assignment notes are ready."
    ])

    assert predictions[0] in ["spam", "ham"]


if __name__ == "__main__":
    test_model_predictions()
    test_model_output_labels()
    print("All spam detector tests passed!")
