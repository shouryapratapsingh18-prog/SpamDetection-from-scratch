# Spam Detection from Scratch

A machine learning project that classifies email and text messages as **Spam** or **Ham (legitimate)** using Natural Language Processing (NLP) and supervised learning.

## Project Overview

This project compares three machine learning algorithms and uses TF-IDF text vectorization to convert messages into numerical features.

The models evaluated are:
- Multinomial Naive Bayes
- Logistic Regression
- Linear Support Vector Machine (Linear SVM)

Linear SVM achieved the best results on the held-out test set.

## Model Performance

**Dataset:** 10,162 messages before cleaning  
**Test set:** 2,033 messages  
**Selected model:** Linear SVM

| Metric | Score |
|---|---:|
| Accuracy | 97.44% |
| Spam Precision | 94.92% |
| Spam Recall | 92.67% |
| Spam F1-score | 93.78% |

## Confusion Matrix

| Actual / Predicted | Ham | Spam |
|---|---:|---:|
| Ham | 1,589 | 21 |
| Spam | 31 | 392 |

The model missed 31 spam messages and incorrectly flagged 21 legitimate messages in the test set.

## Technologies Used

- Python
- Pandas
- Scikit-learn
- TF-IDF Vectorization
- Linear SVM
- Joblib
- Streamlit

## Project Structure

```text
SpamDetection-from-scratch/
├── data/                  # Local datasets; not committed
├── models/                # Saved model; not committed
├── src/
│   ├── analyze_data.py
│   ├── clean_data.py
│   ├── explore_data.py
│   ├── train_model.py
│   ├── save_model.py
│   ├── predict.py
│   ├── compare_models.py
│   └── evaluate_model.py
├── web_app.py
├── requirements.txt
├── .gitignore
└── README.md
```

## How It Works

1. Load and clean the message dataset.
2. Split the data into training and test sets.
3. Convert text into TF-IDF features.
4. Train and compare machine learning classifiers.
5. Evaluate performance using accuracy, precision, recall, F1-score, and a confusion matrix.
6. Save the selected Linear SVM pipeline.
7. Predict whether new messages are spam or legitimate using a Streamlit web app.

## Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/shouryapratapsingh18-prog/SpamDetection-from-scratch.git
cd SpamDetection-from-scratch
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Add the dataset

Place your authorized dataset at:

```text
data/spamham.csv
```

Run the data cleaning script to generate `data/cleaned_spamham.csv`.

### 5. Train and save the model

```bash
python src/save_model.py
```

### 6. Launch the web app

```bash
python -m streamlit run web_app.py
```

## Limitations

- Predictions are not always correct.
- The model may perform differently on messages from sources unlike the training dataset.
- SVM decision scores are not calibrated probabilities.
- Evaluation results depend on the dataset and test split.
- Dataset redistribution rights should be checked before sharing the dataset.

## Future Improvements

- Tune hyperparameters using cross-validation.
- Analyze false positives and false negatives.
- Add automated tests.
- Add a more detailed Streamlit interface.
- Explore model explainability and deployment.

## Author

Shourya Pratap Singh

This project was built for learning and demonstrating practical machine learning and NLP skills.
