import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Step 1: Load the cleaned dataset
df = pd.read_csv("data/cleaned_spamham.csv", encoding="latin-1")


#Step2: Separate input messages (X) and target lables(Y)
X = df["Message"]
Y = df["Label"]

#Step3: Split the dataset into training and testing sets
X_train , X_test , Y_train , Y_test = train_test_split(X, Y, test_size=0.2, random_state=42 , stratify = Y)
print("Training Messages:" , len(X_train))
print("Testing Messages:" , len(X_test))

#Step4: Convert text  into TF-IDF numerical features
vectorizer = TfidfVectorizer(
    lowercase = True,
    stop_words = "english",
    max_features = 10000
)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


#step5: Train a Multinomial Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train_tfidf , Y_train)

#Step6: Make predictions on unseen test messages
y_pred = model.predict(X_test_tfidf)

#Step7: Evaluate the model's performance
print("\n--- Model Accuracy ---")
print(accuracy_score(Y_test, y_pred))

print("\n--- Confusion Matrix ---")
print(confusion_matrix(Y_test, y_pred, labels = ["ham", "spam"]))

print("\n--- Classification Report ---")
print(classification_report(Y_test, y_pred, zero_division=0))