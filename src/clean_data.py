import pandas as pd

#Step 1: Load the dataset
df = pd.read_csv("data/spamham.csv", encoding="latin-1")

print("--- Original Dataset ---")
print("Shape:", df.shape)
print("Labels:", df["Label"].unique())

#Step2: Remove unnecessary spaces from labels
df["labels"] = df["Label"].str.strip().str.lower()

#Step3: Remove rows with missing labels or messages
df = df.dropna(subset=["labels", "Message"])

#Step4: Remove messages that are empty after trimming space
df["Messages"] = df["Message"].str.strip()
df = df[df["Messages"] != ""]

#step 5: Keep only the expected labels
df = df[df["Label"].isin(["spam", "ham"])]

# Step 6: Remove exact duplicate rows
df = df.drop_duplicates()

# Step 7: Display the cleaned dataset
print("\n--- CLEANED DATASET ---")
print("Shape:", df.shape)

print("\n--- LABEL COUNTS ---")
print(df["Label"].value_counts())

print("\n--- MISSING VALUES ---")
print(df[["Label", "Message"]].isnull().sum())

# Step 8: Save the cleaned dataset
df.to_csv("data/cleaned_spamham.csv", index=False)

print("\nCleaned dataset saved successfully!")
