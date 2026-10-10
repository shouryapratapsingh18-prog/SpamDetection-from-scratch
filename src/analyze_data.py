import pandas as pd 

#load the dataset
df = pd.read_csv("data/spamham.csv" , encoding ="latin-1")

#1.Check the number of spam and ham messages  in the dataset
print("\n--- SPAM AND HAM MESSAGE COUNTS ---")
print(df['Label'].value_counts())

#2.Check percentage of each label 
print("\n--- Percentage of rach label ---")
print(df["Label"].value_counts(normalize=True)*100)

#3.Check whether label contain unexpected values
print("\n--- UNIQUE LABELS ---")
print(df["Label"].unique())

#4. Check the length of each message
print("\n--- MESSAGE LENGTHS ---")
df["Message_Length"] = df["Message"].str.len()

print("\n--- MESSAGE LENGTH STATISTICS ---")
print(df["Message_Length"].describe())


# 5. Display the shortest messages
print("\n--- FIVE SHORTEST MESSAGES ---")
print(
    df.nsmallest(5, "Message_Length")[["Label", "Message_Length", "Message"]]
)

# 6. Display the longest messages
print("\n--- FIVE LONGEST MESSAGES ---")
print(
    df.nlargest(5, "Message_Length")[["Label", "Message_Length", "Message"]]
)
