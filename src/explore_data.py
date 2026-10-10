
# Step 1: Import pandas
import pandas as pd

# Step 2: Load our dataset
file_path = "data/spamham.csv"

df = pd.read_csv(file_path, encoding="latin-1")

# Step 3: Display the dataset dimensions
print("\n--- DATASET SHAPE ---")
print(df.shape)

# Step 4: Display the column names
print("\n--- COLUMN NAMES ---")
print(df.columns.tolist())

# Step 5: Display the first five rows
print("\n--- FIRST FIVE ROWS ---")
print(df.head())

# Step 6: Check missing values
print("\n--- MISSING VALUES ---")
print(df.isnull().sum())

# Step 7: Check duplicate rows
print("\n--- DUPLICATE ROWS ---")
print(df.duplicated().sum())

# Step 8: Display column data types
print("\n--- DATA TYPES ---")
print(df.dtypes)

