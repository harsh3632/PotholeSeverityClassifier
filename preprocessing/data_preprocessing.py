import os
import pandas as pd

INPUT_FILE = "dataset/pothole_dataset.csv"
OUTPUT_FILE = "dataset/pothole_dataset_clean.csv"

df = pd.read_csv(INPUT_FILE)

print("=" * 60)
print("ORIGINAL DATASET INFORMATION")
print("=" * 60)

print("\nShape :", df.shape)

print("\nColumns")
print(df.columns.tolist())

print("\nData Types")
print(df.dtypes)

print("\nMissing Values")
print(df.isnull().sum())

print("\nDuplicate Rows :", df.duplicated().sum())

# Remove duplicate rows
df = df.drop_duplicates()

# Remove missing values (if any)
df = df.dropna()

# Reset index
df = df.reset_index(drop=True)

print("\n" + "=" * 60)
print("CLEANED DATASET INFORMATION")
print("=" * 60)

print("\nShape :", df.shape)

print("\nClass Distribution")

print(df["label"].value_counts())

# Save cleaned dataset
df.to_csv(OUTPUT_FILE, index=False)

print("\nClean dataset saved successfully.")
print("Location :", OUTPUT_FILE)