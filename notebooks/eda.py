import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Paths
# -----------------------------
INPUT_FILE = "dataset/pothole_dataset_clean.csv"
OUTPUT_DIR = "outputs/eda"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv(INPUT_FILE)

print("=" * 60)
print("EDA REPORT")
print("=" * 60)

print("\nDataset Shape :", df.shape)

print("\nStatistical Summary")
print(df.describe())

# -----------------------------
# 1. Severity Distribution
# -----------------------------
plt.figure(figsize=(8,5))

sns.countplot(
    data=df,
    x="label",
    order=df["label"].value_counts().index
)

plt.title("Pothole Severity Distribution")
plt.xlabel("Severity")
plt.ylabel("Count")
plt.xticks(rotation=15)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "severity_distribution.png"
    )
)

plt.close()

# -----------------------------
# 2. Box Area Distribution
# -----------------------------
plt.figure(figsize=(8,5))

sns.histplot(
    df["box_area"],
    bins=30,
    kde=True
)

plt.title("Bounding Box Area Distribution")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "box_area_distribution.png"
    )
)

plt.close()

# -----------------------------
# 3. Width vs Height
# -----------------------------
plt.figure(figsize=(8,6))

sns.scatterplot(
    data=df,
    x="box_width",
    y="box_height",
    hue="label"
)

plt.title("Width vs Height")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "width_height_relationship.png"
    )
)

plt.close()

# -----------------------------
# 4. Correlation
# -----------------------------
numeric_df = df.select_dtypes(include=["number"])

plt.figure(figsize=(10,7))

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="Blues"
)

plt.title("Correlation Heatmap")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "correlation_heatmap.png"
    )
)

plt.close()

print("\nEDA Completed Successfully")
print("Graphs Saved :", OUTPUT_DIR)