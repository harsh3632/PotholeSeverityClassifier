import os
import pandas as pd
from sklearn.model_selection import train_test_split

INPUT_FILE = "dataset/pothole_features.csv"
OUTPUT_DIR = "dataset"

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(INPUT_FILE)

# -----------------------------
# Feature Selection
# -----------------------------

FEATURES = [
    "image_width",
    "image_height",
    "xmin",
    "ymin",
    "xmax",
    "ymax",
    "box_width",
    "box_height",
    "box_area",
    "aspect_ratio",
    "area_percentage",
    "center_x",
    "center_y",
    "relative_center_x",
    "relative_center_y",
    "width_percentage",
    "height_percentage"
]

TARGET = "label"

X = df[FEATURES]
y = df[TARGET]

# -----------------------------
# Train + Temp
# -----------------------------

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# -----------------------------
# Validation + Test
# -----------------------------

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

# -----------------------------
# Save
# -----------------------------

train_df = X_train.copy()
train_df["label"] = y_train.values

val_df = X_val.copy()
val_df["label"] = y_val.values

test_df = X_test.copy()
test_df["label"] = y_test.values

train_df.to_csv("dataset/train_dataset.csv", index=False)
val_df.to_csv("dataset/validation_dataset.csv", index=False)
test_df.to_csv("dataset/test_dataset.csv", index=False)

print("=" * 60)
print("DATASET SPLITTING COMPLETED")
print("=" * 60)

print("\nTraining Samples :", len(train_df))
print("Validation Samples :", len(val_df))
print("Testing Samples :", len(test_df))

print("\nFiles Saved")

print("dataset/train_dataset.csv")
print("dataset/validation_dataset.csv")
print("dataset/test_dataset.csv")