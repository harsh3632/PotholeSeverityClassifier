import os
import cv2
import numpy as np
import pandas as pd

IMAGE_DIR = "dataset/archive (9)/images"
INPUT_FILE = "dataset/pothole_features.csv"

OUTPUT_FEATURES = "dataset/training_features.csv"
OUTPUT_LABELS = "dataset/training_labels.csv"

IMAGE_SIZE = 64

df = pd.read_csv(INPUT_FILE)

X = []
y = []

invalid_samples = []

for index, row in df.iterrows():

    image_name = row["image_name"]

    # Remove hash prefix if present
    if "-" in image_name:
        first, second = image_name.split("-", 1)
        if not first.startswith("img"):
            image_name = second

    image_path = os.path.join(IMAGE_DIR, image_name)

    image = cv2.imread(image_path)

    if image is None:
        invalid_samples.append(index)
        continue

    h, w = image.shape[:2]

    xmin = max(0, int(row["xmin"]))
    ymin = max(0, int(row["ymin"]))
    xmax = min(w, int(row["xmax"]))
    ymax = min(h, int(row["ymax"]))

    if xmax <= xmin or ymax <= ymin:
        invalid_samples.append(index)
        continue

    crop = image[ymin:ymax, xmin:xmax]

    if crop.size == 0:
        invalid_samples.append(index)
        continue

    crop = cv2.resize(crop, (IMAGE_SIZE, IMAGE_SIZE))

    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

    feature_vector = crop.flatten()

    X.append(feature_vector)

    y.append(row["label"])

X = np.array(X)

feature_columns = []

for i in range(X.shape[1]):
    feature_columns.append("pixel_" + str(i))

feature_df = pd.DataFrame(X, columns=feature_columns)
label_df = pd.DataFrame({"label": y})

feature_df.to_csv(OUTPUT_FEATURES, index=False)
label_df.to_csv(OUTPUT_LABELS, index=False)

print("=" * 60)
print("TRAINING DATA PREPARED")
print("=" * 60)

print("Valid Samples :", len(feature_df))
print("Invalid Samples :", len(invalid_samples))
print()

print("Feature Shape :", feature_df.shape)

print()

print("Label Distribution")
print(label_df["label"].value_counts())

print()

print("Saved")
print(OUTPUT_FEATURES)
print(OUTPUT_LABELS)