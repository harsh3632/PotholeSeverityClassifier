import os
import pandas as pd

INPUT_FILE = "dataset/pothole_dataset_clean.csv"
OUTPUT_FILE = "dataset/pothole_features.csv"

df = pd.read_csv(INPUT_FILE)

# -----------------------------------
# Feature Engineering
# -----------------------------------

# Bounding Box Aspect Ratio
df["aspect_ratio"] = (
    df["box_width"] /
    df["box_height"].replace(0, 1)
)

# Bounding Box Area Percentage
df["area_percentage"] = (
    df["box_area"] /
    (df["image_width"] * df["image_height"])
)

# Box Center Coordinates
df["center_x"] = (
    df["xmin"] + df["xmax"]
) / 2

df["center_y"] = (
    df["ymin"] + df["ymax"]
) / 2

# Relative Center Position
df["relative_center_x"] = (
    df["center_x"] /
    df["image_width"]
)

df["relative_center_y"] = (
    df["center_y"] /
    df["image_height"]
)

# Width Percentage
df["width_percentage"] = (
    df["box_width"] /
    df["image_width"]
)

# Height Percentage
df["height_percentage"] = (
    df["box_height"] /
    df["image_height"]
)

# Save
df.to_csv(OUTPUT_FILE, index=False)

print("=" * 60)
print("FEATURE ENGINEERING COMPLETED")
print("=" * 60)

print("\nDataset Shape :", df.shape)

print("\nNew Features")

new_columns = [
    "aspect_ratio",
    "area_percentage",
    "center_x",
    "center_y",
    "relative_center_x",
    "relative_center_y",
    "width_percentage",
    "height_percentage"
]

for column in new_columns:
    print(column)

print("\nSaved To :", OUTPUT_FILE)