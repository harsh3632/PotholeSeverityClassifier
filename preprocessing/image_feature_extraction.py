import os
import cv2
import pandas as pd

IMAGE_DIR = "dataset/archive (9)/images"
INPUT_FILE = "dataset/pothole_features.csv"
OUTPUT_FILE = "dataset/image_features.csv"

df = pd.read_csv(INPUT_FILE)

records = []

missing_images = []

for _, row in df.iterrows():

    image_name = row["image_name"]

    # Remove hash prefix if present
    if "-" in image_name:
        parts = image_name.split("-", 1)

        if parts[0].startswith("img"):
            corrected_name = image_name
        else:
            corrected_name = parts[1]

    else:
        corrected_name = image_name

    image_path = os.path.join(IMAGE_DIR, corrected_name)

    image = cv2.imread(image_path)

    if image is None:
        missing_images.append(corrected_name)
        continue

    xmin = int(row["xmin"])
    ymin = int(row["ymin"])
    xmax = int(row["xmax"])
    ymax = int(row["ymax"])

    crop = image[ymin:ymax, xmin:xmax]

    if crop.size == 0:
        continue

    crop = cv2.resize(crop, (64, 64))

    blue_mean = crop[:, :, 0].mean()
    green_mean = crop[:, :, 1].mean()
    red_mean = crop[:, :, 2].mean()

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    gray_mean = gray.mean()
    gray_std = gray.std()

    edges = cv2.Canny(gray, 100, 200)

    edge_density = edges.mean()

    records.append({

        "image_name": corrected_name,

        "label": row["label"],

        "blue_mean": blue_mean,
        "green_mean": green_mean,
        "red_mean": red_mean,

        "gray_mean": gray_mean,
        "gray_std": gray_std,

        "edge_density": edge_density,

        "box_width": row["box_width"],
        "box_height": row["box_height"],
        "box_area": row["box_area"],

        "aspect_ratio": row["aspect_ratio"],
        "area_percentage": row["area_percentage"]

    })

feature_df = pd.DataFrame(records)

feature_df.to_csv(OUTPUT_FILE, index=False)

print("=" * 60)
print("IMAGE FEATURE EXTRACTION COMPLETED")
print("=" * 60)

print("Extracted Samples :", len(feature_df))

print("Missing Images :", len(set(missing_images)))

if len(missing_images) > 0:
    print("\nMissing Image List")
    for name in sorted(set(missing_images)):
        print(name)

print("\nSaved To")
print(OUTPUT_FILE)