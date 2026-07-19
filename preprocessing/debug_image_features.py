import os
import cv2
import pandas as pd

IMAGE_DIR = "dataset/archive (9)/images"

df = pd.read_csv("dataset/pothole_features.csv")

missing = []
invalid_crop = []
resize_error = []

for index, row in df.iterrows():

    image_name = row["image_name"]

    if "-" in image_name:
        first, second = image_name.split("-", 1)
        if first.startswith("img"):
            image_name = row["image_name"]
        else:
            image_name = second

    image_path = os.path.join(IMAGE_DIR, image_name)

    image = cv2.imread(image_path)

    if image is None:
        missing.append(image_name)
        continue

    xmin = int(row["xmin"])
    ymin = int(row["ymin"])
    xmax = int(row["xmax"])
    ymax = int(row["ymax"])

    h, w = image.shape[:2]

    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(w, xmax)
    ymax = min(h, ymax)

    crop = image[ymin:ymax, xmin:xmax]

    if crop.size == 0:
        invalid_crop.append((index, image_name))
        continue

    try:
        cv2.resize(crop, (64, 64))
    except Exception:
        resize_error.append((index, image_name))

print("=" * 60)

print("Missing Images :", len(missing))
print("Invalid Crops :", len(invalid_crop))
print("Resize Errors :", len(resize_error))

print("=" * 60)

if invalid_crop:
    print("\nInvalid Crops")
    for item in invalid_crop:
        print(item)

if resize_error:
    print("\nResize Errors")
    for item in resize_error:
        print(item)