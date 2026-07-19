import os
import xml.etree.ElementTree as ET
import pandas as pd

# Dataset Paths
ANNOTATION_DIR = "dataset/archive (9)/annotations"
OUTPUT_DIR = "dataset"

os.makedirs(OUTPUT_DIR, exist_ok=True)

records = []

xml_files = sorted(
    [file for file in os.listdir(ANNOTATION_DIR) if file.endswith(".xml")]
)

for xml_file in xml_files:

    xml_path = os.path.join(ANNOTATION_DIR, xml_file)

    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename = root.find("filename").text

    size = root.find("size")
    width = int(size.find("width").text)
    height = int(size.find("height").text)

    for obj in root.findall("object"):

        label = obj.find("name").text

        bbox = obj.find("bndbox")

        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)

        box_width = xmax - xmin
        box_height = ymax - ymin
        area = box_width * box_height

        records.append({
            "image_name": filename,
            "image_width": width,
            "image_height": height,
            "label": label,
            "xmin": xmin,
            "ymin": ymin,
            "xmax": xmax,
            "ymax": ymax,
            "box_width": box_width,
            "box_height": box_height,
            "box_area": area
        })

df = pd.DataFrame(records)

output_file = os.path.join(OUTPUT_DIR, "pothole_dataset.csv")

df.to_csv(output_file, index=False)

print("=" * 50)
print("Dataset Created Successfully")
print("=" * 50)
print("Total Records :", len(df))
print("Unique Images :", df["image_name"].nunique())
print("\nClass Distribution")
print(df["label"].value_counts())
print("\nSaved To :", output_file)