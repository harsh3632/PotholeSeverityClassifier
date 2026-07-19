from utils.repair_priority import get_repair_priority

print("=" * 50)

labels = [
    "major_pothole",
    "medium_pothole",
    "minor_pothole"
]

for label in labels:

    print(
        label,
        " --> ",
        get_repair_priority(label)
    )

print("=" * 50)