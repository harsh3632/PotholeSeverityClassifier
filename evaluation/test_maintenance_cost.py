from utils.maintenance_cost import get_maintenance_cost

print("=" * 50)

labels = [
    "major_pothole",
    "medium_pothole",
    "minor_pothole"
]

for label in labels:
    print(
        f"{label} --> ₹{get_maintenance_cost(label):,}"
    )

print("=" * 50)