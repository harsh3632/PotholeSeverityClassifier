import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

DATASET_DIR = "dataset"
MODEL_DIR = "models"

os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURE_FILE = os.path.join(DATASET_DIR, "training_features_pca.csv")
LABEL_FILE = os.path.join(DATASET_DIR, "training_labels.csv")

print("=" * 60)
print("Loading Dataset...")
print("=" * 60)

X = pd.read_csv(FEATURE_FILE)
y = pd.read_csv(LABEL_FILE)["label"]

print("Features Shape :", X.shape)
print("Labels Shape   :", y.shape)

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)

X_train.to_csv(os.path.join(DATASET_DIR, "X_train.csv"), index=False)
X_test.to_csv(os.path.join(DATASET_DIR, "X_test.csv"), index=False)

pd.DataFrame({"label": y_train}).to_csv(
    os.path.join(DATASET_DIR, "y_train.csv"),
    index=False
)

pd.DataFrame({"label": y_test}).to_csv(
    os.path.join(DATASET_DIR, "y_test.csv"),
    index=False
)

joblib.dump(encoder, os.path.join(MODEL_DIR, "label_encoder.pkl"))

print()
print("=" * 60)
print("TRAIN TEST SPLIT COMPLETED")
print("=" * 60)

print("Training Samples :", len(X_train))
print("Testing Samples  :", len(X_test))

print()

print("Saved Files")

print("dataset/X_train.csv")
print("dataset/X_test.csv")
print("dataset/y_train.csv")
print("dataset/y_test.csv")

print()

print("models/label_encoder.pkl")