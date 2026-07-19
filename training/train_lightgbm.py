import os
import joblib
import pandas as pd

from lightgbm import LGBMClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

DATASET_DIR = "dataset"
MODEL_DIR = "models"

X_train = pd.read_csv(os.path.join(DATASET_DIR, "X_train.csv"))
X_test = pd.read_csv(os.path.join(DATASET_DIR, "X_test.csv"))

y_train = pd.read_csv(os.path.join(DATASET_DIR, "y_train.csv"))["label"]
y_test = pd.read_csv(os.path.join(DATASET_DIR, "y_test.csv"))["label"]

encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))

print("=" * 60)
print("Training LightGBM...")
print("=" * 60)

model = LGBMClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6,
    random_state=42
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

accuracy = accuracy_score(y_test, pred)
precision = precision_score(y_test, pred, average="weighted")
recall = recall_score(y_test, pred, average="weighted")
f1 = f1_score(y_test, pred, average="weighted")

print("\n" + "=" * 60)
print("LIGHTGBM RESULTS")
print("=" * 60)

print("Accuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1 Score :", round(f1, 4))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, pred))

print("\nClassification Report")
print(classification_report(
    y_test,
    pred,
    target_names=encoder.classes_
))

joblib.dump(model, os.path.join(MODEL_DIR, "lightgbm.pkl"))

print("\nModel Saved Successfully")
print("models/lightgbm.pkl")