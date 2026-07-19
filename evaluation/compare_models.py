import os
import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

DATASET_DIR = "dataset"
MODEL_DIR = "models"
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("Loading Test Dataset...")
print("=" * 60)

X_test = pd.read_csv(os.path.join(DATASET_DIR, "X_test.csv"))
y_test = pd.read_csv(os.path.join(DATASET_DIR, "y_test.csv"))["label"]

models = {
    "Random Forest": "random_forest.pkl",
    "XGBoost": "xgboost.pkl",
    "LightGBM": "lightgbm.pkl",
    "Gradient Boosting": "gradient_boosting.pkl",
    "SVM": "svm.pkl"
}

results = []

print("\nEvaluating Models...\n")

for model_name, filename in models.items():

    model_path = os.path.join(MODEL_DIR, filename)

    model = joblib.load(model_path)

    prediction = model.predict(X_test)

    accuracy = accuracy_score(y_test, prediction)
    precision = precision_score(
        y_test,
        prediction,
        average="weighted",
        zero_division=0
    )
    recall = recall_score(
        y_test,
        prediction,
        average="weighted",
        zero_division=0
    )
    f1 = f1_score(
        y_test,
        prediction,
        average="weighted",
        zero_division=0
    )

    results.append({
        "Model": model_name,
        "Accuracy": round(accuracy, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1 Score": round(f1, 4)
    })

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Accuracy",
    ascending=False
).reset_index(drop=True)

print("=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(results_df)

output_file = os.path.join(
    OUTPUT_DIR,
    "model_comparison.csv"
)

results_df.to_csv(output_file, index=False)

print("\nComparison saved successfully.")
print(output_file)