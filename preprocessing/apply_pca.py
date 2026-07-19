import os
import joblib
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

INPUT_FILE = "dataset/training_features.csv"
OUTPUT_FILE = "dataset/training_features_pca.csv"

print("=" * 60)
print("Loading Features...")
print("=" * 60)

X = pd.read_csv(INPUT_FILE)

print("Original Shape :", X.shape)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(
    n_components=256,
    random_state=42
)

X_pca = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame(X_pca)

pca_df.to_csv(OUTPUT_FILE, index=False)

joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
joblib.dump(pca, os.path.join(MODEL_DIR, "pca.pkl"))

print()
print("=" * 60)
print("PCA COMPLETED")
print("=" * 60)

print("Original Shape :", X.shape)
print("Reduced Shape  :", pca_df.shape)

print()
print("Explained Variance Ratio :",
      round(pca.explained_variance_ratio_.sum(), 4))

print()

print("Saved")
print(OUTPUT_FILE)
print("models/scaler.pkl")
print("models/pca.pkl") 