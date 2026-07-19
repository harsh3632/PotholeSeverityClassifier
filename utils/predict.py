import os
import cv2
import joblib
import numpy as np
import pandas as pd

MODEL_DIR = "models"

MODEL = joblib.load(os.path.join(MODEL_DIR, "svm.pkl"))
PCA = joblib.load(os.path.join(MODEL_DIR, "pca.pkl"))
SCALER = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
LABEL_ENCODER = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))


def preprocess_image(image_path):
    """
    Load image, preprocess it exactly like training,
    then apply StandardScaler and PCA.
    """

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Unable to read image.")

    # Resize
    image = cv2.resize(image, (64, 64))

    # Convert BGR -> RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Normalize
    image = image.astype(np.float32) / 255.0

    # Flatten
    features = image.flatten().reshape(1, -1)

    # Convert to DataFrame so feature names match the scaler
    features = pd.DataFrame(
        features,
        columns=SCALER.feature_names_in_
    )

    # Apply Standard Scaling
    features = SCALER.transform(features)

    # Apply PCA
    features = PCA.transform(features)

    return features


def predict_severity(image_path):
    """
    Predict pothole severity from an image.
    """

    features = preprocess_image(image_path)

    prediction = MODEL.predict(features)[0]

    label = LABEL_ENCODER.inverse_transform([prediction])[0]

    return label


if __name__ == "__main__":

    test_image = "dataset/archive (9)/images/img-338.jpg"

    if os.path.exists(test_image):

        prediction = predict_severity(test_image)

        print("=" * 50)
        print("Predicted Severity :", prediction)
        print("=" * 50)

    else:
        print("Test image not found.")