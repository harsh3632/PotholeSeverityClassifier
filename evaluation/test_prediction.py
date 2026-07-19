from utils.predict import predict_severity

IMAGE_PATH = "dataset/archive (9)/images/img-338.jpg"

prediction = predict_severity(IMAGE_PATH)

print("=" * 50)
print("Predicted Severity :", prediction)
print("=" * 50)