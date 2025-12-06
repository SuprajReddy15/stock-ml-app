from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = joblib.load("rf_model.pkl")
scaler = joblib.load("scaler.pkl")

# Simple health check
@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"status": "ok"}), 200

# Prediction endpoint
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    # Expect these keys in JSON
    try:
        close = float(data["close"])
        ret = float(data["return"])
        ma5 = float(data["ma5"])
        ma10 = float(data["ma10"])
        volume = float(data["volume"])
    except (KeyError, ValueError, TypeError):
        return jsonify({"error": "Invalid input format"}), 400

    X = np.array([[close, ret, ma5, ma10, volume]])
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)[0]

    return jsonify({"predicted_next_close": float(pred)}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
