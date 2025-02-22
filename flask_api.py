from flask import Flask, request, jsonify
import joblib
import pandas as pd
from flask_cors import CORS

# ✅ Initialize Flask App
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# ✅ Load the trained model and selected features
model = joblib.load("fire_detection_model.pkl")
selected_features = joblib.load("selected_features.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # ✅ Get JSON data from request
        data = request.get_json()
        df = pd.DataFrame(data)

        # ✅ Ensure only the selected features are used
        df = df[selected_features]

        # ✅ Predict using the trained model
        predictions = model.predict(df)

        return jsonify({"predictions": predictions.tolist()})
    
    except KeyError as e:
        return jsonify({"error": f"Missing required feature: {str(e)}"})
    
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=False, port=5001)
