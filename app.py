import numpy as np
from flask import Flask, request, jsonify
import joblib
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from utils import *

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
app.config["CORS_HEADERS"] = "Content-Type"

# Define the custom MSE loss function (if applicable to any deep learning models)
def custom_mse_loss(y_true, y_pred):
    return K.mean(K.square(y_true - y_pred), axis=-1)

# **Load the TF-IDF vectorizer and models**
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"
MODEL_PATHS = {
    "cnn": "models/cnn.h5",
    "gradient_boosting": "models/gradient_boosting.pkl",
    "logistic_regression": "models/logistic_regression.pkl",
    "mlp": "models/mlp.h5",
    "random_forest": "models/random_forest.pkl",
    "svm": "models/svm.pkl",
}

# Initialize containers for models and vectorizer
models = {}
tfidf_vectorizer = None

# Load TF-IDF vectorizer
try:
    tfidf_vectorizer = joblib.load(VECTORIZER_PATH)
    print("TF-IDF vectorizer loaded successfully.")
except Exception as e:
    print(f"Error loading TF-IDF vectorizer: {e}")

# Load models
for model_name, model_path in MODEL_PATHS.items():
    try:
        if model_path.endswith(".h5"):
            models[model_name] = load_model(model_path, custom_objects={'custom_mse_loss': custom_mse_loss})
        else:
            models[model_name] = joblib.load(model_path)
        print(f"{model_name} model loaded successfully.")
    except Exception as e:
        print(f"Error loading {model_name}: {e}")


# **Flask Routes**
@app.route("/")
def home():
    return "Welcome to the AI Prediction API!"

@app.route("/predict", methods=["POST", "GET"])
def predict():
    if request.method == "POST":
        try:
            # Parse the request data
            data = request.get_json()
            biography = data.get("biography")

            # Preprocess the user input and extract features
            cleaned_input_features = clean_process_and_extract_features(biography, tfidf_vectorizer)
            print(f"Processed features: {cleaned_input_features}")

            # Iterate through models and make predictions
            best_model = None
            best_prediction = None
            highest_confidence = -1
            confidence_scores = {}

            for model_name, model in models.items():
                print(f"Making prediction with {model_name}...")

                if model_name in ["cnn", "mlp"]:
                    reshaped_input = cleaned_input_features.reshape((1, -1))
                    prediction = model.predict(reshaped_input)
                    confidence = np.max(prediction)
                    prediction = np.argmax(prediction)
                elif model_name == "svm":
                    decision_score = model.decision_function(cleaned_input_features)
                    confidence = 1 / (1 + np.exp(-decision_score[0]))
                    prediction = model.predict(cleaned_input_features)
                elif hasattr(model, "predict_proba"):
                    prediction_probabilities = model.predict_proba(cleaned_input_features)
                    confidence = max(prediction_probabilities[0])
                    prediction = model.predict(cleaned_input_features)
                else:
                    prediction = model.predict(cleaned_input_features)
                    confidence = 1  # Default for non-probabilistic models

                confidence_scores[model_name] = confidence
                if confidence > highest_confidence:
                    highest_confidence = confidence
                    best_prediction = prediction[0] if isinstance(prediction, np.ndarray) else prediction

            # Return the best prediction
            return jsonify({
                "prediction": int(best_prediction) if best_prediction is not None else None,
                "confidence": float(highest_confidence) if highest_confidence is not None else None,
                "confidence_scores": {model: float(conf) for model, conf in confidence_scores.items()}
            })


        except Exception as e:
            return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True, port=7070)