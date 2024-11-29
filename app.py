import os
import re
import string
import emoji
import pickle
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from flask import Flask, request, jsonify
from flask_cors import CORS
import nltk
from scipy import sparse  # For padding/truncating sparse matrices

# Download necessary NLTK data
nltk.download('punkt')

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
app.config["CORS_HEADERS"] = "Content-Type"

# Define preprocessing functions
def remove_html_tags(text):
    if isinstance(text, str):
        soup = BeautifulSoup(text, "html.parser")
        return soup.get_text(separator=" ")
    return text

def remove_urls(text):
    return re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

def remove_emojis(text):
    return emoji.replace_emoji(text, replace='')

def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

def remove_extra_whitespace(text):
    return re.sub(r'\s+', ' ', text).strip()

def expand_contractions(text):
    contractions_dict = {
        # Add contractions mapping here
    }
    for contraction, expansion in contractions_dict.items():
        text = text.replace(contraction, expansion)
    return text

def tokenize_text(text):
    return word_tokenize(text)

# Preprocess user input
def preprocess_user_input(input_data):
    cleaned_data = {}
    key_mapping = {
        "vocationaltraining": "vocational_training",
        "currentconviction": "current_conviction",
        "previousconviction": "previous_conviction",
        "gender": "gender",
        "age": "age",
        "location": "location",
        "familysupport": "family_support"
    }
    for key, value in input_data.items():
        normalized_key = key.strip().lower().replace(" ", "")
        mapped_key = key_mapping.get(normalized_key, normalized_key)
        if isinstance(value, str):
            cleaned_text = value
            cleaned_text = remove_html_tags(cleaned_text)
            cleaned_text = remove_urls(cleaned_text)
            cleaned_text = remove_emojis(cleaned_text)
            cleaned_text = remove_punctuation(cleaned_text)
            cleaned_text = remove_extra_whitespace(cleaned_text)
            cleaned_text = expand_contractions(cleaned_text)
            cleaned_text = tokenize_text(cleaned_text)
            cleaned_data[mapped_key] = cleaned_text
        else:
            cleaned_data[mapped_key] = value

    # Debugging print
    print("Preprocessed input data:", cleaned_data)

    if 'vocational_training' not in cleaned_data:
        raise ValueError("Key 'vocational_training' is missing from preprocessed data.")
    return cleaned_data

# Load models
def load_models():
    models = {}
    model_names = ["svm", "random_forest", "gradient_boosting"]
    for model_name in model_names:
        model_path = os.path.join("models", f'{model_name}.pkl')
        with open(model_path, 'rb') as file:
            models[model_name] = pickle.load(file)
    return models

# Load vectorizer
def load_vectorizer():
    vectorizer_path = os.path.join("models", "tfidf_vectorizer.pkl")
    with open(vectorizer_path, 'rb') as file:
        vectorizer = pickle.load(file)
    # Debugging prints
    print("Vectorizer vocabulary size:", len(vectorizer.vocabulary_))
    print("Sample vocabulary:", list(vectorizer.vocabulary_.items())[:10])
    return vectorizer

# Map prediction to label
def map_prediction_to_label(prediction):
    print(f"Mapping prediction: {prediction}")
    if prediction == 0:
        return "Non-Recidivism"
    elif prediction == 1:
        return "Recidivism"
    else:
        return "Unknown"  # Handle unexpected cases explicitly

# Align input with model expectations
def align_features_with_model(numeric_features, model):
    expected_features = model.n_features_in_
    actual_features = numeric_features.shape[1]
    print(f"Aligning features: model expects {expected_features}, input has {actual_features}")
    
    if actual_features < expected_features:
        padding = sparse.csr_matrix((numeric_features.shape[0], expected_features - actual_features))
        numeric_features = sparse.hstack([numeric_features, padding])
    elif actual_features > expected_features:
        numeric_features = numeric_features[:, :expected_features]
    print("Aligned features shape:", numeric_features.shape)
    return numeric_features

def make_predictions(models, cleaned_data):
    try:
        # Ensure 'vocational_training' key exists
        if 'vocational_training' not in cleaned_data:
            raise ValueError("Key 'vocational_training' is missing from preprocessed data.")

        # Preprocess and transform input text
        feature_input = " ".join(cleaned_data['vocational_training'])
        vectorizer = load_vectorizer()
        numeric_features = vectorizer.transform([feature_input])
        print("Transformed feature shape:", numeric_features.shape)

        predictions = {}
        for model_name, model in models.items():
            try:
                # Handle SVM separately if it requires dense input
                numeric_features_dense = numeric_features.toarray() if model_name == "svm" else numeric_features
                
                # Align features with model's expected input dimensions
                aligned_features = align_features_with_model(numeric_features_dense, model)
                print(f"Aligned features shape: {aligned_features.shape}")

                # Get the prediction and probability
                prediction = model.predict(aligned_features)
                prediction_proba = model.predict_proba(aligned_features)  # Get class probabilities

                print(f"Raw prediction for {model_name}: {prediction}")
                print(f"Prediction probabilities for {model_name}: {prediction_proba}")

                # Assuming binary classification, the probability for class 1 (Recidivism)
                confidence_score = prediction_proba[0][1]  # Probability of class 1 (Recidivism)

                # Directly return the raw prediction and confidence score
                predictions[model_name] = {
                    "prediction": prediction[0],  # Raw prediction
                    "confidence_score": confidence_score  # Confidence score
                }

                print(f"Prediction for {model_name}: {prediction[0]}, Confidence Score: {confidence_score}")
            except Exception as e:
                # Log and skip problematic models
                print(f"Skipping model '{model_name}' due to error: {str(e)}")

        # If no valid predictions were made
        if not predictions:
            raise RuntimeError("No valid predictions could be made with the loaded models.")

        # Return the predictions and confidence scores as the final response
        return predictions
    except Exception as e:
        raise RuntimeError(f"Error during prediction: {str(e)}")



# Predict endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse and validate input data
        user_input = request.json
        if not user_input:
            return jsonify({"error": "No input data provided"}), 400
        print("Received input:", user_input)  # Debugging print
        
        # Preprocess input and load models
        cleaned_user_input = preprocess_user_input(user_input)
        models = load_models()
        
        # Make predictions
        predictions = make_predictions(models, cleaned_user_input)
        print(f"#######{predictions}#######")
        return jsonify(predictions), 200
    except Exception as e:
        # Return error in response
        print("Error during prediction:", str(e))  # Debugging print
        return jsonify({"error": str(e)}), 500

# Run Flask app
if __name__ == '__main__':
    app.run(debug=True, port=3000)
