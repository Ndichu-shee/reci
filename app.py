import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
import joblib
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
import logging
from langdetect import detect
# Keywords for feature extraction
mo_keywords = [
    'young', 'children', 'teenager', 'strangulation', 'weapon', 
    'knife', 'firearm', 'poison', 'arson', 'blunt object', 
    'suffocation', 'drowning', 'abduction', 'disguise', 'ritual', 
    'sedation', 'chloroform', 'ambush', 'stalking', 'decapitation', 
    'asphyxiation', 'bludgeoning', 'toxic', 'drugging', 'overdose', 
    'hanging', 'beheading', 'mutilation', 'explosives', 'torture', 
    'electrocution', 'rope', 'binding', 'axe', 'club', 
    'scalpel', 'acid', 'trap', 'entrapment', 'concealment', 
    'disposal', 'body parts', 'massacre', 'premeditation', 
    'manipulation', 'coercion', 'vehicle', 'hunting', 'booby trap', 
    'ruse', 'camouflage', 'home invasion', 'brute force', 'slashing'
]
behavior_keywords = [
    'torture', 'murdered', 'stalking', 'rape', 'assault', 
    'necrophilia', 'psychosis', 'suicidal', 'abuse', 'depression',
    'homicide', 'strangulation', 'dismemberment', 'ritualistic', 
    'kidnapping', 'sadism', 'voyeurism', 'obsession', 'manipulation', 
    'arson', 'compulsion', 'delusions', 'schizophrenia', 'paranoia',
    'narcissism', 'psychopathy', 'violence', 'trauma', 'mutilation',
    'control', 'dominance', 'fantasies', 'sociopathy', 'pedophilia',
    'exploitation', 'betrayal', 'infliction', 'victimization',
    'revenge', 'power', 'intimidation', 'hunting', 'isolation',
    'harassment', 'addiction', 'escalation', 'deceit', 'maniacal',
    'disguise', 'repression', 'manhunt', 'insanity', 'hostility'
]

psychological_keywords = [
    'depression', 'paranoia', 'aggression', 'narcissism', 'anxiety',
    'guilt', 'psychosis', 'trauma', 'sadism', 'obsession', 'mania',
    'delusion', 'schizophrenia', 'compulsion', 'neurosis', 'psychopathy',
    'sociopathy', 'self-harm', 'melancholia', 'euphoria', 'apathy',
    'insomnia', 'detachment', 'hallucination', 'fear', 'anger',
    'isolation', 'hostility', 'addiction', 'impulsivity', 'violence',
    'phobia', 'grief', 'emotional', 'instability', 'manipulation',
    'revenge', 'control', 'loneliness', 'despair', 'suicidal',
    'bipolar', 'hypomania', 'dissociation', 'alienation',
    'repression', 'hypervigilance', 'anhedonia', 'helplessness',
    'dependency', 'remorse', 'mood swings', 'withdrawal'
]

filler_words = [
        "um", "uh", "like", "you know", "actually", "basically", "sort of", "kind of",
        "I mean", "you see", "well", "so", "right", "okay", "anyways", "look", "intially"
        "listen", "believe me", "virtually", "practically", "probably", "maybe", "perhaps",
        "frankly", "nearly", "almost", "essentially", "utterly", "completely","really"
        "just", "only", "mainly", "stuff", "thing", "things", "somewhat", "seemingly"
        "in a sense", "at the end of the day", "to be honest", "you know what I mean",
        "to tell you the truth", "in other words", "as a matter of fact", "in fact",
        "for the most part", "on the whole", "generally", "usually", "frequently",
        "often", "occasionally", "at least", "kind of", "sort of", "more or less",
        "up to a point", "in many ways", "in some way", "as it were", "in effect",
        "in reality", "to some extent", "to a certain extent", "likely", "really",
        "presumably", "ostensibly", "evidently", "seemingly", "apparently",
        "probably", "likely", "somehow", "someway", "let's say", "let's",
        "roughly", "you know", "at all", "basically", "let me be clear",
        "if you will", "as such", "let's be honest", "in truth", "sort of like",
        "in general", "like I said", "as far as I know", "suffice it to say",
        "for what it's worth", "I guess", "I suppose", "certainly", "absolutely",
        "totally", "merely", "specifically", "explicitly", "implicitly","exactly"
        "in particular", "especially", "not really", "not entirely", "particularly"
        "not totally", "to be fair", "to clarify", "to put it simply",
        "to put it mildly", "let me put it this way", "the thing is",
        "it's like", "I mean to say", "just saying", "and all",
        "if you know what I mean", "as far as I'm concerned", "for instance",
        "for example", "such as", "you might say", "that is",
        "like I said before", "and stuff", "or something", "or whatever",
        "basically", "kind of like", "just about", "in some respects",
        "to make a long story short", "when it comes to", "in this case",
        "in light of", "and so on", "and whatnot",
        'actually', 'ah', 'alright', 'anyway', 'apparently', 'basically', 'er', 'frankly',
        'honestly', 'just', 'kinda', 'like', 'literally', 'look', 'obviously', 'okay',
        'really', 'right', 'so', 'truthfully', 'uh', 'um', 'well', 'yeah',"Aaah", "Umm",
        "Uhh", "Awww", "Eh", "Erm", "Hmmm", "Oh", "Mmm", "Like", "You know", "Sort of",
        "Kind of", "I mean", "Actually", "Basically", "Literally", "Right", "Okay", "Well",
        "So", "Uh-huh", "Yeah", "Hmm", "Ahh", "Hmmmm", "Ehhh", "Y'know", "Let's see", "Alright",
        "Right?", "Honestly", "Seriously", "Just", "Anyway", "Look", "Listen", "Believe me", "To be honest",
        "At the end of the day","Aah", "Aaah", "Aaaaah", "Umm", "Ummm", "Ummmm", "Uhh", "Uhhh", "Uhhhh", "Awww",
        "Awwww", "Awwwww", "Hmm", "Hm", "Hmmmm", "Oh", "Ooh", "Ohhh", "Oooooh", "Er", "Err", "Errr", "Yeah", "Yeaah",
        "Yeeeah", "Yeeeaaaah", "Like", "Liiike", "Liiiiike", "Okay", "Okaaay", "Ok", "Oooookay", "Right", "Riiight", "Riiiight", "So", "Sooo", "Soooo", "Sooooo", "No", "Nooo", "Noooo", "Nooooo", "Well", "Weell", "Weeellll"
    ]

additional_stop_words = [
        "movie", "film", "films", "cinema", "watch", "watched", "watching", "seen","theatre"
        "story", "stories", "character", "characters", "plot", "scene", "scenes","rehearsal"
        "performance", "performances", "acting", "actor", "actress", "directing","release"
        "director", "script", "scripts", "dialogue", "dialogues", "sound", "music","setting"
        "score", "visuals", "effects", "special", "time", "just", "still","advertised"
        "too", "even", "also", "that", "it", "its", "them", "they", "you", "your",
        "there", "here", "these", "those", "but", "and",
        "or", "as", "if", "for", "to", "of", "in", "with", "at", "by", "about",
        "from", "this", "what", "where", "who", "when", "why", "how", "which",
        "all", "any", "some", "no", "not", "only", "than", "then", "now", "back",
        "let", "see", "go", "make", "need", "could", "should", "would", "can",
        "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
        "do", "does", "did", "doing", "s", "t", "ve", "ll", "m",'into', 'wouldn', 'end', 'then', 'viewers', 'didn', 'after', 'these', 'above', 'same',
    'actresses', 'does', 'screens', 'companies', 'minute', 'feelings', 'minutes', 'so',
    'second', 've', 'show', 'about', 'out', 'he', 'netflix', 'ours', 'itself', 'hours',
    'fan', 'hollywood', 'most', 'beginning', 'anybody', 'if', 'having', 'casted', 'setting',
    'day', 'few', 'opinion', 'location', 'again', 'of', 'actors', 'some', 'off', 'anyone',
    'story', 'scripts', 'under', 'ratings', 'shan', 'below', 'people', 'amazon', 'y',
    'everyone', 'her', 'fans', 'won', 'who', 'hadn', 'watching', 'but', 'them', 'person',
    'movie', 'view', 'soundtrack', 'not', 'actress', 'everybody', 'you', 'characters',
    'themselves', 'between', 'any', 'himself', 'days', 'll', 'an', 'plot', 'times',
    'performances', 'ending', 'has', 'production', 'all', 'company', 'wasn', 'me', 'she',
    'moment', 'doing', 'review', 'd', 'because', 'myself', 'mustn', 'have', 'screen',
    'places', 'episodes', 'on', 'fox', 'storyline', 'project', 'tomatoes', 'somebody',
    'can', 'through', 'rotten', 'o', 'cinema', 'this', 'roles', 'series', 'imdb', 'that',
    'no', 'marvel', 'disney', 'character', 'than', 'both', 'as', 'now', 'against', 'further',
    'ain', 'thoughts', 'in', 'views', 'shouldn', 'ourselves', 'what', 'when', 'don', 'mightn',
    'episode', 'rating', 'are', 'and', 'script', 'trailer', 'plus', 'haven', 'just', 'bollywood',
    'film', 'projects', 'down', 'set', 'our', 't', 'music', 'how', 'was', 'theme', 'hour',
    'being', 'where', 'place', 'score', 'dc', 'yourself', 'should', 'scenes', 'whom',
    'tomorrow', 'trailers', 'why', 'moments', 'nor', 'too', 'actor', 'they', 'time', 'hasn',
    'at', 'start', 'while', 'years', 'work', 'been', 'each', 'performance', 'had', 'his',
    'hers', 's', 'viewer', 'reviews', 'emotion', 'to', 'very', 'from', 'watch', 'its', 'am',
    'works', 'we', 'before', 'ma', 'their', 'my', 'here', 'the', 'm', 'more', 'up', 'which',
    'only', 'prime', 'weren', 'produced', 'it', 'scene', 'year', 'do', 'over', 'youtube',
    'such', 'isn', 're', 'theirs', 'seasons', 'is', 'be', 'or', 'once', 'opinions', 'yours',
    'feeling', 'seconds', 'other', 'were', 'own', 'night', 'with', 'for', 'doesn', 'there',
    'during', 'did', 'season', 'those', 'role', 'cast', 'locations', 'director', 'herself',
    'a', 'hulu', 'by', 'will', 'needn', 'nights', 'aren', 'your', 'themes', 'someone',
    'couldn', 'emotions', 'until', 'yesterday', 'i', 'today', 'yourselves', 'him', 'producer'
    ]

# Set up logging for error handling
logging.basicConfig(filename='app.log', level=logging.ERROR)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
app.config["CORS_HEADERS"] = "Content-Type"

# **Custom MSE Loss (if applicable to Keras models)**
def custom_mse_loss(y_true, y_pred):
    return K.mean(K.square(y_true - y_pred), axis=-1)

# **File Paths**
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"
MODEL_PATHS = {
    "cnn": "models/cnn.h5",
    # "gradient_boosting": "models/gradient_boosting.pkl",
    "logistic_regression": "models/logistic_regression.pkl",
    "mlp": "models/mlp.h5",
    "random_forest": "models/random_forest.pkl",
    "svm": "models/svm.pkl",
}

# **Load TF-IDF Vectorizer and Models**
models = {}
tfidf_vectorizer = None

try:
    tfidf_vectorizer = joblib.load(VECTORIZER_PATH)
    print("TF-IDF vectorizer loaded successfully.")
except Exception as e:
    logging.error(f"Error loading TF-IDF vectorizer: {e}")
    print(f"Error loading TF-IDF vectorizer: {e}")

for model_name, model_path in MODEL_PATHS.items():
    try:
        if model_path.endswith(".h5"):
            models[model_name] = load_model(model_path, custom_objects={'custom_mse_loss': custom_mse_loss})
        else:
            models[model_name] = joblib.load(model_path)
        print(f"{model_name} model loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading {model_name}: {e}")
        print(f"Error loading {model_name}: {e}")

# **Preprocessing Functions**
def clean_text(text):
    """Cleans the input text."""
    text = text.lower()
    text = re.sub(r'\([^()]*\)', '', text)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def remove_stop_words(text):
    """Removes stop words."""
    stop_words = set(stopwords.words('english'))
    additional_stop_words = ["um", "uh", "like", "actually", "basically", "you know"]
    stop_words.update(additional_stop_words)
    return ' '.join([word for word in text.split() if word not in stop_words])

def tokenize_text(text):
    """Tokenizes the text."""
    return word_tokenize(text)

def pos_tagging(tokens):
    """Performs POS tagging."""
    return pos_tag(tokens)

def extract_relevant_words(pos_tagged):
    """Extracts relevant words based on POS tagging."""
    return ' '.join([word for word, pos in pos_tagged if pos in ['NN', 'JJ']])

def filter_english_text(text):
    """Filters out non-English text."""
    try:
        return text if detect(text) == 'en' else ''
    except:
        return ''

def clean_process_and_extract_features(text, tfidf_vectorizer):
    """Cleans, processes, and extracts features for the model."""
    # Step 1: Clean the text
    text = clean_text(text)
    text = remove_stop_words(text)
    tokens = tokenize_text(text)
    pos_tags = pos_tagging(tokens)
    relevant_words = extract_relevant_words(pos_tags)

    # Step 2: Handle empty input and vectorize
    if not relevant_words.strip():
        tfidf_features = np.zeros((1, len(tfidf_vectorizer.get_feature_names_out())))
    else:
        tfidf_features = tfidf_vectorizer.transform([relevant_words]).toarray()

    print(f"TF-IDF Features Shape: {tfidf_features.shape}")

    # Step 3: Handle keyword features
    mo_features = np.array([1 if keyword in relevant_words else 0 for keyword in mo_keywords]).reshape(1, -1)
    behavior_features = np.array([1 if keyword in relevant_words else 0 for keyword in behavior_keywords]).reshape(1, -1)
    psychological_features = np.array([1 if keyword in relevant_words else 0 for keyword in psychological_keywords]).reshape(1, -1)

    # Step 4: Combine all features
    combined_features = np.hstack([tfidf_features, mo_features, behavior_features, psychological_features])
    print(f"Combined Features Shape: {combined_features.shape}")

    return combined_features

# **Flask Routes**
@app.route("/")
def home():
    return "Welcome to the AI Prediction API!"

@app.route("/predict", methods=["POST", "GET"])
def predict():
    if request.method == "POST":
        try:
            data = request.get_json()
            biography = data.get("biography", "").strip()

            if not biography:
                return jsonify({"error": "Biography text is required"}), 400

            # Process and extract features
            cleaned_input_features = clean_process_and_extract_features(biography, tfidf_vectorizer)
            print(f"Processed features shape: {cleaned_input_features.shape}")

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
                    confidence = float(np.max(prediction))
                    prediction = np.argmax(prediction)
                elif model_name == "svm":
                    decision_score = model.decision_function(cleaned_input_features)
                    confidence = float(1 / (1 + np.exp(-decision_score[0])))
                    prediction = model.predict(cleaned_input_features)
                    prediction = prediction[0]  # Ensure it's not an array
                elif hasattr(model, "predict_proba"):
                    prediction_probabilities = model.predict_proba(cleaned_input_features)
                    confidence = float(max(prediction_probabilities[0]))
                    prediction = model.predict(cleaned_input_features)
                    prediction = prediction[0]  # Ensure it's not an array
                else:
                    prediction = model.predict(cleaned_input_features)
                    prediction = prediction[0] if isinstance(prediction, (np.ndarray, list)) else prediction
                    confidence = 1  # Default confidence for non-probabilistic models

                # Store the confidence score
                confidence_scores[model_name] = confidence
                if confidence > highest_confidence:
                    highest_confidence = confidence
                    best_prediction = prediction

            # Return the best prediction
            return jsonify({
                "prediction": str(best_prediction) if best_prediction is not None else None,  # Convert to string for JSON compatibility
                "confidence": float(highest_confidence) if highest_confidence is not None else None,
                "confidence_scores": {model: float(conf) for model, conf in confidence_scores.items()}
            })

        except Exception as e:
            logging.error(f"Error during prediction: {e}")
            return jsonify({"error": str(e)}), 500
if __name__ == "__main__":
    app.run(debug=True, port=7070)