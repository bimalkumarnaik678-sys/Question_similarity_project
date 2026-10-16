# ============================================================
# QUESTION SIMILARITY WEB APPLICATION
# ============================================================

from flask import Flask, render_template, request, jsonify

from pathlib import Path
import json
import pickle
import re

import joblib
import numpy as np

from sklearn.preprocessing import normalize
from xgboost import XGBClassifier


# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(__name__)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(
    r"C:\Users\bimal\OneDrive\Desktop\bibhu sir project"
    r"\question_similarity_project"
)

MODEL_DIR = BASE_DIR / "model"

FEATURE_DIR = MODEL_DIR / "feature_artifacts"


# ============================================================
# MODEL ARTIFACTS
# ============================================================

MODEL_PATH = (
    MODEL_DIR /
    "question_similarity_xgboost.json"
)

THRESHOLD_PATH = (
    MODEL_DIR /
    "similarity_threshold.pkl"
)

FEATURE_NAMES_PATH = (
    MODEL_DIR /
    "feature_names.pkl"
)

WORD_TFIDF_PATH = (
    FEATURE_DIR /
    "word_tfidf.pkl"
)

CHAR_TFIDF_PATH = (
    FEATURE_DIR /
    "char_tfidf.pkl"
)

FEATURE_SCALER_PATH = (
    FEATURE_DIR /
    "feature_scaler.pkl"
)

FEATURE_CONFIG_PATH = (
    FEATURE_DIR /
    "feature_config.json"
)

HANDCRAFTED_NAMES_PATH = (
    FEATURE_DIR /
    "handcrafted_feature_names.json"
)

SEMANTIC_CONFIG_PATH = (
    FEATURE_DIR /
    "semantic_model_config.json"
)


# ============================================================
# GLOBAL VARIABLES
# ============================================================

model = None

similarity_threshold = 0.39

word_tfidf = None

char_tfidf = None

feature_scaler = None

feature_names = None

feature_config = None

handcrafted_feature_names = None

semantic_model = None

MODEL_LOADED = False

LOAD_ERROR = None


# ============================================================
# EXPECTED 15 TRAINING FEATURES
# ============================================================

EXPECTED_FEATURES = [
    "q1_char_count",
    "q2_char_count",
    "q1_word_count",
    "q2_word_count",
    "char_length_difference",
    "word_length_difference",
    "char_length_ratio",
    "word_length_ratio",
    "common_token_count",
    "jaccard_similarity",
    "overlap_coefficient",
    "common_word_ratio",
    "exact_match",
    "word_tfidf_cosine",
    "char_tfidf_cosine"
]


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    if text is None:
        return ""

    text = str(text)

    text = text.lower()

    # URLs
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text
    )

    # Keep letters, numbers and spaces
    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    # Multiple spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# TOKENIZATION
# ============================================================

def get_tokens(text):

    cleaned = clean_text(text)

    if not cleaned:
        return []

    return cleaned.split()


# ============================================================
# SAFE DIVISION
# ============================================================

def safe_divide(a, b):

    if b == 0:
        return 0.0

    return float(a) / float(b)


# ============================================================
# BASIC TEXT FEATURES
# ============================================================

def create_basic_text_features(
    question1,
    question2
):

    q1 = clean_text(question1)

    q2 = clean_text(question2)

    tokens1 = get_tokens(question1)

    tokens2 = get_tokens(question2)

    # --------------------------------------------------------
    # Character counts
    # --------------------------------------------------------

    q1_char_count = len(q1)

    q2_char_count = len(q2)

    # --------------------------------------------------------
    # Word counts
    # --------------------------------------------------------

    q1_word_count = len(tokens1)

    q2_word_count = len(tokens2)

    # --------------------------------------------------------
    # Differences
    # --------------------------------------------------------

    char_length_difference = abs(
        q1_char_count -
        q2_char_count
    )

    word_length_difference = abs(
        q1_word_count -
        q2_word_count
    )

    # --------------------------------------------------------
    # Ratios
    # --------------------------------------------------------

    char_length_ratio = safe_divide(
        min(
            q1_char_count,
            q2_char_count
        ),
        max(
            q1_char_count,
            q2_char_count,
            1
        )
    )

    word_length_ratio = safe_divide(
        min(
            q1_word_count,
            q2_word_count
        ),
        max(
            q1_word_count,
            q2_word_count,
            1
        )
    )

    # --------------------------------------------------------
    # Sets
    # --------------------------------------------------------

    set1 = set(tokens1)

    set2 = set(tokens2)

    intersection = set1.intersection(set2)

    union = set1.union(set2)

    # --------------------------------------------------------
    # Common token count
    # --------------------------------------------------------

    common_token_count = len(
        intersection
    )

    # --------------------------------------------------------
    # Jaccard similarity
    # --------------------------------------------------------

    if len(union) == 0:

        jaccard_similarity = 1.0

    else:

        jaccard_similarity = (
            len(intersection) /
            len(union)
        )

    # --------------------------------------------------------
    # Overlap coefficient
    # --------------------------------------------------------

    smallest_set = min(
        len(set1),
        len(set2)
    )

    if smallest_set == 0:

        overlap_coefficient = 0.0

    else:

        overlap_coefficient = (
            len(intersection) /
            smallest_set
        )

    # --------------------------------------------------------
    # Common word ratio
    # --------------------------------------------------------

    common_word_ratio = safe_divide(
        len(intersection),
        max(
            len(tokens1),
            len(tokens2),
            1
        )
    )

    # --------------------------------------------------------
    # Exact match
    # --------------------------------------------------------

    exact_match = int(
        q1 == q2 and
        q1 != ""
    )

    # --------------------------------------------------------
    # Return
    # --------------------------------------------------------

    return {

        "q1_char_count":
            q1_char_count,

        "q2_char_count":
            q2_char_count,

        "q1_word_count":
            q1_word_count,

        "q2_word_count":
            q2_word_count,

        "char_length_difference":
            char_length_difference,

        "word_length_difference":
            word_length_difference,

        "char_length_ratio":
            char_length_ratio,

        "word_length_ratio":
            word_length_ratio,

        "common_token_count":
            common_token_count,

        "jaccard_similarity":
            jaccard_similarity,

        "overlap_coefficient":
            overlap_coefficient,

        "common_word_ratio":
            common_word_ratio,

        "exact_match":
            exact_match
    }


# ============================================================
# TF-IDF COSINE SIMILARITY
# ============================================================

def calculate_tfidf_cosine(
    question1,
    question2
):

    if word_tfidf is None:

        raise RuntimeError(
            "Word TF-IDF model is not loaded."
        )

    if char_tfidf is None:

        raise RuntimeError(
            "Character TF-IDF model is not loaded."
        )

    q1 = clean_text(question1)

    q2 = clean_text(question2)

    # ========================================================
    # WORD TF-IDF
    # ========================================================

    q1_word = word_tfidf.transform(
        [q1]
    )

    q2_word = word_tfidf.transform(
        [q2]
    )

    q1_word = normalize(
        q1_word,
        norm="l2"
    )

    q2_word = normalize(
        q2_word,
        norm="l2"
    )

    word_tfidf_cosine = float(
        q1_word.multiply(
            q2_word
        ).sum()
    )

    # ========================================================
    # CHARACTER TF-IDF
    # ========================================================

    q1_char = char_tfidf.transform(
        [q1]
    )

    q2_char = char_tfidf.transform(
        [q2]
    )

    q1_char = normalize(
        q1_char,
        norm="l2"
    )

    q2_char = normalize(
        q2_char,
        norm="l2"
    )

    char_tfidf_cosine = float(
        q1_char.multiply(
            q2_char
        ).sum()
    )

    return (
        word_tfidf_cosine,
        char_tfidf_cosine
    )


# ============================================================
# CREATE EXACT 15 FEATURES
# ============================================================

def create_feature_vector(
    question1,
    question2
):

    # --------------------------------------------------------
    # Basic features
    # --------------------------------------------------------

    features = create_basic_text_features(
        question1,
        question2
    )

    # --------------------------------------------------------
    # TF-IDF
    # --------------------------------------------------------

    (
        word_tfidf_cosine,
        char_tfidf_cosine
    ) = calculate_tfidf_cosine(
        question1,
        question2
    )

    features[
        "word_tfidf_cosine"
    ] = word_tfidf_cosine

    features[
        "char_tfidf_cosine"
    ] = char_tfidf_cosine

    # ========================================================
    # IMPORTANT
    # ========================================================
    #
    # Do NOT add semantic features here.
    #
    # The trained XGBoost model has exactly 15 features.
    #
    # ========================================================

    # --------------------------------------------------------
    # Validate all expected features
    # --------------------------------------------------------

    missing = []

    for feature in EXPECTED_FEATURES:

        if feature not in features:

            missing.append(feature)

    if missing:

        raise RuntimeError(
            "Missing feature(s):\n" +
            "\n".join(missing)
        )

    # --------------------------------------------------------
    # Create vector in EXACT training order
    # --------------------------------------------------------

    vector = np.array(
        [
            features[name]
            for name in EXPECTED_FEATURES
        ],
        dtype=np.float32
    )

    vector = vector.reshape(
        1,
        -1
    )

    return vector


# ============================================================
# LOAD PICKLE / JOBLIB
# ============================================================

def load_pickle_artifact(path):

    path = Path(path)

    if not path.exists():

        raise FileNotFoundError(
            f"Artifact not found:\n{path}"
        )

    # First joblib
    try:

        return joblib.load(path)

    except Exception:

        pass

    # Then pickle
    try:

        with open(
            path,
            "rb"
        ) as f:

            return pickle.load(f)

    except Exception as e:

        raise RuntimeError(
            f"Unable to load artifact:\n{path}\n\n{e}"
        )


# ============================================================
# LOAD SEMANTIC MODEL
# ============================================================

def load_semantic_model():

    global semantic_model

    try:

        from sentence_transformers import (
            SentenceTransformer
        )

    except ImportError:

        print(
            "sentence-transformers is not installed."
        )

        return None

    if not SEMANTIC_CONFIG_PATH.exists():

        print(
            "Semantic configuration not found."
        )

        return None

    with open(
        SEMANTIC_CONFIG_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        config = json.load(f)

    print(
        "Semantic configuration:"
    )

    print(config)

    model_name = config.get(
        "model_name"
    )

    if not model_name:

        print(
            "Semantic model name not found."
        )

        return None

    print(
        f"Loading semantic model: {model_name}"
    )

    semantic_model = SentenceTransformer(
        model_name
    )

    print(
        "✓ Semantic model loaded."
    )

    return semantic_model


# ============================================================
# LOAD ALL MODEL ARTIFACTS
# ============================================================

def load_artifacts():

    global model
    global similarity_threshold
    global word_tfidf
    global char_tfidf
    global feature_scaler
    global feature_names
    global feature_config
    global handcrafted_feature_names
    global MODEL_LOADED
    global LOAD_ERROR

    try:

        print("=" * 60)
        print(
            "QUESTION SIMILARITY WEB APPLICATION"
        )
        print("=" * 60)

        print()
        print("=" * 60)
        print(
            "Loading Question Similarity Model"
        )
        print("=" * 60)

        # ====================================================
        # XGBOOST
        # ====================================================

        if not MODEL_PATH.exists():

            raise FileNotFoundError(
                f"XGBoost model not found:\n"
                f"{MODEL_PATH}"
            )

        model = XGBClassifier()

        model.load_model(
            str(MODEL_PATH)
        )

        print(
            "✓ XGBoost model loaded."
        )

        # ====================================================
        # THRESHOLD
        # ====================================================

        similarity_threshold = float(
            load_pickle_artifact(
                THRESHOLD_PATH
            )
        )

        print(
            "✓ Threshold:",
            similarity_threshold
        )

        # ====================================================
        # FEATURE NAMES
        # ====================================================

        feature_names = load_pickle_artifact(
            FEATURE_NAMES_PATH
        )

        print(
            "✓ Feature names loaded:",
            len(feature_names)
        )

        # ====================================================
        # WORD TF-IDF
        # ====================================================

        word_tfidf = load_pickle_artifact(
            WORD_TFIDF_PATH
        )

        print(
            "✓ Word TF-IDF loaded."
        )

        # ====================================================
        # CHARACTER TF-IDF
        # ====================================================

        char_tfidf = load_pickle_artifact(
            CHAR_TFIDF_PATH
        )

        print(
            "✓ Character TF-IDF loaded."
        )

        # ====================================================
        # FEATURE SCALER
        # ====================================================

        feature_scaler = load_pickle_artifact(
            FEATURE_SCALER_PATH
        )

        print(
            "✓ Feature scaler loaded."
        )

        # ====================================================
        # FEATURE CONFIG
        # ====================================================

        if not FEATURE_CONFIG_PATH.exists():

            raise FileNotFoundError(
                f"Feature config not found:\n"
                f"{FEATURE_CONFIG_PATH}"
            )

        with open(
            FEATURE_CONFIG_PATH,
            "r",
            encoding="utf-8"
        ) as f:

            feature_config = json.load(f)

        print(
            "✓ Feature configuration loaded."
        )

        # ====================================================
        # HANDCRAFTED FEATURE NAMES
        # ====================================================

        if HANDCRAFTED_NAMES_PATH.exists():

            with open(
                HANDCRAFTED_NAMES_PATH,
                "r",
                encoding="utf-8"
            ) as f:

                handcrafted_feature_names = (
                    json.load(f)
                )

            print(
                "✓ Handcrafted feature names loaded."
            )

        # ====================================================
        # SEMANTIC MODEL
        # ====================================================

        load_semantic_model()

        # ====================================================
        # VALIDATE MODEL
        # ====================================================

        model_feature_count = (
            model.get_booster()
            .num_features()
        )

        print(
            "Model feature count:",
            model_feature_count
        )

        print(
            "Expected application feature count:",
            len(EXPECTED_FEATURES)
        )

        if model_feature_count != len(
            EXPECTED_FEATURES
        ):

            raise RuntimeError(
                f"\nFeature count mismatch.\n"
                f"XGBoost expects "
                f"{model_feature_count} features.\n"
                f"Application creates "
                f"{len(EXPECTED_FEATURES)} features."
            )

        # ====================================================
        # VALIDATE FEATURE CONFIG
        # ====================================================

        configured_basic = (
            feature_config.get(
                "basic_features",
                []
            )
        )

        if configured_basic != EXPECTED_FEATURES:

            print(
                "\nWARNING:"
            )

            print(
                "feature_config basic feature order "
                "is different from EXPECTED_FEATURES."
            )

            print(
                "Configured:"
            )

            print(
                configured_basic
            )

        MODEL_LOADED = True

        LOAD_ERROR = None

        print("=" * 60)
        print(
            "✓ MODEL LOADED SUCCESSFULLY"
        )
        print("=" * 60)

    except Exception as e:

        MODEL_LOADED = False

        LOAD_ERROR = str(e)

        print("=" * 60)
        print(
            "MODEL LOADING FAILED"
        )
        print(e)
        print("=" * 60)


# ============================================================
# PREDICTION
# ============================================================

def predict_similarity(
    question1,
    question2
):

    if not MODEL_LOADED:

        raise RuntimeError(
            LOAD_ERROR or
            "Model is not loaded."
        )

    # --------------------------------------------------------
    # Input validation
    # --------------------------------------------------------

    if question1 is None:

        raise ValueError(
            "Question 1 is missing."
        )

    if question2 is None:

        raise ValueError(
            "Question 2 is missing."
        )

    question1 = str(
        question1
    ).strip()

    question2 = str(
        question2
    ).strip()

    if not question1:

        raise ValueError(
            "Question 1 cannot be empty."
        )

    if not question2:

        raise ValueError(
            "Question 2 cannot be empty."
        )

    # --------------------------------------------------------
    # Create EXACT 15 features
    # --------------------------------------------------------

    X = create_feature_vector(
        question1,
        question2
    )

    print("\nPrediction features:")

    for i, value in enumerate(
        X[0]
    ):

        print(
            f"{i:02d} "
            f"{EXPECTED_FEATURES[i]:30s} "
            f"{value:.6f}"
        )

    # --------------------------------------------------------
    # Validate shape
    # --------------------------------------------------------

    expected_count = (
        model.get_booster()
        .num_features()
    )

    actual_count = X.shape[1]

    if actual_count != expected_count:

        raise RuntimeError(
            f"Feature mismatch: "
            f"model expects {expected_count}, "
            f"but application generated "
            f"{actual_count}."
        )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    probability = float(
        model.predict_proba(X)[0][1]
    )

    # --------------------------------------------------------
    # Threshold
    # --------------------------------------------------------

    is_duplicate = (
        probability >=
        similarity_threshold
    )

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    if is_duplicate:

        result = (
            "Duplicate / Similar"
        )

    else:

        result = (
            "Not Duplicate / Different"
        )

    return {

        "probability":
            round(
                probability,
                6
            ),

        "similarity_percentage":
            round(
                probability * 100,
                2
            ),

        "threshold":
            round(
                similarity_threshold * 100,
                2
            ),

        "result":
            result,

        "is_duplicate":
            bool(is_duplicate)
    }


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html",
        model_loaded=MODEL_LOADED
    )


# ============================================================
# PREDICTION API
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    try:

        data = request.get_json(
            silent=True
        )

        if data is None:

            return jsonify({
                "success": False,
                "error":
                    "Invalid JSON request."
            }), 400

        question1 = data.get(
            "question1",
            ""
        )

        question2 = data.get(
            "question2",
            ""
        )

        result = predict_similarity(
            question1,
            question2
        )

        return jsonify({

            "success": True,

            **result

        })

    except Exception as e:

        print(
            "\nPrediction error:"
        )

        print(e)

        return jsonify({

            "success": False,

            "error": str(e)

        }), 500


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route(
    "/health",
    methods=["GET"]
)
def health():

    return jsonify({

        "status":
            "ok",

        "model_loaded":
            MODEL_LOADED,

        "model_features":
            (
                model.get_booster()
                .num_features()
                if model is not None
                else None
            ),

        "application_features":
            len(EXPECTED_FEATURES),

        "threshold":
            similarity_threshold
    })


# ============================================================
# APPLICATION START
# ============================================================

if __name__ == "__main__":

    load_artifacts()

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )