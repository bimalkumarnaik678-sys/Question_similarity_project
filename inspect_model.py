from pathlib import Path
import pickle
import joblib
import json
import numpy as np


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
# HELPER
# ============================================================

def load_pickle(path):

    try:
        return joblib.load(path)

    except Exception:

        with open(path, "rb") as f:
            return pickle.load(f)


# ============================================================
# FEATURE NAMES
# ============================================================

print("=" * 80)
print("FEATURE NAMES")
print("=" * 80)

feature_names_path = MODEL_DIR / "feature_names.pkl"

feature_names = load_pickle(
    feature_names_path
)

print("Count:", len(feature_names))

for i, name in enumerate(feature_names):

    print(
        f"{i:02d} -> {name}"
    )


# ============================================================
# ENGINEERED FEATURES
# ============================================================

print("\n" + "=" * 80)
print("ENGINEERED FEATURES")
print("=" * 80)

engineered_path = (
    FEATURE_DIR /
    "engineered_features.pkl"
)

engineered = load_pickle(
    engineered_path
)

print(
    "Object type:",
    type(engineered)
)

if isinstance(engineered, dict):

    print("\nKeys:")

    for key in engineered:

        value = engineered[key]

        print(
            f"\nKEY: {key}"
        )

        print(
            "TYPE:",
            type(value)
        )

        if hasattr(value, "shape"):

            print(
                "SHAPE:",
                value.shape
            )

        elif isinstance(value, list):

            print(
                "LENGTH:",
                len(value)
            )

        elif isinstance(value, dict):

            print(
                "DICT KEYS:",
                list(value.keys())
            )

        else:

            print(
                "VALUE:",
                value
            )


# ============================================================
# FEATURE CONFIG
# ============================================================

print("\n" + "=" * 80)
print("FEATURE CONFIG")
print("=" * 80)

config_path = (
    FEATURE_DIR /
    "feature_config.json"
)

with open(
    config_path,
    "r",
    encoding="utf-8"
) as f:

    config = json.load(f)

print(
    json.dumps(
        config,
        indent=4
    )
)


# ============================================================
# HANDCRAFTED FEATURES
# ============================================================

print("\n" + "=" * 80)
print("HANDCRAFTED FEATURES")
print("=" * 80)

hand_path = (
    FEATURE_DIR /
    "handcrafted_feature_names.json"
)

with open(
    hand_path,
    "r",
    encoding="utf-8"
) as f:

    hand_names = json.load(f)

print(
    "Count:",
    len(hand_names)
)

for i, name in enumerate(hand_names):

    print(
        f"{i:02d} -> {name}"
    )


# ============================================================
# NPZ FILES
# ============================================================

print("\n" + "=" * 80)
print("NPZ FEATURES")
print("=" * 80)

for filename in [
    "basic_features.npz",
    "advanced_features.npz",
    "advanced_features_scaled.npz"
]:

    path = FEATURE_DIR / filename

    print(
        f"\n--- {filename} ---"
    )

    data = np.load(
        path,
        allow_pickle=True
    )

    print(
        "Keys:",
        data.files
    )

    for key in data.files:

        arr = data[key]

        print(
            f"{key}: "
            f"shape={arr.shape}, "
            f"dtype={arr.dtype}"
        )


# ============================================================
# SEMANTIC CONFIG
# ============================================================

print("\n" + "=" * 80)
print("SEMANTIC CONFIG")
print("=" * 80)

semantic_path = (
    FEATURE_DIR /
    "semantic_model_config.json"
)

with open(
    semantic_path,
    "r",
    encoding="utf-8"
) as f:

    semantic_config = json.load(f)

print(
    json.dumps(
        semantic_config,
        indent=4
    )
)


print("\n" + "=" * 80)
print("DIAGNOSTIC FINISHED")
print("=" * 80)
