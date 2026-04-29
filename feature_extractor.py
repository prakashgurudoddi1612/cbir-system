"""
Feature Extractor — DenseNet121 with custom embedding head.
Loads the trained model and extracts 128-dim feature vectors.
"""

import os
import numpy as np
from PIL import Image
import io

# ── Constants ───────────────────────────────────────────────────────────────
IMG_SIZE   = 224
FEATURE_DIM = 128
MODEL_PATH = os.path.join(os.path.dirname(__file__), "data", "fold_3_best_model.h5")

_model = None   # lazy-loaded singleton


def _load_model():
    """Load TF/Keras model once, cache in module-level variable."""
    global _model
    if _model is not None:
        return _model

    try:
        import tensorflow as tf
        tf.get_logger().setLevel("ERROR")
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

        model = tf.keras.models.load_model(MODEL_PATH, compile=False)

        # Build a sub-model that stops at the embedding/penultimate layer.
        # We try common layer names; fall back to second-to-last layer.
        candidate_names = [
            "embedding", "features", "dense_features",
            "global_average_pooling2d", "flatten", "dense"
        ]
        output_layer = None
        for name in candidate_names:
            try:
                output_layer = model.get_layer(name).output
                break
            except ValueError:
                pass

        if output_layer is None:
            # Use the last layer whose output is a 1-D feature vector
            for layer in reversed(model.layers):
                shape = layer.output_shape
                if isinstance(shape, tuple) and len(shape) == 2:
                    output_layer = layer.output
                    break

        if output_layer is None:
            # Absolute fallback: second-to-last layer
            output_layer = model.layers[-2].output

        _model = tf.keras.Model(inputs=model.input, outputs=output_layer)
        print(f"[FeatureExtractor] Model loaded. Output shape: {_model.output_shape}")

    except Exception as e:
        print(f"[FeatureExtractor] WARNING: Could not load TF model — {e}")
        print("[FeatureExtractor] Falling back to random projection (demo mode).")
        _model = None

    return _model


def preprocess_image(image_input):
    """
    Accept a file path, bytes, PIL Image, or file-like object.
    Returns a (1, 224, 224, 3) float32 numpy array ready for the model.
    """
    if isinstance(image_input, (str, os.PathLike)):
        img = Image.open(image_input)
    elif isinstance(image_input, bytes):
        img = Image.open(io.BytesIO(image_input))
    elif hasattr(image_input, "read"):          # file-like
        img = Image.open(image_input)
    elif isinstance(image_input, Image.Image):
        img = image_input
    else:
        raise ValueError(f"Unsupported input type: {type(image_input)}")

    img = img.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0

    # ImageNet-style normalisation (used during training)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr  = (arr - mean) / std

    return np.expand_dims(arr, axis=0)          # (1, 224, 224, 3)


def extract_features(image_input):
    """
    Extract a 1-D feature vector (numpy array, float32) from an image.
    Falls back to a deterministic random projection when the model is unavailable.
    """
    arr = preprocess_image(image_input)
    model = _load_model()

    if model is not None:
        vec = model.predict(arr, verbose=0)[0]
    else:
        # Reproducible pseudo-features for demo / testing without GPU
        seed = int(arr.sum() * 1e4) % (2 ** 31)
        rng  = np.random.RandomState(seed)
        vec  = rng.randn(FEATURE_DIM).astype(np.float32)

    # L2 normalise
    norm = np.linalg.norm(vec)
    if norm > 1e-8:
        vec = vec / norm
    return vec.astype(np.float32)


def model_available():
    return os.path.exists(MODEL_PATH)
