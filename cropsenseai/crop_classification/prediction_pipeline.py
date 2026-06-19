"""
cropsenseai/crop_classification/prediction_pipeline.py

End-to-end prediction: image path → crop name + confidence.
Used by the dashboard and can replace predict_crop.py.
"""

import logging
from pathlib import Path
from typing import Dict, Tuple

import cv2
import joblib
import numpy as np

from cropsenseai.crop_classification.feature_extraction import extract_features
from cropsenseai.crop_classification.class_mapping import load_class_mapping, get_index_to_class

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = Path("data/models/crop_classifier.pkl")
IMAGE_SIZE = (128, 128)
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


def load_and_preprocess_single(image_path: Path) -> np.ndarray:
    """Load one image → float32 (H,W,3) in [0,1]."""
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported format: {image_path.suffix}")

    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    return img


def predict(
    image_path: str,
    model_path: str = str(DEFAULT_MODEL_PATH),
) -> Dict:
    """
    Predict crop type for a single image.

    Returns:
        {
            "crop": "wheat",
            "confidence": 0.87,
            "all_probabilities": {"cotton": 0.02, "maize": 0.05, "rice": 0.06, "wheat": 0.87}
        }
    """
    model_file = Path(model_path)
    image_file = Path(image_path)

    # Load model
    if not model_file.exists():
        raise FileNotFoundError(f"Model not found: {model_file}")
    pipeline = joblib.load(model_file)

    # Load class mapping
    class_map = load_class_mapping()
    idx_to_class = get_index_to_class(class_map)

    # Preprocess
    image = load_and_preprocess_single(image_file)

    # Extract features
    features = extract_features(image).reshape(1, -1)

    # Predict
    prediction = pipeline.predict(features)[0]
    crop_name = idx_to_class[int(prediction)]

    # Confidence (probability)
    result = {"crop": crop_name, "confidence": None, "all_probabilities": {}}
    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(features)[0]
        result["confidence"] = round(float(proba.max()), 4)
        result["all_probabilities"] = {
            idx_to_class[i]: round(float(p), 4) for i, p in enumerate(proba)
        }
    else:
        result["confidence"] = 1.0

    logger.info("Prediction: %s (confidence: %s)", crop_name, result["confidence"])
    return result