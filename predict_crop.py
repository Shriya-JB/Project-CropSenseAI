"""predict_crop.py — CropSenseAI command-line prediction utility.

Loads a serialized crop classifier, preprocesses an input image using the
same HOG + color-histogram pipeline used during training, and returns the
predicted crop name: cotton, maize, rice, or wheat.

FIXES applied:
  - CRITICAL: Fixed indentation bug on `image = image.reshape(1, -1)` line
    (was indented one level too far, making it unreachable / part of wrong block)
  - Uses extract_features() from feature_extraction.py instead of raw reshape
    so inference matches training exactly (no data-leakage risk)
  - Default model path updated to data/models/crop_classifier.pkl

Usage:
    python predict_crop.py path/to/image.jpg
    python predict_crop.py path/to/image.jpg --model data/models/crop_classifier.pkl --verbose
"""

import argparse
import logging
from pathlib import Path
from typing import List

import cv2
import joblib
import numpy as np

from cropsenseai.crop_classification.feature_extraction import extract_features
from cropsenseai.crop_classification.class_mapping import load_class_mapping, get_index_to_class

logger = logging.getLogger(__name__)

IMAGE_SIZE = (128, 128)
SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


def load_model(model_path: Path) -> object:
    """Load a pickled crop classifier model."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    try:
        model = joblib.load(model_path)
    except Exception as exc:
        raise RuntimeError(f"Failed to load model from {model_path}: {exc}") from exc
    logger.info("Loaded model from %s", model_path)
    return model


def validate_image_path(image_path: Path) -> None:
    """Validate that the image path exists and has a supported extension."""
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    if image_path.suffix.lower() not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported image format '{image_path.suffix}'. "
            f"Supported: {sorted(SUPPORTED_FORMATS)}"
        )


def preprocess_image(image_path: Path) -> np.ndarray:
    """
    Load image, convert to RGB, resize, normalize to [0,1], extract features.

    FIXED: uses extract_features() — same pipeline as training — instead of
           raw pixel flattening.  Also fixed the indentation bug that placed
           the reshape call inside the wrong scope.
    """
    validate_image_path(image_path)

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Unable to read image file: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    image = image.astype(np.float32) / 255.0

    # Extract HOG + color histogram features (matches training pipeline)
    features = extract_features(image).reshape(1, -1)  # FIXED: correct indentation
    return features


def decode_prediction(prediction, idx_to_class: dict) -> str:
    """Decode a model prediction array or scalar into a crop name."""
    if isinstance(prediction, np.ndarray):
        if prediction.ndim == 2:
            label_index = int(np.argmax(prediction, axis=1)[0])
        else:
            label_index = int(prediction.ravel()[0])
    else:
        try:
            label_index = int(prediction)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Unexpected prediction format: {prediction}") from exc

    if label_index not in idx_to_class:
        raise ValueError(f"Predicted label index {label_index} not in class mapping")

    return idx_to_class[label_index]


def predict_crop(
    image_path: str,
    model_path: str = "data/models/crop_classifier.pkl",
    mapping_path: str = "data/models/class_mapping.json",
) -> str:
    """Predict the crop type for the given image. Returns the crop name string."""
    model = load_model(Path(model_path))
    class_map = load_class_mapping(Path(mapping_path))
    idx_to_class = get_index_to_class(class_map)

    features = preprocess_image(Path(image_path))

    try:
        prediction = model.predict(features)
    except AttributeError as exc:
        raise RuntimeError(
            "Loaded model does not support predict(). "
            "Ensure the model is a scikit-learn compatible estimator."
        ) from exc
    except Exception as exc:
        raise RuntimeError(f"Prediction failed: {exc}") from exc

    return decode_prediction(prediction, idx_to_class)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict crop type from a single image using a saved classifier."
    )
    parser.add_argument("image_path", type=str, help="Path to the crop image.")
    parser.add_argument(
        "--model",
        type=str,
        default="data/models/crop_classifier.pkl",
        help="Path to the serialized classifier file.",
    )
    parser.add_argument(
        "--mapping",
        type=str,
        default="data/models/class_mapping.json",
        help="Path to the class mapping JSON file.",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose logging."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        crop_name = predict_crop(args.image_path, args.model, args.mapping)
        print(f"Predicted crop: {crop_name}")
    except Exception as exc:
        logger.error("Prediction failed: %s", exc)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
