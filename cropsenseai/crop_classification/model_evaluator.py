"""
cropsenseai/crop_classification/model_evaluator.py

Loads a trained model and evaluates it on a test split.
Saves metrics to data/models/model_metadata.json.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split

from cropsenseai.core.data_preprocessing import CropImagePreprocessor
from cropsenseai.crop_classification.feature_extraction import extract_features_batch
from cropsenseai.crop_classification.class_mapping import load_class_mapping, get_index_to_class

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = Path("data/models/crop_classifier.pkl")
DEFAULT_METADATA_PATH = Path("data/models/model_metadata.json")
DEFAULT_RAW_DATA = Path("data/raw")


def evaluate(
    model_path: Path = DEFAULT_MODEL_PATH,
    raw_data_dir: Path = DEFAULT_RAW_DATA,
    test_size: float = 0.2,
    random_state: int = 42,
    metadata_output_path: Path = DEFAULT_METADATA_PATH,
) -> Dict:
    """
    Load model, recreate test split (same seed as training), evaluate.

    Returns a dict with accuracy, per-class F1, confusion matrix.
    Also writes the dict to metadata_output_path as JSON.
    """
    # Load model
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    pipeline = joblib.load(model_path)
    logger.info("Model loaded from %s", model_path)

    # Load class mapping
    class_map = load_class_mapping()
    idx_to_class = get_index_to_class(class_map)
    class_names = [idx_to_class[i] for i in sorted(idx_to_class)]

    # Load data and extract features (must use same seed as training)
    preprocessor = CropImagePreprocessor(str(raw_data_dir))
    X_images, y, _ = preprocessor.load_and_preprocess(normalize=True)
    X_features = extract_features_batch(X_images)

    _, X_test, _, y_test = train_test_split(
        X_features, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # Predict
    y_pred = pipeline.predict(X_test)

    # Metrics
    accuracy = float(accuracy_score(y_test, y_pred))
    macro_f1 = float(f1_score(y_test, y_pred, average="macro"))
    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    cm = confusion_matrix(y_test, y_pred).tolist()

    metadata = {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "test_samples": int(len(y_test)),
        "class_names": class_names,
        "classification_report": report,
        "confusion_matrix": cm,
        "model_path": str(model_path),
    }

    # Save
    metadata_output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Accuracy: %.4f | Macro F1: %.4f", accuracy, macro_f1)
    logger.info("Metrics saved to %s", metadata_output_path)

    print(f"\n{'='*50}")
    print(f"  Accuracy : {accuracy:.4f}")
    print(f"  Macro F1 : {macro_f1:.4f}")
    print(f"  Test set : {len(y_test)} samples")
    print(f"{'='*50}\n")
    print(classification_report(y_test, y_pred, target_names=class_names))

    return metadata

if __name__ == "__main__":
    import argparse
    import logging as _logging

    _logging.basicConfig(
        level=_logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    _parser = argparse.ArgumentParser(description="Evaluate the CropSenseAI crop classifier.")
    _parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    _parser.add_argument("--data", type=Path, default=DEFAULT_RAW_DATA)
    _parser.add_argument("--output", type=Path, default=DEFAULT_METADATA_PATH)
    _parser.add_argument("--test-size", type=float, default=0.2)
    _args = _parser.parse_args()

    evaluate(
        model_path=_args.model,
        raw_data_dir=_args.data,
        test_size=_args.test_size,
        metadata_output_path=_args.output,
    )
