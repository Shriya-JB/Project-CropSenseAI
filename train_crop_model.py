"""train_crop_model.py — Train the CropSenseAI crop classification model.

FIXES applied:
  - Moved feature-extraction imports AFTER stdlib imports (PEP 8 / runtime order)
  - Fixed indentation error in preprocess_image() in original predict_crop.py
  - `prepare_dataset()` uses HOG+color histogram features, NOT raw pixel flattening
  - `train_and_evaluate()` prints class names in the classification report
  - `save_class_mapping()` is called so prediction always uses matching labels

Usage:
    python train_crop_model.py
    python train_crop_model.py --raw-data data/raw --output data/models/crop_classifier.pkl
"""

import argparse
import logging
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Project imports — must come after stdlib / third-party
from cropsenseai.core.data_preprocessing import CropImagePreprocessor
from cropsenseai.crop_classification.feature_extraction import extract_features_batch
from cropsenseai.crop_classification.class_mapping import save_class_mapping

MODEL_FILENAME = "crop_classifier.pkl"
DEFAULT_RAW_DATA_DIR = Path("data") / "raw"
DEFAULT_MODEL_DIR = Path("data") / "models"


def build_training_pipeline() -> Pipeline:
    """Build a scikit-learn pipeline: StandardScaler → RandomForestClassifier."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=15,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def prepare_dataset(raw_data_path: Path) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Load images → extract HOG + color histogram features → return feature matrix.

    Returns:
        X             : (N, feature_dim) float32
        y             : (N,) int64 labels
        class_mapping : {class_name: label_index}
    """
    preprocessor = CropImagePreprocessor(str(raw_data_path))
    X_images, y, class_mapping = preprocessor.load_and_preprocess(normalize=True)
    logging.info("Images loaded: %s, Labels: %s", X_images.shape, y.shape)

    # Feature extraction — replaces raw pixel flattening (huge quality improvement)
    X = extract_features_batch(X_images)
    logging.info("Feature matrix: %d samples × %d features", X.shape[0], X.shape[1])
    return X, y, class_mapping


def train_and_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    class_mapping: dict,
    model_output_path: Path,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Pipeline:
    """Train the pipeline and evaluate on a held-out test set."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,          # balanced split across classes
    )
    logging.info("Train: %d samples | Test: %d samples", len(X_train), len(X_test))

    pipeline = build_training_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    accuracy = float(np.mean(y_pred == y_test))

    # FIXED: pass target_names so report shows class names instead of 0/1/2/3
    idx_to_class = {v: k for k, v in class_mapping.items()}
    class_names = [idx_to_class[i] for i in sorted(idx_to_class)]

    logging.info("Training complete.")
    logging.info("Accuracy : %.4f", accuracy)
    logging.info(
        "Classification report:\n%s",
        classification_report(y_test, y_pred, target_names=class_names),
    )
    logging.info("Confusion matrix:\n%s", confusion_matrix(y_test, y_pred))

    # Save model
    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_output_path)
    logging.info("Saved trained model to %s", model_output_path)

    # FIXED: always save class mapping alongside the model
    save_class_mapping(class_mapping)

    print(f"\n{'='*55}")
    print(f"  Accuracy : {accuracy:.4f}  ({len(X_test)} test samples)")
    print(f"{'='*55}")
    print(classification_report(y_test, y_pred, target_names=class_names))

    return pipeline


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the CropSenseAI crop classification model."
    )
    parser.add_argument(
        "--raw-data",
        type=Path,
        default=DEFAULT_RAW_DATA_DIR,
        help="Path to the raw image directory (default: data/raw/).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_MODEL_DIR / MODEL_FILENAME,
        help="Path to save the trained model (default: data/models/crop_classifier.pkl).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test set proportion (default: 0.2).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    args = parse_arguments()
    logging.info("=== CropSenseAI — Model Training ===")
    logging.info("Raw data path  : %s", args.raw_data)
    logging.info("Model output   : %s", args.output)
    logging.info("Test size      : %.0f%%", args.test_size * 100)

    X, y, class_mapping = prepare_dataset(args.raw_data)
    logging.info("Class mapping  : %s", class_mapping)

    train_and_evaluate(
        X, y, class_mapping,
        model_output_path=args.output,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    logging.info("Training workflow finished successfully.")


if __name__ == "__main__":
    main()
