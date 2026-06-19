"""
cropsenseai/crop_classification/model_trainer.py

Encapsulates the full training pipeline:
  load data → extract features → train → save model + class mapping
"""

import logging
from pathlib import Path
from typing import Tuple, Dict

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from cropsenseai.core.data_preprocessing import CropImagePreprocessor
from cropsenseai.crop_classification.feature_extraction import extract_features_batch
from cropsenseai.crop_classification.class_mapping import save_class_mapping

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = Path("data/models/crop_classifier.pkl")
DEFAULT_RAW_DATA = Path("data/raw")


def build_pipeline() -> Pipeline:
    """
    Returns a scikit-learn Pipeline:
      StandardScaler → RandomForestClassifier
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            random_state=42,
            n_jobs=-1,
        )),
    ])


def train(
    raw_data_dir: Path = DEFAULT_RAW_DATA,
    model_output_path: Path = DEFAULT_MODEL_PATH,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[Pipeline, Dict]:
    """
    Full training pipeline. Returns (trained_pipeline, class_map).

    Steps:
    1. Load images using CropImagePreprocessor
    2. Extract HOG + color histogram features
    3. Split train/test
    4. Train Random Forest pipeline
    5. Save model and class mapping
    """
    # Step 1: Load images
    preprocessor = CropImagePreprocessor(str(raw_data_dir))
    X_images, y, class_map = preprocessor.load_and_preprocess(normalize=True)
    logger.info("Images loaded: %s, Labels: %s", X_images.shape, y.shape)

    # Step 2: Extract features
    X_features = extract_features_batch(X_images)  # shape: (N, feature_dim)
    logger.info("Feature matrix: %s", X_features.shape)

    # Step 3: Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    logger.info("Train: %d, Test: %d", len(X_train), len(X_test))

    # Step 4: Train
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    logger.info("Training complete")

    # Step 5: Save
    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_output_path)
    save_class_mapping(class_map)

    logger.info("Model saved to %s", model_output_path)
    return pipeline, class_map