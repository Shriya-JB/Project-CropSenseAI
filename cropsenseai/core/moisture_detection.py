"""Moisture stress detection for CropSenseAI.

This module provides a simple machine learning-based detector for crop stress
levels based on vegetation index, temperature, and rainfall.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import joblib
import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier

logger = logging.getLogger(__name__)

STRESS_LEVELS: List[str] = ["low stress", "medium stress", "high stress"]
DEFAULT_MODEL_FILENAME = "moisture_detector.pkl"
FEATURE_NAMES = ["vegetation_index", "temperature", "rainfall"]


@dataclass
class MoistureStressDetector:
    """Detector for low/medium/high moisture stress."""

    model: Pipeline
    label_encoder: LabelEncoder

    @classmethod
    def create_default(cls, random_state: int = 42) -> "MoistureStressDetector":
        """Build and train a detector from synthetic stress examples."""
        X, y = cls._generate_synthetic_training_data(random_state=random_state)
        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(y)

        pipeline = cls._build_pipeline()
        pipeline.fit(X, y_encoded)

        logger.info("Trained default moisture stress detector with synthetic data.")
        return cls(model=pipeline, label_encoder=encoder)

    @staticmethod
    def _build_pipeline() -> Pipeline:
        numeric_transformer = Pipeline(
            steps=[("scaler", StandardScaler())]
        )

        transformer = ColumnTransformer(
            transformers=[("num", numeric_transformer, [0, 1, 2])],
            remainder="passthrough",
        )

        return Pipeline(
            steps=[("transform", transformer), ("classifier", DecisionTreeClassifier(random_state=42))]
        )

    @staticmethod
    def _generate_synthetic_training_data(random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(random_state)
        samples_per_class = 120

        low_stress = np.column_stack(
            [
                rng.normal(loc=0.82, scale=0.05, size=samples_per_class),
                rng.normal(loc=24.0, scale=2.5, size=samples_per_class),
                rng.normal(loc=95.0, scale=15.0, size=samples_per_class),
            ]
        )

        medium_stress = np.column_stack(
            [
                rng.normal(loc=0.55, scale=0.08, size=samples_per_class),
                rng.normal(loc=29.0, scale=3.5, size=samples_per_class),
                rng.normal(loc=45.0, scale=12.0, size=samples_per_class),
            ]
        )

        high_stress = np.column_stack(
            [
                rng.normal(loc=0.28, scale=0.07, size=samples_per_class),
                rng.normal(loc=36.0, scale=4.5, size=samples_per_class),
                rng.normal(loc=18.0, scale=8.0, size=samples_per_class),
            ]
        )

        X = np.vstack([low_stress, medium_stress, high_stress])
        y = np.array(
            ["low stress"] * samples_per_class
            + ["medium stress"] * samples_per_class
            + ["high stress"] * samples_per_class
        )

        return X.astype(np.float32), y

    def predict(self, vegetation_index: float, temperature: float, rainfall: float) -> str:
        """Predict the moisture stress category for a single input sample."""
        sample = np.array([[vegetation_index, temperature, rainfall]], dtype=np.float32)
        prediction = self.model.predict(sample)
        stress_label = self.label_encoder.inverse_transform(prediction)[0]
        logger.debug(
            "Predicted stress level %s for sample %s",
            stress_label,
            sample.tolist(),
        )
        return stress_label

    def predict_proba(self, vegetation_index: float, temperature: float, rainfall: float) -> dict:
        """Return a probability distribution across stress categories."""
        if not hasattr(self.model, "predict_proba"):
            raise AttributeError("Underlying model does not support probability estimates.")

        sample = np.array([[vegetation_index, temperature, rainfall]], dtype=np.float32)
        probabilities = self.model.predict_proba(sample)[0]
        decoded = self.label_encoder.inverse_transform(np.arange(probabilities.shape[0]))
        return {str(label): float(probabilities[idx]) for idx, label in enumerate(decoded)}

    def save(self, model_path: str) -> None:
        """Persist the trained detector to disk."""
        path = Path(model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info("Saved moisture detector to %s", path)

    @classmethod
    def load(cls, model_path: str) -> "MoistureStressDetector":
        """Load a saved detector from disk."""
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Detector file not found: {path}")

        detector = joblib.load(path)
        if not isinstance(detector, MoistureStressDetector):
            raise TypeError("Loaded object is not a MoistureStressDetector instance.")

        logger.info("Loaded moisture detector from %s", path)
        return detector


def build_and_save_default_detector(model_path: Optional[str] = None) -> MoistureStressDetector:
    """Train a default synthetic moisture detector and save it to disk."""
    detector = MoistureStressDetector.create_default()
    if model_path is None:
        model_path = DEFAULT_MODEL_FILENAME
    detector.save(model_path)
    return detector


def load_or_create_detector(model_path: Optional[str] = None) -> MoistureStressDetector:
    """Load a detector if present; otherwise train and return a default detector."""
    if model_path is None:
        model_path = DEFAULT_MODEL_FILENAME

    try:
        return MoistureStressDetector.load(model_path)
    except FileNotFoundError:
        logger.warning("Moisture detector not found at %s. Training default detector.", model_path)
        return build_and_save_default_detector(model_path)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    detector = load_or_create_detector()
    print("Moisture stress detector is ready.")
    print("Example output:")
    example_label = detector.predict(vegetation_index=0.62, temperature=28.0, rainfall=50.0)
    print(f"Predicted stress level: {example_label}")
    print("Probability distribution:")
    try:
        print(detector.predict_proba(vegetation_index=0.62, temperature=28.0, rainfall=50.0))
    except AttributeError:
        print("Probability estimates not available for this model.")
