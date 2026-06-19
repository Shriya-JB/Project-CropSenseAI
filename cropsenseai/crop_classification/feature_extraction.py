"""
cropsenseai/crop_classification/feature_extraction.py

Extracts HOG features and color histograms from preprocessed images.
These features replace raw pixel flattening for much better ML performance.

Dependencies: scikit-image (skimage) — included in requirements.txt
"""

import logging
from typing import Tuple

import cv2
import numpy as np

# FIXED: Added try/except with a clear error message.
# scikit-image must be installed: pip install scikit-image
try:
    from skimage.feature import hog as skimage_hog
except ImportError as _e:
    raise ImportError(
        "scikit-image is required for feature extraction. "
        "Install it with: pip install scikit-image"
    ) from _e

logger = logging.getLogger(__name__)

HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL: Tuple[int, int] = (8, 8)
HOG_CELLS_PER_BLOCK: Tuple[int, int] = (2, 2)
COLOR_HIST_BINS = 32


def extract_hog_features(image: np.ndarray) -> np.ndarray:
    """
    Extract HOG (Histogram of Oriented Gradients) features.

    Args:
        image: float32 array of shape (H, W, 3), values in [0, 1]

    Returns:
        1-D numpy float32 array of HOG features.
    """
    # Convert to uint8 grayscale for HOG
    gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    features = skimage_hog(
        gray,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        block_norm="L2-Hys",
        feature_vector=True,
    )
    return features.astype(np.float32)


def extract_color_histogram(image: np.ndarray, bins: int = COLOR_HIST_BINS) -> np.ndarray:
    """
    Extract a normalized color histogram for each RGB channel.

    Args:
        image: float32 array of shape (H, W, 3), values in [0, 1]
        bins:  number of histogram bins per channel

    Returns:
        1-D numpy float32 array of length ``bins * 3``.
    """
    features = []
    for channel in range(3):
        hist, _ = np.histogram(image[:, :, channel], bins=bins, range=(0.0, 1.0))
        hist = hist.astype(np.float32)
        total = hist.sum()
        if total > 0:
            hist /= total
        features.append(hist)
    return np.concatenate(features)


def extract_features(image: np.ndarray) -> np.ndarray:
    """
    Combine HOG + color histogram into a single feature vector.
    Called by ``model_trainer.py`` and ``prediction_pipeline.py``.

    Args:
        image: float32 array of shape (H, W, 3), values in [0, 1]

    Returns:
        1-D float32 feature vector.
    """
    hog_feat = extract_hog_features(image)
    color_feat = extract_color_histogram(image)
    return np.concatenate([hog_feat, color_feat])


def extract_features_batch(X: np.ndarray) -> np.ndarray:
    """
    Extract features for a batch of images.

    Args:
        X: float32 array of shape (N, H, W, 3)

    Returns:
        float32 array of shape (N, feature_dim)
    """
    n = X.shape[0]
    logger.info("Extracting features from %d images ...", n)
    feature_list = []
    for i in range(n):
        feature_list.append(extract_features(X[i]))
        if (i + 1) % 100 == 0 or (i + 1) == n:
            logger.info("  Processed %d / %d images", i + 1, n)
    result = np.stack(feature_list, axis=0)
    logger.info("Feature matrix shape: %s", result.shape)
    return result
