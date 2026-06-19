"""
tests/test_crop_classification.py

Unit tests for the crop classification Data & Model Layer.
Run with: pytest tests/test_crop_classification.py -v

FIXES applied:
  - Added sys.path insert so tests run from project root without pip install -e .
  - dummy_image fixture shape verified to match (128,128,3)
"""

import sys
from pathlib import Path

# FIXED: ensure project root is on path when running pytest from any directory
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from cropsenseai.crop_classification.feature_extraction import (
    extract_hog_features,
    extract_color_histogram,
    extract_features,
    extract_features_batch,
)
from cropsenseai.crop_classification.class_mapping import (
    save_class_mapping,
    load_class_mapping,
    get_index_to_class,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def dummy_image():
    """128×128 RGB float32 image with random values in [0, 1]."""
    rng = np.random.default_rng(42)
    return rng.random((128, 128, 3)).astype(np.float32)


@pytest.fixture
def dummy_batch(dummy_image):
    """Batch of 5 identical dummy images, shape (5, 128, 128, 3)."""
    return np.stack([dummy_image] * 5, axis=0)


@pytest.fixture
def sample_class_map():
    return {"cotton": 0, "maize": 1, "rice": 2, "wheat": 3}


# ── HOG Feature Tests ─────────────────────────────────────────────────────────

class TestHOGFeatures:
    def test_output_is_1d(self, dummy_image):
        feat = extract_hog_features(dummy_image)
        assert feat.ndim == 1

    def test_output_dtype_is_float32(self, dummy_image):
        feat = extract_hog_features(dummy_image)
        assert feat.dtype == np.float32

    def test_same_image_gives_same_features(self, dummy_image):
        f1 = extract_hog_features(dummy_image)
        f2 = extract_hog_features(dummy_image)
        np.testing.assert_array_equal(f1, f2)

    def test_feature_vector_not_empty(self, dummy_image):
        feat = extract_hog_features(dummy_image)
        assert feat.size > 0


# ── Color Histogram Tests ─────────────────────────────────────────────────────

class TestColorHistogram:
    def test_output_length(self, dummy_image):
        feat = extract_color_histogram(dummy_image, bins=32)
        assert feat.shape == (96,)  # 32 bins × 3 channels

    def test_each_channel_sums_to_one(self, dummy_image):
        feat = extract_color_histogram(dummy_image, bins=32)
        for i in range(3):
            channel_hist = feat[i * 32:(i + 1) * 32]
            assert abs(channel_hist.sum() - 1.0) < 1e-5

    def test_custom_bin_count(self, dummy_image):
        feat = extract_color_histogram(dummy_image, bins=16)
        assert feat.shape == (48,)  # 16 × 3


# ── Combined Feature Tests ────────────────────────────────────────────────────

class TestExtractFeatures:
    def test_returns_1d_array(self, dummy_image):
        feat = extract_features(dummy_image)
        assert feat.ndim == 1

    def test_feature_dim_is_consistent(self, dummy_image):
        f1 = extract_features(dummy_image)
        f2 = extract_features(dummy_image)
        assert f1.shape == f2.shape

    def test_batch_shape(self, dummy_batch):
        feats = extract_features_batch(dummy_batch)
        assert feats.ndim == 2
        assert feats.shape[0] == 5

    def test_batch_row_equals_single(self, dummy_image, dummy_batch):
        single = extract_features(dummy_image)
        batch = extract_features_batch(dummy_batch)
        np.testing.assert_array_almost_equal(single, batch[0])


# ── Class Mapping Tests ───────────────────────────────────────────────────────

class TestClassMapping:
    def test_save_and_load_roundtrip(self, tmp_path, sample_class_map):
        path = tmp_path / "class_mapping.json"
        save_class_mapping(sample_class_map, path)
        loaded = load_class_mapping(path)
        assert loaded == sample_class_map

    def test_invert_mapping(self, sample_class_map):
        inverted = get_index_to_class(sample_class_map)
        assert inverted[0] == "cotton"
        assert inverted[3] == "wheat"
        assert len(inverted) == len(sample_class_map)

    def test_missing_file_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_class_mapping(tmp_path / "nonexistent.json")

    def test_saved_json_is_valid(self, tmp_path, sample_class_map):
        import json
        path = tmp_path / "cm.json"
        save_class_mapping(sample_class_map, path)
        with open(path) as f:
            data = json.load(f)
        assert isinstance(data, dict)
        assert all(isinstance(v, int) for v in data.values())
