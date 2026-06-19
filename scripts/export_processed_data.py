"""
scripts/export_processed_data.py

Preprocesses all images and saves them as numpy arrays to data/processed/.
This avoids reloading images from disk on every training run.

Saves:
    data/processed/X.npy   — shape (N, 128, 128, 3)
    data/processed/y.npy   — shape (N,)

Usage:
    python scripts/export_processed_data.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from cropsenseai.core.data_preprocessing import CropImagePreprocessor
from cropsenseai.crop_classification.class_mapping import save_class_mapping

DATA_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/processed")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    preprocessor = CropImagePreprocessor(str(DATA_DIR))
    X, y, class_map = preprocessor.load_and_preprocess(normalize=True)

    np.save(OUTPUT_DIR / "X.npy", X)
    np.save(OUTPUT_DIR / "y.npy", y)
    save_class_mapping(class_map, Path("data/models/class_mapping.json"))

    print(f"Saved X.npy: {X.shape}")
    print(f"Saved y.npy: {y.shape}")
    print(f"Class map  : {class_map}")
    print(f"Output dir : {OUTPUT_DIR}")


if __name__ == "__main__":
    main()