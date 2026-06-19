"""
scripts/generate_dataset_report.py

Generates a CSV report with per-class image counts and class distribution.
Saves to data/processed/dataset_report.csv.

Usage:
    python scripts/generate_dataset_report.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from cropsenseai.core.data_preprocessing import CropImagePreprocessor

DATA_DIR = Path("data/raw")
OUTPUT_PATH = Path("data/processed/dataset_report.csv")


def main():
    preprocessor = CropImagePreprocessor(str(DATA_DIR))
    stats = preprocessor.get_dataset_statistics()

    rows = []
    total_valid = stats["_summary"]["total_valid"]
    for class_name, info in stats.items():
        if class_name.startswith("_"):
            continue
        pct = round(info["valid"] / total_valid * 100, 2) if total_valid else 0
        rows.append({
            "class": class_name,
            "valid_images": info["valid"],
            "corrupt_images": info["corrupt"],
            "percentage": pct,
        })

    df = pd.DataFrame(rows).sort_values("class")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(df.to_string(index=False))
    print(f"\nReport saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()