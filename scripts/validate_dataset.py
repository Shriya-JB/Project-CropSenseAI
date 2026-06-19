"""
scripts/validate_dataset.py

Run this FIRST before training to check your dataset.
Detects corrupt images and prints per-class counts.

Usage:
    python scripts/validate_dataset.py
    python scripts/validate_dataset.py --data-dir data/raw
"""

import argparse
import logging
import sys
from pathlib import Path

# Make sure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cropsenseai.core.data_preprocessing import CropImagePreprocessor

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Validate the CropSenseAI dataset.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    args = parser.parse_args()

    logger.info("Validating dataset at: %s", args.data_dir)
    preprocessor = CropImagePreprocessor(str(args.data_dir))
    stats = preprocessor.get_dataset_statistics()

    print("\n" + "="*50)
    print("  DATASET VALIDATION REPORT")
    print("="*50)
    for class_name, info in stats.items():
        if class_name.startswith("_"):
            continue
        status = "✅" if info["corrupt"] == 0 else "⚠️ "
        print(f"  {status}  {class_name:<10}  valid={info['valid']:>4}  corrupt={info['corrupt']:>3}")

    summary = stats["_summary"]
    print("-"*50)
    print(f"  Total valid  : {summary['total_valid']}")
    print(f"  Total corrupt: {summary['total_corrupt']}")
    print(f"  Num classes  : {summary['num_classes']}")
    print("="*50 + "\n")

    if summary["total_corrupt"] > 0:
        logger.warning("%d corrupt images found. Remove them before training.", summary["total_corrupt"])
        sys.exit(1)
    else:
        logger.info("Dataset is clean. Ready to train.")


if __name__ == "__main__":
    main()