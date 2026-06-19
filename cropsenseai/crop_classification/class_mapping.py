"""
cropsenseai/crop_classification/class_mapping.py

Save and load class mapping (class_name ↔ integer label).
Ensures prediction always uses the same label order as training.
"""

import json
import logging
from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)

DEFAULT_MAPPING_PATH = Path("data/models/class_mapping.json")


def save_class_mapping(class_map: Dict[str, int], path: Path = DEFAULT_MAPPING_PATH) -> None:
    """
    Save {class_name: label_index} to a JSON file.

    Args:
        class_map: e.g. {"cotton": 0, "maize": 1, "rice": 2, "wheat": 3}
        path:      where to save the file
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(class_map, f, indent=2)
    logger.info("Class mapping saved to %s", path)


def load_class_mapping(path: Path = DEFAULT_MAPPING_PATH) -> Dict[str, int]:
    """
    Load {class_name: label_index} from a JSON file.

    Returns:
        e.g. {"cotton": 0, "maize": 1, "rice": 2, "wheat": 3}
    """
    if not path.exists():
        raise FileNotFoundError(f"Class mapping file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        class_map = json.load(f)
    logger.info("Class mapping loaded from %s", path)
    return class_map


def get_index_to_class(class_map: Dict[str, int]) -> Dict[int, str]:
    """
    Invert the class map for decoding predictions.

    Args:
        class_map: {"cotton": 0, "maize": 1, ...}
    Returns:
        {0: "cotton", 1: "maize", ...}
    """
    return {v: k for k, v in class_map.items()}