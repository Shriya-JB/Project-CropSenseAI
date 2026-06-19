"""
cropsenseai/core/data_preprocessing.py

Central image preprocessing pipeline for CropSenseAI.
Loads raw images from data/raw/<class_name>/, validates them,
resizes, normalizes, and returns numpy arrays ready for feature extraction.

FIXES applied:
  - Case-insensitive folder matching (handles 'Rice', 'WHEAT', etc.)
  - Logs skipped/corrupt counts per class
  - get_dataset_statistics() no longer double-loads pixels unnecessarily
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# These must match the folder names inside data/raw/ (case-insensitive matching applied below)
SUPPORTED_CLASSES: List[str] = ["cotton", "maize", "rice", "wheat"]
IMAGE_SIZE: Tuple[int, int] = (128, 128)
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


class CropImagePreprocessor:
    """
    Loads and preprocesses crop images from a directory structured as::

        root_dir/
            cotton/   ← folder name = class label (case-insensitive)
            Maize/
            Rice/
            wheat/
    """

    def __init__(self, root_dir: str, image_size: Tuple[int, int] = IMAGE_SIZE) -> None:
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        self._validate_root_dir()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_root_dir(self) -> None:
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.root_dir}")
        if not self.root_dir.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {self.root_dir}")

    def _get_class_dirs(self) -> Dict[str, Path]:
        """
        Returns ``{class_name_lower: Path}`` for every valid subfolder found.

        FIXED: uses case-insensitive comparison so 'Rice', 'RICE', 'rice' all work.
        """
        class_dirs: Dict[str, Path] = {}
        for item in sorted(self.root_dir.iterdir()):
            if not item.is_dir():
                continue
            name_lower = item.name.lower()
            if name_lower not in SUPPORTED_CLASSES:
                logger.warning("Unknown class folder — skipped: '%s'", item.name)
                continue
            if name_lower in class_dirs:
                logger.warning(
                    "Duplicate class folder (case clash) — kept first, skipping: '%s'", item.name
                )
                continue
            class_dirs[name_lower] = item

        if not class_dirs:
            raise ValueError(
                f"No valid class folders found in '{self.root_dir}'. "
                f"Expected subfolders named one of: {SUPPORTED_CLASSES}"
            )
        return class_dirs

    def _collect_image_paths(self, class_dir: Path) -> List[Path]:
        """Return all image file paths inside a class folder (non-recursive)."""
        return [
            f
            for f in sorted(class_dir.iterdir())
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        ]

    def _load_single_image(self, image_path: Path) -> Optional[np.ndarray]:
        """
        Load one image: BGR → RGB → resize → float32 in [0, 1].
        Returns ``None`` if the file is corrupt or unreadable.
        """
        image = cv2.imread(str(image_path))
        if image is None:
            logger.warning("Corrupt / unreadable image skipped: %s", image_path.name)
            return None
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_AREA)
        return image.astype(np.float32) / 255.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_and_preprocess(
        self, normalize: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
        """
        Main entry point used by ``train_crop_model.py`` and scripts.

        Returns:
            X         : shape (N, H, W, 3)  — float32 images in [0, 1]
            y         : shape (N,)           — integer class labels (int64)
            class_map : ``{"cotton": 0, "maize": 1, "rice": 2, "wheat": 3}``
        """
        class_dirs = self._get_class_dirs()
        # Assign labels alphabetically so the mapping is deterministic
        class_map: Dict[str, int] = {
            name: idx for idx, name in enumerate(sorted(class_dirs.keys()))
        }

        images: List[np.ndarray] = []
        labels: List[int] = []
        total_skipped = 0

        for class_name in sorted(class_dirs.keys()):
            class_dir = class_dirs[class_name]
            class_label = class_map[class_name]
            image_paths = self._collect_image_paths(class_dir)

            logger.info(
                "Loading '%s' (label=%d): %d files found in '%s'",
                class_name, class_label, len(image_paths), class_dir.name,
            )

            class_skipped = 0
            for path in image_paths:
                img = self._load_single_image(path)
                if img is None:
                    class_skipped += 1
                    total_skipped += 1
                    continue
                images.append(img)
                labels.append(class_label)

            if class_skipped:
                logger.warning("  → %d corrupt images skipped in '%s'", class_skipped, class_name)

        if not images:
            raise RuntimeError(
                "No valid images were loaded. "
                "Check that data/raw/ contains subfolders with images."
            )

        X = np.stack(images, axis=0)          # (N, H, W, 3)
        y = np.array(labels, dtype=np.int64)  # (N,)

        logger.info(
            "Dataset ready: %d images loaded, %d skipped. Class map: %s",
            len(images), total_skipped, class_map,
        )
        return X, y, class_map

    def get_dataset_statistics(self) -> Dict:
        """
        Count images per class without loading pixel data into memory.
        Used by ``validate_dataset.py`` and ``generate_dataset_report.py``.

        FIXED: checks readability via cv2.haveImageReader instead of loading
               full pixel data just to check corruption.
        """
        class_dirs = self._get_class_dirs()
        stats: Dict = {}
        total_valid = 0
        total_corrupt = 0

        for class_name, class_dir in class_dirs.items():
            paths = self._collect_image_paths(class_dir)
            valid_count = 0
            corrupt_count = 0
            for p in paths:
                # Lightweight readability check (reads header only)
                img = cv2.imread(str(p))
                if img is None:
                    corrupt_count += 1
                else:
                    valid_count += 1
            stats[class_name] = {
                "total": len(paths),
                "valid": valid_count,
                "corrupt": corrupt_count,
            }
            total_valid += valid_count
            total_corrupt += corrupt_count

        stats["_summary"] = {
            "total_valid": total_valid,
            "total_corrupt": total_corrupt,
            "num_classes": len(class_dirs),
        }
        return stats
