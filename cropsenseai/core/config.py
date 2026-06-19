"""Configuration helper for CropSenseAI."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


class Config:
    """Simple YAML-based application configuration."""

    def __init__(self, path: Optional[str] = None) -> None:
        if path is None:
            path = Path(__file__).resolve().parents[2] / "config" / "config.yaml"

        self.path = Path(path)
        self.values = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        if not self.path.exists():
            return {}

        with self.path.open("r", encoding="utf-8") as file:
            return yaml.safe_load(file) or {}

    def get(self, key: str, default: Any = None) -> Any:
        return self.values.get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self.values[key]

    def __contains__(self, key: str) -> bool:
        return key in self.values
