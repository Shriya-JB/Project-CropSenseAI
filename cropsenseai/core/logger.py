"""Logging utilities for CropSenseAI."""

import logging
from logging import Logger
from typing import Optional


def setup_logger(name: str, level: int = logging.INFO) -> Logger:
    """Create and configure a logger for the CropSenseAI application."""
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False

    return logger


def get_logger(name: str, level: int = logging.INFO) -> Logger:
    """Return a configured logger instance."""
    return setup_logger(name, level)
