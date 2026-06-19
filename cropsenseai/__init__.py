"""
CropSenseAI - AI System for Crop Classification, Stress Detection, and Irrigation Management
"""

__version__ = "1.0.0"
__author__ = "CropSenseAI Team"

# FIXED: Removed broken __all__ entries ("Config", "logger") that don't resolve to importable names.
# Graceful fallback imports are fine; they prevent crash on partial installs.

try:
    from cropsenseai.core.config import Config
    from cropsenseai.core.logger import setup_logger
except Exception:
    Config = None       # type: ignore
    setup_logger = None # type: ignore
