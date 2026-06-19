"""
Crop Classification Module

Handles crop type identification from images using HOG + color histogram
features and a Random Forest classifier.

Supported crops: cotton, maize, rice, wheat
"""
# FIXED: Removed phantom __all__ entries ('models', 'preprocessing', 'utils')
# that don't exist as files. Listed what actually exists.
__all__ = [
    "class_mapping",
    "feature_extraction",
    "model_trainer",
    "model_evaluator",
    "prediction_pipeline",
]
