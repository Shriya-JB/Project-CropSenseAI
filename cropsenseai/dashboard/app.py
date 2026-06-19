"""cropsenseai/dashboard/app.py — Streamlit dashboard application for CropSenseAI.

FIXES applied:
  - CRITICAL: predict_crop() now uses extract_features() (HOG + color hist) instead
    of raw pixel flattening — must match training pipeline to get correct predictions
  - Fixed Streamlit deprecation: use_column_width → use_container_width
  - Fixed: model loaded with st.cache_resource (not st.cache_data) — models are
    not serialisable by st.cache_data and would cause a runtime error
  - Added class-mapping load so class indices decode correctly
"""

from pathlib import Path
from typing import Optional, Tuple

import cv2
import joblib
import numpy as np
import streamlit as st

try:
    from cropsenseai.crop_classification.feature_extraction import extract_features
    from cropsenseai.crop_classification.class_mapping import load_class_mapping, get_index_to_class
    FEATURES_AVAILABLE = True
except ImportError:
    FEATURES_AVAILABLE = False

IMAGE_SIZE = (128, 128)
SUPPORTED_EXTENSIONS = ["jpg", "jpeg", "png", "bmp", "tiff"]
DEFAULT_MODEL_PATH = "data/models/crop_classifier.pkl"
DEFAULT_MAPPING_PATH = Path("data/models/class_mapping.json")


def configure_page() -> None:
    st.set_page_config(
        page_title="CropSenseAI",
        page_icon="🌾",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .stButton>button { background-color: #2e7d32; color: white; }
        .stButton>button:hover { background-color: #1b5e20; }
        .stMetricValue { color: #1b5e20; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# FIXED: use st.cache_resource — models are not pickle-serialisable by cache_data
@st.cache_resource(show_spinner="Loading model …")
def load_model(model_path: str = DEFAULT_MODEL_PATH):
    root = Path(__file__).resolve().parents[2]
    candidates = [Path(model_path), root / model_path, root / "data" / "models" / Path(model_path).name]
    for p in candidates:
        if p.exists():
            return joblib.load(p)
    return None


def preprocess_image(image_bytes: bytes) -> Optional[np.ndarray]:
    buf = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    return img.astype(np.float32) / 255.0


def predict_crop(image: np.ndarray, model) -> Tuple[str, Optional[float]]:
    """
    FIXED: uses extract_features() — same pipeline as training.
    Original code used image.reshape(1,-1) (raw pixels) which mismatches training.
    """
    if not FEATURES_AVAILABLE:
        raise RuntimeError("scikit-image not installed. Run: pip install scikit-image")

    # Load class mapping
    try:
        class_map = load_class_mapping(DEFAULT_MAPPING_PATH)
        idx_to_class = get_index_to_class(class_map)
    except FileNotFoundError:
        # Fallback order matches alphabetical training order
        idx_to_class = {0: "cotton", 1: "maize", 2: "rice", 3: "wheat"}

    features = extract_features(image).reshape(1, -1)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(features)
        best_idx = int(np.argmax(proba, axis=1)[0])
        confidence = float(np.max(proba, axis=1)[0])
    else:
        predicted = model.predict(features)
        best_idx = int(np.asarray(predicted).ravel()[0])
        confidence = None

    crop_name = idx_to_class.get(best_idx, f"class_{best_idx}")
    return crop_name, confidence


def render_header() -> None:
    st.markdown("# 🌾 CropSenseAI Predictor")
    st.markdown("Upload a crop image to classify it and view the model confidence score.")


def render_sidebar() -> str:
    st.sidebar.header("Model settings")
    model_path = st.sidebar.text_input(
        "Classifier file path",
        value=DEFAULT_MODEL_PATH,
        help="Path to the saved crop classifier model (.pkl).",
    )
    st.sidebar.markdown(
        "---\n"
        "**Supported classes:** cotton, maize, rice, wheat\n\n"
        "**Accepted formats:** jpg, jpeg, png, bmp, tiff"
    )
    return model_path


def render_prediction_area(model_path: str) -> None:
    uploaded_file = st.file_uploader(
        "Upload a crop image",
        type=SUPPORTED_EXTENSIONS,
        help="Upload a clear image of the crop for classification.",
    )

    if uploaded_file is None:
        st.info("Upload an image to see crop type predictions.")
        return

    image_bytes = uploaded_file.read()
    image = preprocess_image(image_bytes)

    if image is None:
        st.error("Unable to decode the uploaded image. Try a different file.")
        return

    model = load_model(model_path)
    if model is None:
        st.warning(
            "⚠️ No trained model found. Train first:\n```bash\npython train_crop_model.py\n```"
        )
        return

    col_left, col_right = st.columns(2)
    with col_left:
        # FIXED: use_container_width replaces deprecated use_column_width
        st.image(image, caption="Uploaded image", use_container_width=True)

    with col_right:
        if st.button("Predict crop type", type="primary"):
            with st.spinner("Classifying image …"):
                try:
                    crop_name, confidence = predict_crop(image, model)
                except Exception as err:
                    st.error(f"Prediction error: {err}")
                    return

            st.success(f"Predicted crop: **{crop_name.capitalize()}**")
            if confidence is not None:
                st.metric("Confidence", f"{confidence * 100:.1f}%")
            else:
                st.info("Confidence score unavailable for this model type.")

            st.markdown("---")
            st.write("**Prediction details**")
            st.write(f"Model file: `{model_path}`")
            st.write(f"Image size: {IMAGE_SIZE[0]} × {IMAGE_SIZE[1]} px")


def main() -> None:
    configure_page()
    inject_styles()
    render_header()
    model_path = render_sidebar()
    render_prediction_area(model_path)


if __name__ == "__main__":
    main()
