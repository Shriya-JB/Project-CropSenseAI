"""
CropSenseAI Dashboard — Main Entry Point
Run with:  streamlit run main.py

FIXES applied:
  - Replaced all placeholder page stubs with fully functional implementations
  - Crop Classification page: upload → preprocess → HOG features → predict → show result
  - Stress Analysis page: sliders for vegetation index / temperature / rainfall → predict stress
  - Irrigation Recommendations page: integrated with both crop + stress results
  - History & Analytics page: session-state based log with charts
  - Model loading is cached (st.cache_resource) so it is not reloaded on every interaction
  - All pages handle missing model gracefully with a clear instruction message
"""

import logging
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import joblib
import numpy as np
import streamlit as st

# ── Logger ────────────────────────────────────────────────────────────────────
try:
    from cropsenseai.core.logger import setup_logger
    logger = setup_logger(__name__)
except Exception:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

# ── Project imports (graceful fallbacks so dashboard still partially loads) ──
try:
    from cropsenseai.crop_classification.feature_extraction import extract_features
    from cropsenseai.crop_classification.class_mapping import load_class_mapping, get_index_to_class
    FEATURE_EXTRACTION_AVAILABLE = True
except ImportError:
    FEATURE_EXTRACTION_AVAILABLE = False
    logger.warning("Feature extraction unavailable — classification page disabled.")

try:
    from cropsenseai.core.moisture_detection import load_or_create_detector
    STRESS_DETECTION_AVAILABLE = True
except ImportError:
    STRESS_DETECTION_AVAILABLE = False
    logger.warning("Moisture detection unavailable — stress page disabled.")

try:
    from irrigation_recommendation import recommend_irrigation, get_recommendation_rationale, get_water_amount_mm
    IRRIGATION_AVAILABLE = True
except ImportError:
    IRRIGATION_AVAILABLE = False
    logger.warning("Irrigation recommendation unavailable.")

# ── Constants ─────────────────────────────────────────────────────────────────
IMAGE_SIZE = (128, 128)
SUPPORTED_EXTENSIONS = ["jpg", "jpeg", "png", "bmp", "tiff"]
DEFAULT_MODEL_PATH = Path("data/models/crop_classifier.pkl")
DEFAULT_MAPPING_PATH = Path("data/models/class_mapping.json")
DEFAULT_STRESS_MODEL_PATH = "data/models/moisture_detector.pkl"
CROP_EMOJI = {"cotton": "🌿", "maize": "🌽", "rice": "🌾", "wheat": "🌾"}


# ── Cached loaders ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading crop classifier …")
def _load_crop_model():
    """Load and cache the crop classification model."""
    candidates = [
        DEFAULT_MODEL_PATH,
        Path("crop_classifier.pkl"),
    ]
    for p in candidates:
        if p.exists():
            return joblib.load(p)
    return None


@st.cache_resource(show_spinner="Loading stress detector …")
def _load_stress_detector():
    """Load and cache the moisture stress detector."""
    if not STRESS_DETECTION_AVAILABLE:
        return None
    try:
        return load_or_create_detector(DEFAULT_STRESS_MODEL_PATH)
    except Exception as exc:
        logger.warning("Could not load stress detector: %s", exc)
        return None


# ── Page config ───────────────────────────────────────────────────────────────

def _configure_page() -> None:
    st.set_page_config(
        page_title="CropSenseAI Dashboard",
        page_icon="🌾",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        .main-title { font-size: 2.2rem; font-weight: 700; color: #1b5e20; }
        .section-header { font-size: 1.3rem; font-weight: 600; color: #2e7d32; margin-top: 1rem; }
        .result-box { background: #f1f8e9; border-left: 4px solid #558b2f;
                      padding: 1rem; border-radius: 6px; margin-top: 1rem; }
        .warning-box { background: #fff3e0; border-left: 4px solid #e65100;
                       padding: 1rem; border-radius: 6px; margin-top: 1rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────

def _render_sidebar() -> str:
    st.sidebar.markdown("## 🌾 CropSenseAI")
    st.sidebar.markdown("---")
    page = st.sidebar.radio(
        "Navigate",
        ["🏠 Home", "🍃 Crop Classification", "📊 Stress Analysis",
         "💧 Irrigation Recommendations", "📈 History & Analytics"],
    )
    st.sidebar.markdown("---")
    st.sidebar.caption("Model paths")
    st.sidebar.code(str(DEFAULT_MODEL_PATH), language=None)
    return page


# ── Home page ─────────────────────────────────────────────────────────────────

def _show_home() -> None:
    st.markdown('<p class="main-title">Welcome to CropSenseAI 🌾</p>', unsafe_allow_html=True)
    st.markdown(
        """
        **AI-Powered Precision Agriculture Platform**

        CropSenseAI helps farmers and agronomists:

        | Feature | Description |
        |---------|-------------|
        | 🍃 **Crop Classification** | Identify cotton, maize, rice, or wheat from a leaf / field photo |
        | 📊 **Stress Analysis** | Detect moisture stress level (low / medium / high) |
        | 💧 **Irrigation Recommendations** | Get actionable irrigation decisions with water depth |
        | 📈 **History & Analytics** | Track predictions and spot trends across sessions |

        ---
        **How to use:**
        1. Place your dataset in `data/raw/<crop_name>/`
        2. Run `python train_crop_model.py` to train the classifier
        3. Use the sidebar to navigate to any feature
        """
    )

    col1, col2, col3 = st.columns(3)
    model_ready = DEFAULT_MODEL_PATH.exists()
    mapping_ready = DEFAULT_MAPPING_PATH.exists()
    col1.metric("Classifier", "✅ Ready" if model_ready else "❌ Not trained")
    col2.metric("Class Mapping", "✅ Found" if mapping_ready else "❌ Missing")
    col3.metric("Supported Crops", "4  (cotton · maize · rice · wheat)")


# ── Crop Classification page ──────────────────────────────────────────────────

def _preprocess_uploaded_image(image_bytes: bytes) -> Optional[np.ndarray]:
    """Decode bytes → RGB float32 (H,W,3) ready for feature extraction."""
    buf = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    return img.astype(np.float32) / 255.0


def _show_classification() -> None:
    st.markdown('<p class="main-title">Crop Classification 🍃</p>', unsafe_allow_html=True)

    if not FEATURE_EXTRACTION_AVAILABLE:
        st.error("scikit-image not installed. Run: `pip install scikit-image`")
        return

    model = _load_crop_model()
    if model is None:
        st.warning(
            "⚠️ No trained model found at `data/models/crop_classifier.pkl`.\n\n"
            "**Train the model first:**\n```bash\npython train_crop_model.py\n```"
        )
        return

    # Load class mapping
    try:
        class_map = load_class_mapping(DEFAULT_MAPPING_PATH)
        idx_to_class = get_index_to_class(class_map)
    except FileNotFoundError:
        st.error("Class mapping file not found. Re-run training to regenerate it.")
        return

    uploaded = st.file_uploader(
        "Upload a crop image (jpg / jpeg / png / bmp / tiff)",
        type=SUPPORTED_EXTENSIONS,
    )

    if uploaded is None:
        st.info("📂 Upload an image to begin classification.")
        return

    image_bytes = uploaded.read()
    img_display = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    img_display = cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB)

    col_img, col_result = st.columns([1, 1])
    with col_img:
        st.image(img_display, caption="Uploaded image", use_container_width=True)

    with col_result:
        if st.button("🔍 Classify Crop", type="primary"):
            with st.spinner("Extracting features and classifying …"):
                t0 = time.time()
                img_float = _preprocess_uploaded_image(image_bytes)
                if img_float is None:
                    st.error("Could not decode image. Try a different file.")
                    return

                features = extract_features(img_float).reshape(1, -1)
                prediction = model.predict(features)
                label_idx = int(np.asarray(prediction).ravel()[0])
                crop_name = idx_to_class.get(label_idx, "unknown")
                elapsed = time.time() - t0

            emoji = CROP_EMOJI.get(crop_name, "🌿")
            st.success(f"### {emoji} Predicted: **{crop_name.capitalize()}**")

            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(features)[0]
                st.metric("Confidence", f"{proba.max() * 100:.1f}%")
                st.markdown("**All class probabilities:**")
                for i, p in enumerate(proba):
                    name = idx_to_class.get(i, str(i))
                    st.progress(float(p), text=f"{name}: {p*100:.1f}%")
            else:
                st.info("Confidence scores not available for this model.")

            st.caption(f"Inference time: {elapsed*1000:.1f} ms | Image: {IMAGE_SIZE[0]}×{IMAGE_SIZE[1]} px")

            # Store for History
            _add_to_history("classification", {"crop": crop_name, "file": uploaded.name})


# ── Stress Analysis page ───────────────────────────────────────────────────────

def _show_stress_analysis() -> None:
    st.markdown('<p class="main-title">Moisture Stress Analysis 📊</p>', unsafe_allow_html=True)
    st.markdown(
        "Enter field sensor readings to detect the moisture stress level of your crop."
    )

    detector = _load_stress_detector()
    if detector is None:
        st.error("Stress detector could not be loaded. Check installation.")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        veg_index = st.slider(
            "Vegetation Index (NDVI)", min_value=0.0, max_value=1.0,
            value=0.65, step=0.01,
            help="Higher = healthier vegetation. Typical stressed range: < 0.4"
        )
    with col2:
        temperature = st.slider(
            "Temperature (°C)", min_value=10.0, max_value=50.0,
            value=28.0, step=0.5,
            help="Ambient / canopy temperature. High values indicate stress."
        )
    with col3:
        rainfall = st.slider(
            "Rainfall last 7 days (mm)", min_value=0.0, max_value=200.0,
            value=50.0, step=1.0,
            help="Lower rainfall = higher stress risk."
        )

    if st.button("🔬 Detect Stress Level", type="primary"):
        with st.spinner("Analysing …"):
            stress_label = detector.predict(veg_index, temperature, rainfall)

        color_map = {
            "low stress": "🟢",
            "medium stress": "🟡",
            "high stress": "🔴",
        }
        icon = color_map.get(stress_label, "⚪")
        st.markdown(
            f'<div class="result-box"><h3>{icon} Stress Level: <strong>{stress_label.title()}</strong></h3></div>',
            unsafe_allow_html=True,
        )

        # Show probability breakdown if available
        try:
            proba_dict = detector.predict_proba(veg_index, temperature, rainfall)
            st.markdown("**Probability breakdown:**")
            for label, prob in sorted(proba_dict.items(), key=lambda x: -x[1]):
                st.progress(prob, text=f"{label}: {prob*100:.1f}%")
        except AttributeError:
            pass

        st.session_state["last_stress"] = stress_label
        _add_to_history("stress", {
            "stress": stress_label,
            "veg_index": veg_index,
            "temperature": temperature,
            "rainfall": rainfall,
        })

        st.success("✅ Stress level detected. Go to **Irrigation Recommendations** for next steps.")


# ── Irrigation Recommendations page ───────────────────────────────────────────

def _show_recommendations() -> None:
    st.markdown('<p class="main-title">Irrigation Recommendations 💧</p>', unsafe_allow_html=True)

    if not IRRIGATION_AVAILABLE:
        st.error("`irrigation_recommendation.py` not found. Ensure it is in the project root.")
        return

    col1, col2 = st.columns(2)
    with col1:
        crop = st.selectbox("Crop type", ["wheat", "rice", "cotton", "maize"])
    with col2:
        # Pre-fill from stress analysis if available
        default_stress = st.session_state.get("last_stress", "low stress")
        stress = st.selectbox(
            "Moisture stress level",
            ["low stress", "medium stress", "high stress"],
            index=["low stress", "medium stress", "high stress"].index(default_stress),
        )

    forecast = st.slider(
        "Rainfall forecast (next 48 hours, mm)", 0.0, 60.0, 5.0, step=0.5
    )

    if st.button("💧 Get Irrigation Recommendation", type="primary"):
        action = recommend_irrigation(crop, stress, forecast)
        rationale = get_recommendation_rationale(crop, stress, forecast)
        water_mm = get_water_amount_mm(action, stress)

        action_colors = {
            "irrigate now": "🔴",
            "irrigate in 2 days": "🟡",
            "no irrigation needed": "🟢",
        }
        icon = action_colors.get(action, "⚪")

        st.markdown(
            f'<div class="result-box">'
            f'<h3>{icon} Action: <strong>{action.title()}</strong></h3>'
            f'<p>{rationale}</p>'
            f'</div>',
            unsafe_allow_html=True,
        )

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Recommended Depth", f"{water_mm} mm" if water_mm > 0 else "—")
        col_b.metric("Crop", crop.capitalize())
        col_c.metric("Stress Level", stress.title())

        _add_to_history("irrigation", {
            "crop": crop, "stress": stress,
            "forecast_mm": forecast, "action": action, "water_mm": water_mm,
        })


# ── History & Analytics page ───────────────────────────────────────────────────

def _show_history() -> None:
    st.markdown('<p class="main-title">History & Analytics 📈</p>', unsafe_allow_html=True)

    history = st.session_state.get("history", [])
    if not history:
        st.info("No predictions yet in this session. Use the other pages to generate data.")
        return

    import pandas as pd

    # Classification history
    clf_records = [h["data"] for h in history if h["type"] == "classification"]
    if clf_records:
        st.subheader("Crop Classification History")
        df_clf = pd.DataFrame(clf_records)
        st.dataframe(df_clf, use_container_width=True)
        crop_counts = df_clf["crop"].value_counts()
        st.bar_chart(crop_counts)

    # Stress history
    stress_records = [h["data"] for h in history if h["type"] == "stress"]
    if stress_records:
        st.subheader("Stress Analysis History")
        df_stress = pd.DataFrame(stress_records)
        st.dataframe(df_stress, use_container_width=True)

        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(12, 3))
        for ax, col in zip(axes, ["veg_index", "temperature", "rainfall"]):
            ax.plot(df_stress[col].values, marker="o")
            ax.set_title(col.replace("_", " ").title())
            ax.set_xlabel("Reading #")
        st.pyplot(fig)
        plt.close(fig)

    # Irrigation history
    irr_records = [h["data"] for h in history if h["type"] == "irrigation"]
    if irr_records:
        st.subheader("Irrigation Recommendation History")
        df_irr = pd.DataFrame(irr_records)
        st.dataframe(df_irr, use_container_width=True)
        action_counts = df_irr["action"].value_counts()
        st.bar_chart(action_counts)

    if st.button("🗑️ Clear History"):
        st.session_state["history"] = []
        st.rerun()


# ── Session history helper ────────────────────────────────────────────────────

def _add_to_history(event_type: str, data: dict) -> None:
    if "history" not in st.session_state:
        st.session_state["history"] = []
    st.session_state["history"].append({"type": event_type, "data": data})


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    _configure_page()
    page = _render_sidebar()

    if page == "🏠 Home":
        _show_home()
    elif page == "🍃 Crop Classification":
        _show_classification()
    elif page == "📊 Stress Analysis":
        _show_stress_analysis()
    elif page == "💧 Irrigation Recommendations":
        _show_recommendations()
    elif page == "📈 History & Analytics":
        _show_history()


if __name__ == "__main__":
    main()
