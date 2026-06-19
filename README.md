# CropSenseAI — Complete Step-by-Step Run Instructions

## Prerequisites

| Item | Minimum Version |
|------|-----------------|
| Python | 3.9+ |
| pip | latest |
| RAM | 4 GB (8 GB recommended for large datasets) |
| Disk | ~500 MB for dependencies + your dataset |

---

## STEP 1 — Dataset Setup (Do This First)

Your dataset folder must look **exactly** like this inside `data/raw/`:

```
data/
└── raw/
    ├── cotton/       ← put all cotton images here
    │   ├── img001.jpg
    │   ├── img002.jpg
    │   └── ...
    ├── maize/        ← all maize images here
    ├── rice/         ← all rice images here
    └── wheat/        ← all wheat images here
```

**Folder naming rules:**
- Folder names must be lowercase: `cotton`, `maize`, `rice`, `wheat`
- If your folders are uppercase (e.g. `Rice`, `WHEAT`), the code handles this automatically
- Supported image formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`
- Recommended: at least 100 images per class (200+ for better accuracy)

---

## STEP 2 — Create Virtual Environment

Open a terminal in the `CropSenseAI/` project folder, then run:

### Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

### macOS / Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` appear at the start of your terminal prompt.

---

## STEP 3 — Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs everything needed: numpy, scikit-learn, scikit-image, opencv, streamlit, etc.

**If you see an error about `scikit-image` specifically:**
```bash
pip install scikit-image
```

---

## STEP 4 — Create Logs Directory

```bash
mkdir logs
```

(Already created if you extracted the ZIP — skip if `logs/` exists.)

---

## STEP 5 — Validate Your Dataset (Optional but Recommended)

Before training, check your dataset for corrupt images:

```bash
python scripts/validate_dataset.py
```

Expected output:
```
==================================================
  DATASET VALIDATION REPORT
==================================================
  ✅  cotton      valid= 300  corrupt=  0
  ✅  maize       valid= 287  corrupt=  0
  ✅  rice        valid= 310  corrupt=  0
  ✅  wheat       valid= 295  corrupt=  0
--------------------------------------------------
  Total valid  : 1192
  Total corrupt:    0
  Num classes  :    4
==================================================
```

If you see corrupt images (❌), delete them from the folder before training.

---

## STEP 6 — Train the Model

```bash
python train_crop_model.py
```

This will:
1. Load all images from `data/raw/`
2. Extract HOG + color histogram features from each image
3. Train a Random Forest classifier (200 trees)
4. Evaluate accuracy on a 20% held-out test set
5. Save the model to `data/models/crop_classifier.pkl`
6. Save the class mapping to `data/models/class_mapping.json`

**Expected output (example):**
```
INFO - Images loaded: (1192, 128, 128, 3), Labels: (1192,)
INFO - Feature matrix: 1192 samples × 2208 features
INFO - Train: 953 samples | Test: 239 samples
INFO - Training complete.

=======================================================
  Accuracy : 0.9205  (239 test samples)
=======================================================
              precision    recall  f1-score   support
      cotton       0.94      0.91      0.92        60
       maize       0.93      0.95      0.94        57
        rice       0.91      0.90      0.91        62
       wheat       0.91      0.93      0.92        60
```

Training time: ~1–3 minutes depending on dataset size and your CPU.

**Optional flags:**
```bash
python train_crop_model.py --raw-data data/raw --output data/models/crop_classifier.pkl --test-size 0.2
```

---

## STEP 7 — Run the Dashboard

```bash
streamlit run main.py
```

The browser will open automatically at: **http://localhost:8501**

If it does not open, copy the URL from the terminal and paste it in your browser.

**Dashboard pages:**
- 🏠 **Home** — System status, quick overview
- 🍃 **Crop Classification** — Upload an image → get crop prediction
- 📊 **Stress Analysis** — Input vegetation index / temperature / rainfall → detect stress
- 💧 **Irrigation Recommendations** — Get irrigation action with water depth
- 📈 **History & Analytics** — Charts of all predictions made this session

---

## STEP 8 — Predict From Command Line (Optional)

After training, you can predict a single image without the dashboard:

```bash
python predict_crop.py path/to/your/image.jpg
```

With verbose output:
```bash
python predict_crop.py path/to/your/image.jpg --verbose
```

Expected output:
```
Predicted crop: rice
```

---

## STEP 9 — Run Tests

```bash
pytest tests/ -v
```

Expected:
```
tests/test_crop_classification.py::TestHOGFeatures::test_output_is_1d PASSED
tests/test_crop_classification.py::TestColorHistogram::test_output_length PASSED
...
tests/test_irrigation_recommendation.py::TestIrrigationRecommendation::... PASSED
```

Run with coverage report:
```bash
pytest tests/ --cov=cropsenseai --cov-report=term-missing
```

---

## STEP 10 — Generate Dataset Report (Optional)

```bash
python scripts/generate_dataset_report.py
```

Saves a CSV to `data/processed/dataset_report.csv` with per-class image counts.

---

## Troubleshooting

### "Module not found: cropsenseai"
Make sure you are in the project root directory (where `main.py` is) and your virtual environment is activated.

```bash
# Check you are in the right folder
ls main.py  # should show main.py

# Check virtual environment is active — you should see (venv)
```

### "No valid class folders found in data/raw"
Check that your dataset folders are inside `data/raw/` and named correctly:
```bash
ls data/raw/
# Should show: cotton  maize  rice  wheat
```

### "Model not found: data/models/crop_classifier.pkl"
Run training first:
```bash
python train_crop_model.py
```

### "scikit-image not installed"
```bash
pip install scikit-image
```

### Port already in use
```bash
streamlit run main.py --server.port 8502
```

---

## Full Command Summary (Copy-Paste)

```bash
# 1. Activate environment
source venv/bin/activate          # macOS/Linux
# OR
venv\Scripts\activate             # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Validate dataset
python scripts/validate_dataset.py

# 4. Train model
python train_crop_model.py

# 5. Run dashboard
streamlit run main.py

# 6. Run tests
pytest tests/ -v
```
