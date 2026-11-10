from pathlib import Path


# ============================================================
# PROJECT ROOT
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent


# ============================================================
# DATASET
# ============================================================

DATASET_DIR = BASE_DIR / "dataset"

RAW_DIR = DATASET_DIR / "raw"

IMAGE_DIR = RAW_DIR / "images"

PROCESSED_DIR = DATASET_DIR / "processed"

SELECTED_IMAGE_DIR = DATASET_DIR / "selected_images"


# Original metadata CSV
CSV_FILE = RAW_DIR / "Data_Entry_2017_v2020.csv"


# Reduced dataset CSV
SELECTED_CSV_FILE = DATASET_DIR / "selected_images.csv"


# ============================================================
# MODELS
# ============================================================

MODEL_DIR = BASE_DIR / "models"


# ============================================================
# RESULTS
# ============================================================

RESULTS_DIR = BASE_DIR / "results"

PLOTS_DIR = RESULTS_DIR / "plots"

METRICS_DIR = RESULTS_DIR / "metrics"

PREDICTIONS_DIR = RESULTS_DIR / "predictions"


# ============================================================
# IMAGE SETTINGS
# ============================================================

IMAGE_SIZE = (224, 224)


# ============================================================
# TRAINING SETTINGS
# ============================================================

BATCH_SIZE = 8

EPOCHS = 5

LEARNING_RATE = 0.0001


# ============================================================
# DISEASE LABELS
# ============================================================

DISEASES = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Effusion",
    "Emphysema",
    "Fibrosis",
    "Hernia",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pleural Thickening",
    "Pneumonia",
    "Pneumothorax"
]