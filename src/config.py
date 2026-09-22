from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

METADATA_PATH = RAW_DATA_DIR / "HAM10000_metadata.csv"
IMAGE_DIR_PART_1 = RAW_DATA_DIR / "HAM10000_images_part_1"
IMAGE_DIR_PART_2 = RAW_DATA_DIR / "HAM10000_images_part_2"

MODELS_DIR = PROJECT_ROOT / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
FINAL_MODELS_DIR = MODELS_DIR / "final"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"

CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_LABELS = {
    "akiec": "Actinic Keratosis / Intraepithelial Carcinoma",
    "bcc": "Basal Cell Carcinoma",
    "bkl": "Benign Keratosis-like Lesion",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic Nevus",
    "vasc": "Vascular Lesion",
}

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = len(CLASS_NAMES)
RANDOM_STATE = 42

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def ensure_directories() -> None:
    for directory in (
        PROCESSED_DATA_DIR,
        CHECKPOINTS_DIR,
        FINAL_MODELS_DIR,
        FIGURES_DIR,
        METRICS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
