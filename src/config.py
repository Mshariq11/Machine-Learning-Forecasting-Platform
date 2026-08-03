"""
Project Configuration
---------------------

Centralized configuration used across the project.
"""

from pathlib import Path

# ---------------------------------------------------------------------
# Project Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODEL_DIR = PROJECT_ROOT / "models"

REPORT_DIR = PROJECT_ROOT / "reports"

DASHBOARD_DIR = PROJECT_ROOT / "dashboard"

# ---------------------------------------------------------------------
# Random Seed
# ---------------------------------------------------------------------

RANDOM_STATE = 42

# ---------------------------------------------------------------------
# Model Parameters
# ---------------------------------------------------------------------

VALIDATION_DAYS = 16

TARGET = "sales"

DATE_COLUMN = "date"

STORE_ID = "store_nbr"

FAMILY = "family"

# =========================================================
# OUTPUT DIRECTORIES
# =========================================================

MODEL_DIR = PROJECT_ROOT / "models"

REPORT_DIR = PROJECT_ROOT / "reports"

PREDICTION_DIR = PROJECT_ROOT / "predictions"

PROCESSED_DATA_DIR = DATA_DIR / "processed"


# Create directories automatically
for directory in [
    MODEL_DIR,
    REPORT_DIR,
    PREDICTION_DIR,
    PROCESSED_DATA_DIR,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True
    )