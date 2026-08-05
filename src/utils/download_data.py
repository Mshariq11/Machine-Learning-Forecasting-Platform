"""
=========================================================
Download Project Assets
=========================================================

Downloads datasets and model artifacts from Hugging Face.

Author : Shariq Zia
Project: Store Sales Forecasting
"""

import urllib.request

from src.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    MODEL_DIR,
)

# ==========================================================
# HUGGING FACE URLS
# ==========================================================

DATASET_URL = (
    "https://huggingface.co/datasets/"
    "ShawRickZia/machine-learning-forecasting-data/resolve/main/"
)

MODEL_URL = (
    "https://huggingface.co/"
    "ShawRickZia/store-sales-forecasting-models/resolve/main/"
)

# ==========================================================
# FILES
# ==========================================================

DATA_FILES = {
    "train_features.parquet": PROCESSED_DATA_DIR,
    "test_features.parquet": PROCESSED_DATA_DIR,
}

MODEL_FILES = {
    "xgboost_model.pkl": MODEL_DIR,
    "lightgbm_model.pkl": MODEL_DIR,
    "feature_columns.pkl": MODEL_DIR,
    "category_encoders.pkl": MODEL_DIR,
    "training_metadata.pkl": MODEL_DIR,
}

# ==========================================================
# DOWNLOAD HELPER
# ==========================================================

def download_file(url: str, destination):

    if destination.exists():
        return

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    urllib.request.urlretrieve(
        url,
        destination,
    )

# ==========================================================
# DOWNLOAD DATASETS
# ==========================================================

def download_datasets():

    for filename, folder in DATA_FILES.items():

        download_file(
            DATASET_URL + filename,
            folder / filename,
        )

# ==========================================================
# DOWNLOAD MODELS
# ==========================================================

def download_models():

    for filename, folder in MODEL_FILES.items():

        download_file(
            MODEL_URL + filename,
            folder / filename,
        )

# ==========================================================
# DOWNLOAD EVERYTHING
# ==========================================================

def download_everything():

    download_datasets()

    download_models()