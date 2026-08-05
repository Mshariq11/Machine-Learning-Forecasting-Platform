"""
=========================================================
Dataset Downloader
=========================================================

Downloads required datasets automatically if missing.

Author : Shariq Zia
Project: Store Sales Forecasting
"""

from pathlib import Path
import urllib.request

from src.config import PROCESSED_DATA_DIR


# ==========================================================
# DATASET URLS
# ==========================================================

DATASET_URLS = {

    "train_features.parquet":
    "https://huggingface.co/datasets/ShawRickZia/machine-learning-forecasting-data/resolve/main/train_features.parquet",

    "test_features.parquet":
    "https://huggingface.co/datasets/ShawRickZia/machine-learning-forecasting-data/resolve/main/test_features.parquet",

}


# ==========================================================
# DOWNLOAD ONE FILE
# ==========================================================

def download_file(filename: str):

    destination = PROCESSED_DATA_DIR / filename

    if destination.exists():
        return destination

    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    url = DATASET_URLS[filename]

    urllib.request.urlretrieve(
        url,
        destination
    )

    return destination


# ==========================================================
# DOWNLOAD ALL FILES
# ==========================================================

def download_all():

    paths = {}

    for filename in DATASET_URLS:

        paths[filename] = download_file(
            filename
        )

    return paths


# ==========================================================
# ENSURE TRAIN DATA
# ==========================================================

def ensure_train_data():

    return download_file(
        "train_features.parquet"
    )


# ==========================================================
# ENSURE TEST DATA
# ==========================================================

def ensure_test_data():

    return download_file(
        "test_features.parquet"
    )