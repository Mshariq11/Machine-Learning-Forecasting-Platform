"""
=========================================================
Data Loader
=========================================================

Loads processed datasets after ensuring they exist.

Author : Shariq Zia
Project: Store Sales Forecasting
"""

import pandas as pd

from src.config import PROCESSED_DATA_DIR
from src.utils.download_data import (
    ensure_train_data,
    ensure_test_data,
)


# ==========================================================
# TRAIN DATA
# ==========================================================

def load_train_data():

    ensure_train_data()

    return pd.read_parquet(
        PROCESSED_DATA_DIR /
        "train_features.parquet"
    )


# ==========================================================
# TEST DATA
# ==========================================================

def load_test_data():

    ensure_test_data()

    return pd.read_parquet(
        PROCESSED_DATA_DIR /
        "test_features.parquet"
    )