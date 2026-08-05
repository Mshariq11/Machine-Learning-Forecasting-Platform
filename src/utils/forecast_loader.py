"""
=========================================================
Forecast Loader
=========================================================

Automatically:

1. Download test features
2. Download trained model
3. Download feature columns
4. Generate forecast if missing
5. Return forecast dataframe

Author : Shariq Zia
Project : Store Sales Forecasting
"""

import pandas as pd

from src.config import (
    MODEL_DIR,
    PREDICTION_DIR,
)

from src.models.predict import run_prediction

from src.utils.download_data import (
    ensure_test_data,
    ensure_model_files,
)

from src.utils.data_loader import (
    load_test_data,
)


def load_forecast():

    prediction_file = (
        PREDICTION_DIR /
        "sales_prediction.csv"
    )

    if prediction_file.exists():

        return pd.read_csv(
            prediction_file
        )

    # ----------------------------------------
    # Download required files
    # ----------------------------------------

    ensure_test_data()

    ensure_model_files()

    # ----------------------------------------
    # Load engineered test data
    # ----------------------------------------

    test = load_test_data()

    # ----------------------------------------
    # Generate prediction
    # ----------------------------------------

    run_prediction(
        test=test,
        model_path=MODEL_DIR /
        "xgboost_model.pkl",
        feature_path=MODEL_DIR /
        "feature_columns.pkl",
        output_path=prediction_file,
    )

    return pd.read_csv(
        prediction_file
    )