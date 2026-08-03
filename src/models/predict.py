"""
=========================================================
Prediction Module
=========================================================

Generate sales predictions using trained forecasting model.

Author : Shariq Zia
Project: Store Sales Forecasting
"""

from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd

from src.config import MODEL_DIR


# =========================================================
# LOAD MODEL
# =========================================================

def load_model(
    model_path: Path
):
    """
    Load trained machine learning model.
    """

    model = joblib.load(
        model_path
    )

    return model



# =========================================================
# LOAD FEATURE COLUMNS
# =========================================================

def load_feature_columns(
    path: Path
):
    """
    Load training feature columns.
    """

    features = joblib.load(
        path
    )

    return features


# =========================================================
# LOAD CATEGORY ENCODERS
# =========================================================

def load_encoders(
    path: Path
) -> Dict:
    """
    Load saved LabelEncoders.
    """

    return joblib.load(path)

# =========================================================
# PREPARE TEST FEATURES
# =========================================================

def prepare_prediction_data(
    test: pd.DataFrame,
    feature_columns: list
) -> pd.DataFrame:
    """
    Select model features only.
    """

    X_test = test.copy()


    # Remove target columns if present

    remove_columns = [

        "sales",

        "log_sales"

    ]


    for col in remove_columns:

        if col in X_test.columns:

            X_test = X_test.drop(
                columns=[col]
            )


    # Align with training features

    X_test = X_test.reindex(
        columns=feature_columns,
        fill_value=0
    )


    return X_test

# =========================================================
# APPLY CATEGORY ENCODERS
# =========================================================

def apply_category_encoders(
    df: pd.DataFrame,
    encoders: Dict
) -> pd.DataFrame:
    """
    Apply LabelEncoders learned during training.

    Unknown categories are encoded as -1.
    """

    df = df.copy()

    for column, encoder in encoders.items():

        if column not in df.columns:
            continue

        mapping = {
            cls: idx
            for idx, cls in enumerate(encoder.classes_)
        }

        df[column] = (
            df[column]
            .astype(str)
            .map(mapping)
            .fillna(-1)
            .astype(int)
        )

    return df

# =========================================================
# GENERATE PREDICTIONS
# =========================================================

def generate_predictions(
    model,
    X_test
) -> np.ndarray:
    """
    Generate sales predictions.
    """

    predictions = model.predict(
        X_test
    )


    # Reverse log transformation

    predictions = np.expm1(
        predictions
    )


    # Sales cannot be negative

    predictions = np.clip(
        predictions,
        0,
        None
    )


    return predictions



# =========================================================
# SAVE PREDICTIONS
# =========================================================

def save_predictions(
    test: pd.DataFrame,
    predictions: np.ndarray,
    output_path: Path
) -> pd.DataFrame:
    """
    Save prediction results.

    Returns a business-ready dataframe containing
    original features together with the forecast.
    """

    result = test.copy()

    result["Forecast"] = predictions

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    result.to_csv(
        output_path,
        index=False
    )

    return result



# =========================================================
# COMPLETE PREDICTION PIPELINE
# =========================================================

def run_prediction(
    test: pd.DataFrame,
    model_path: Path,
    feature_path: Path,
    output_path: Path
) -> pd.DataFrame:
    """
    Execute complete prediction pipeline.
    """

    print("\n" + "=" * 70)
    print("STARTING PREDICTION")
    print("=" * 70)

    # ---------------------------------
    # Load model
    # ---------------------------------

    model = load_model(
        model_path
    )

    # ---------------------------------
    # Load feature columns
    # ---------------------------------

    feature_columns = load_feature_columns(
        feature_path
    )

    # ---------------------------------
    # Load category encoders
    # ---------------------------------

    encoders = load_encoders(
        MODEL_DIR / "category_encoders.pkl"
    )

    # ---------------------------------
    # Prepare features
    # ---------------------------------

    X_test = prepare_prediction_data(
        test,
        feature_columns
    )

    # ---------------------------------
    # Encode categorical columns
    # ---------------------------------

    X_test = apply_category_encoders(
        X_test,
        encoders
    )

    # ---------------------------------
    # Predict
    # ---------------------------------

    predictions = generate_predictions(
        model,
        X_test
    )

    # ---------------------------------
    # Save predictions
    # ---------------------------------

    result = save_predictions(
        test,
        predictions,
        output_path
    )

    print(
        f"Prediction shape: {result.shape}"
    )

    print("\n" + "=" * 70)
    print("PREDICTION COMPLETED")
    print("=" * 70)

    return result