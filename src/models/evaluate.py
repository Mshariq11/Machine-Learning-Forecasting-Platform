"""
=========================================================
Model Evaluation Module
=========================================================

Evaluation functions for Store Sales Forecasting models.

Author : Shariq Zia
Project: Store Sales Forecasting
"""

from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_squared_log_error
)


# =========================================================
# LOAD MODEL
# =========================================================

def load_model(
    model_path: Path
):
    """
    Load trained model.
    """

    model = joblib.load(
        model_path
    )

    return model



# =========================================================
# PREDICTION
# =========================================================

def predict_model(
    model,
    X_valid: pd.DataFrame
) -> np.ndarray:
    """
    Generate predictions.
    """

    predictions = model.predict(
        X_valid
    )

    return predictions



# =========================================================
# REVERSE LOG TRANSFORMATION
# =========================================================

def inverse_log_transform(
    values: np.ndarray
) -> np.ndarray:
    """
    Convert log predictions back
    to original sales scale.
    """

    values = np.expm1(
        values
    )

    values = np.clip(
        values,
        0,
        None
    )

    return values



# =========================================================
# RMSLE
# =========================================================

def calculate_rmsle(
    y_true,
    y_pred
) -> float:
    """
    Calculate Root Mean Squared Log Error.
    """

    y_true = np.clip(
        y_true,
        0,
        None
    )

    y_pred = np.clip(
        y_pred,
        0,
        None
    )


    rmsle = np.sqrt(
        mean_squared_log_error(
            y_true,
            y_pred
        )
    )

    return rmsle



# =========================================================
# RMSE
# =========================================================

def calculate_rmse(
    y_true,
    y_pred
) -> float:
    """
    Calculate RMSE.
    """

    return np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )



# =========================================================
# MAE
# =========================================================

def calculate_mae(
    y_true,
    y_pred
) -> float:
    """
    Calculate MAE.
    """

    return mean_absolute_error(
        y_true,
        y_pred
    )



# =========================================================
# COMPLETE METRIC REPORT
# =========================================================

def evaluation_report(
    y_true,
    y_pred,
    model_name="Model"
) -> pd.DataFrame:
    """
    Generate evaluation summary.
    """

    report = {

        "Model":
            model_name,

        "RMSLE":
            calculate_rmsle(
                y_true,
                y_pred
            ),

        "RMSE":
            calculate_rmse(
                y_true,
                y_pred
            ),

        "MAE":
            calculate_mae(
                y_true,
                y_pred
            )

    }


    return pd.DataFrame(
        [report]
    )



# =========================================================
# MODEL EVALUATION PIPELINE
# =========================================================

def run_evaluation(
    model_path: Path,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    model_name="XGBoost",
    log_target=True
) -> pd.DataFrame:
    """
    Execute complete evaluation pipeline.
    """


    print("\n" + "=" * 70)
    print("MODEL EVALUATION")
    print("=" * 70)


    # Load model

    model = load_model(
        model_path
    )


    # Predict

    predictions = predict_model(
        model,
        X_valid
    )


    # Reverse log scale

    if log_target:

        predictions = inverse_log_transform(
            predictions
        )

        y_valid = inverse_log_transform(
            y_valid.values
        )


    report = evaluation_report(
        y_valid,
        predictions,
        model_name
    )


    print(report)


    print("\n" + "=" * 70)
    print("EVALUATION COMPLETED")
    print("=" * 70)


    return report



# =========================================================
# COMPARE MULTIPLE MODELS
# =========================================================

def compare_models(
    models: Dict,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    log_target=True
) -> pd.DataFrame:
    """
    Compare multiple trained models.
    """

    results = []


    for name, model_path in models.items():

        report = run_evaluation(
            model_path=model_path,
            X_valid=X_valid,
            y_valid=y_valid,
            model_name=name,
            log_target=log_target
        )

        results.append(
            report
        )


    return pd.concat(
        results,
        ignore_index=True
    )