"""
=========================================================
Baseline Models
=========================================================

Simple forecasting baselines for comparison.

Author : Shariq Zia
Project: Store Sales Forecasting
"""


import numpy as np
import pandas as pd


# =========================================================
# NAIVE WEEKLY BASELINE
# =========================================================

def weekly_naive_prediction(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Predict sales using previous week's sales.

    Formula:
    sales(t) = sales(t-7)

    """

    df = df.copy()

    df["baseline_prediction"] = (
        df["sales_lag_7"]
    )

    df["baseline_prediction"] = (
        df["baseline_prediction"]
        .fillna(0)
    )

    return df



# =========================================================
# RMSLE
# =========================================================

def rmsle(
    y_true,
    y_pred
):

    y_pred = np.maximum(
        y_pred,
        0
    )

    return np.sqrt(
        np.mean(
            (
                np.log1p(y_pred)
                -
                np.log1p(y_true)
            ) ** 2
        )
    )



# =========================================================
# EVALUATE BASELINE
# =========================================================

def evaluate_baseline(
    df: pd.DataFrame
):

    result = weekly_naive_prediction(df)


    score = rmsle(
        result["sales"],
        result["baseline_prediction"]
    )


    print(
        "="*60
    )

    print(
        f"Weekly Naive RMSLE: {score:.4f}"
    )

    print(
        "="*60
    )


    return score