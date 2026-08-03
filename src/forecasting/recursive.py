"""
Recursive Forecasting
"""

import pandas as pd


def recursive_forecast(
    model,
    features: pd.DataFrame,
    horizon: int = 16
):
    """
    Recursive forecasting placeholder.

    Future lag features should be updated after each prediction.
    """

    predictions = model.predict(features)

    return predictions