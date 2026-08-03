"""
Forecast Inference
"""

import pandas as pd


def predict(model, features: pd.DataFrame):
    """
    Generate predictions from a trained model.
    """
    return model.predict(features)