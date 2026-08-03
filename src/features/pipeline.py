"""
Feature Engineering Pipeline
"""

from .calendar import add_calendar_features
from .lag import add_lag_features
from .rolling import add_rolling_features
from .external import add_external_features
from .encoding import encode_categorical_features


def build_features(df):
    """
    Complete feature engineering pipeline.
    """

    data = add_calendar_features(df)

    data = add_lag_features(data)

    data = add_rolling_features(data)

    data = add_external_features(data)

    data, mappings = encode_categorical_features(data)

    return data, mappings