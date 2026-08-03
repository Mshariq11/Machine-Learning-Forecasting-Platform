"""
Feature Validation
"""

import pandas as pd


def feature_summary(
    df: pd.DataFrame
):

    return pd.DataFrame({

        "dtype": df.dtypes,

        "missing":

            df.isna().sum(),

        "unique":

            df.nunique()

    })