"""
Lag Feature Engineering
"""

import pandas as pd


def add_lag_features(
    df: pd.DataFrame,
    lags=(1, 7, 14, 28)
) -> pd.DataFrame:

    data = df.copy()

    group = ["store_nbr", "family"]

    for lag in lags:

        data[f"sales_lag_{lag}"] = (

            data
            .groupby(group)["sales"]
            .shift(lag)

        )

    return data