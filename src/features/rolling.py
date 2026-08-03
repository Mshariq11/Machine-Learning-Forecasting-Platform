"""
Rolling Window Features
"""

import pandas as pd


def add_rolling_features(
    df: pd.DataFrame,
    windows=(7, 28)
) -> pd.DataFrame:

    data = df.copy()

    group = ["store_nbr", "family"]

    for window in windows:

        rolling = (

            data
            .groupby(group)["sales"]
            .transform(

                lambda x:

                x.shift(1)
                 .rolling(window)
                 .mean()

            )

        )

        data[f"rolling_mean_{window}"] = rolling

        rolling_std = (

            data
            .groupby(group)["sales"]
            .transform(

                lambda x:

                x.shift(1)
                 .rolling(window)
                 .std()

            )

        )

        data[f"rolling_std_{window}"] = rolling_std

    return data