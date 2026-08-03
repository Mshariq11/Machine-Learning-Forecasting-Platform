"""
External Feature Engineering
"""

import pandas as pd


def add_external_features(
    df: pd.DataFrame
) -> pd.DataFrame:

    data = df.copy()

    if "dcoilwtico" in data.columns:

        data["oil_missing"] = (
            data["dcoilwtico"]
            .isna()
            .astype(int)
        )

    if "type" in data.columns:

        data["is_holiday"] = (
            data["type"]
            .notna()
            .astype(int)
        )

    return data