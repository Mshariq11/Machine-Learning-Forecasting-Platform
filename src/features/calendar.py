"""
Calendar Feature Engineering
"""

import pandas as pd


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create calendar based features.
    """

    data = df.copy()

    data["year"] = data["date"].dt.year
    data["quarter"] = data["date"].dt.quarter
    data["month"] = data["date"].dt.month
    data["week"] = data["date"].dt.isocalendar().week.astype(int)
    data["day"] = data["date"].dt.day
    data["day_of_week"] = data["date"].dt.dayofweek
    data["day_of_year"] = data["date"].dt.dayofyear

    data["is_weekend"] = (
        data["day_of_week"] >= 5
    ).astype(int)

    data["is_month_start"] = (
        data["date"].dt.is_month_start
    ).astype(int)

    data["is_month_end"] = (
        data["date"].dt.is_month_end
    ).astype(int)

    return data