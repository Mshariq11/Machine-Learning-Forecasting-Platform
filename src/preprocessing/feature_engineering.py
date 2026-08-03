"""
=========================================================
Feature Engineering Module
=========================================================

Reusable feature engineering functions for
Store Sales Forecasting project.

Author : Shariq Zia
Project: Store Sales Forecasting
"""


from typing import Dict

import numpy as np
import pandas as pd



# =========================================================
# HELPER FUNCTIONS
# =========================================================

def _copy_dataframe(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Return safe dataframe copy.
    """

    return df.copy()



# =========================================================
# DATE FEATURE ENGINEERING
# =========================================================

def add_date_features(
    df: pd.DataFrame,
    date_column: str = "date"
) -> pd.DataFrame:
    """
    Create calendar based features.
    """

    df = df.copy()

    df[date_column] = pd.to_datetime(
        df[date_column]
    )


    # Year

    df["year"] = (
        df[date_column]
        .dt.year
    )


    # Month

    df["month"] = (
        df[date_column]
        .dt.month
    )


    # Day

    df["day"] = (
        df[date_column]
        .dt.day
    )


    # Week number

    df["week_of_year"] = (

        df[date_column]

        .dt.isocalendar()

        .week

        .astype(int)

    )


    # Day of week

    df["day_of_week"] = (

        df[date_column]

        .dt.dayofweek

    )


    # Weekend flag

    df["is_weekend"] = (

        df["day_of_week"]

        >= 5

    ).astype(int)


    # Month boundaries

    df["is_month_start"] = (

        df[date_column]

        .dt.is_month_start

    ).astype(int)



    df["is_month_end"] = (

        df[date_column]

        .dt.is_month_end

    ).astype(int)


    return df




# =========================================================
# PAYDAY FEATURE
# =========================================================

def add_payday_feature(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Add Ecuador salary payment period feature.
    """

    df = df.copy()


    df["is_payday"] = (

        (df["day"] == 15)

        |

        (df["is_month_end"] == 1)

    ).astype(int)


    return df




# =========================================================
# HOLIDAY FEATURES
# =========================================================

def add_holiday_features(
    df: pd.DataFrame,
    holidays: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge holiday information.
    """

    df = df.copy()

    holiday_df = holidays.copy()


    holiday_df["date"] = pd.to_datetime(
        holiday_df["date"]
    )


    holiday_columns = [

        "date",

        "type",

        "locale",

        "description"

    ]


    holiday_df = holiday_df[
        [
            col
            for col in holiday_columns
            if col in holiday_df.columns
        ]
    ]


    holiday_df = (
        holiday_df
        .drop_duplicates()
    )


    df = df.merge(

        holiday_df,

        how="left",

        on="date"

    )


    df["is_holiday"] = (

        df["description"]

        .notna()

    ).astype(int)


    return df




# =========================================================
# STORE FEATURE ENGINEERING
# =========================================================

def add_store_features(
    df: pd.DataFrame,
    stores: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge store metadata.
    """

    df = df.copy()


    df = df.merge(

        stores,

        how="left",

        on="store_nbr"

    )


    return df




# =========================================================
# OIL FEATURE ENGINEERING
# =========================================================

def add_oil_features(
    df: pd.DataFrame,
    oil: pd.DataFrame
) -> pd.DataFrame:
    """
    Add oil price information.
    """

    df = df.copy()

    oil_df = oil.copy()


    oil_df["date"] = pd.to_datetime(
        oil_df["date"]
    )


    oil_df = oil_df.rename(

        columns={

            "dcoilwtico":
            "oil_price"

        }

    )


    df = df.merge(

        oil_df[
            [
                "date",
                "oil_price"
            ]
        ],

        how="left",

        on="date"

    )


    df["oil_price"] = (

        df["oil_price"]

        .ffill()

        .bfill()

    )


    return df




# =========================================================
# LAG FEATURE ENGINEERING
# =========================================================

def add_lag_features(
    df: pd.DataFrame,
    lags=None
) -> pd.DataFrame:
    """
    Create historical sales lag features.
    """

    df = df.copy()


    if lags is None:

        lags = [

            1,

            7,

            14,

            28

        ]


    df = df.sort_values(

        [

            "store_nbr",

            "family",

            "date"

        ]

    )


    for lag in lags:

        df[
            f"sales_lag_{lag}"
        ] = (

            df.groupby(

                [

                    "store_nbr",

                    "family"

                ]

            )["sales"]

            .shift(lag)

        )


    return df




# =========================================================
# ROLLING FEATURE ENGINEERING
# =========================================================

def add_rolling_features(
    df: pd.DataFrame,
    windows=None
) -> pd.DataFrame:
    """
    Create rolling sales statistics.
    """

    df = df.copy()


    if windows is None:

        windows = [
            7,
            28
        ]


    df = df.sort_values(
        [
            "store_nbr",
            "family",
            "date"
        ]
    )


    for window in windows:


        df[f"sales_roll_mean_{window}"] = (

            df.groupby(
                [
                    "store_nbr",
                    "family"
                ]
            )["sales"]

            .transform(
                lambda x:
                x.shift(1)
                .rolling(window)
                .mean()
            )

        )


        df[f"sales_roll_std_{window}"] = (

            df.groupby(
                [
                    "store_nbr",
                    "family"
                ]
            )["sales"]

            .transform(
                lambda x:
                x.shift(1)
                .rolling(window)
                .std()
            )

        )


    return df




# =========================================================
# TARGET TRANSFORMATION
# =========================================================

def add_log_target(
    df: pd.DataFrame,
    column: str = "sales"
) -> pd.DataFrame:
    """
    Add log transformed target.
    """

    df = df.copy()


    df["log_sales"] = np.log1p(
        df[column]
    )


    return df




# =========================================================
# STATIC FEATURES
# =========================================================

def build_static_features(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Add static ML features.
    """

    df = df.copy()


    if "onpromotion" in df.columns:

        df["promotion_flag"] = (

            df["onpromotion"]

            > 0

        ).astype(int)


    return df




# =========================================================
# COMPLETE FEATURE BUILDER
# =========================================================

def build_features(
    datasets: Dict[str, pd.DataFrame]
) -> Dict[str, pd.DataFrame]:
    """
    Execute complete feature engineering pipeline.
    """

    train = datasets["train"]

    test = datasets["test"]


    stores = datasets["stores"]

    holidays = datasets["holidays"]

    oil = datasets["oil"]



    # Combine train/test for consistent features

    train["dataset"] = "train"

    test["dataset"] = "test"


    full = pd.concat(

        [

            train,

            test

        ],

        ignore_index=True

    )



    full = add_date_features(
        full
    )


    full = add_payday_feature(
        full
    )


    full = add_store_features(

        full,

        stores

    )


    full = add_holiday_features(

        full,

        holidays

    )


    full = add_oil_features(

        full,

        oil

    )


    # Lag features only from sales history

    full = add_lag_features(
        full
    )


    full = add_rolling_features(
        full
    )


    full = build_static_features(
        full
    )


    full = add_log_target(
        full
    )



    train_features = (

        full[
            full["dataset"]=="train"
        ]

        .drop(
            columns=["dataset"]
        )

    )


    test_features = (

        full[
            full["dataset"]=="test"
        ]

        .drop(
            columns=["dataset","sales"]
        )

    )


    return {

        "train":
            train_features,

        "test":
            test_features,

        "stores":
            stores,

        "oil":
            oil,

        "holidays":
            holidays

    }




# =========================================================
# MASTER FEATURE ENGINEERING FUNCTION
# =========================================================

def run_feature_engineering(
    datasets: Dict[str,pd.DataFrame]
) -> Dict[str,pd.DataFrame]:
    """
    Execute feature engineering pipeline.
    """

    features = build_features(
        datasets
    )


    print("\n" + "="*70)
    print("FEATURE ENGINEERING SUMMARY")
    print("="*70)


    for name, df in features.items():

        print(
            f"{name}: {df.shape}"
        )


    print("\n" + "="*70)
    print("FEATURE ENGINEERING COMPLETED")
    print("="*70)


    return features
