"""
=========================================================
Data Cleaning Module
=========================================================

Reusable data cleaning functions for
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
    Return a safe dataframe copy.
    """

    return df.copy()


def _convert_date(
    df: pd.DataFrame,
    column: str = "date"
) -> pd.DataFrame:
    """
    Convert date column to datetime.
    """

    df = df.copy()

    if column in df.columns:

        df[column] = pd.to_datetime(df[column])

    return df


def _remove_duplicates(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Remove duplicate records.
    """

    return df.drop_duplicates().reset_index(drop=True)


# =========================================================
# COLUMN STANDARDIZATION
# =========================================================

def standardize_columns(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Standardize column names.
    """

    df = df.copy()

    df.columns = (

        df.columns

        .str.strip()

        .str.lower()

        .str.replace(" ", "_")

    )

    return df


# =========================================================
# MISSING VALUE HANDLING
# =========================================================

def handle_missing_numeric(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Fill missing numeric values.
    """

    df = df.copy()

    numeric = df.select_dtypes(
        include=np.number
    ).columns

    for col in numeric:

        df[col] = df[col].fillna(
            df[col].median()
        )

    return df


def handle_missing_categorical(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Fill missing categorical values.
    """

    df = df.copy()

    categorical = df.select_dtypes(
        include=[
            "object",
            "category"
        ]
    ).columns

    for col in categorical:

        if not df[col].mode().empty:

            df[col] = df[col].fillna(
                df[col].mode()[0]
            )

    return df


# =========================================================
# GENERAL CLEANING
# =========================================================

def clean_dataframe(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Apply common cleaning pipeline.
    """

    df = _copy_dataframe(df)

    df = standardize_columns(df)

    df = _convert_date(df)

    df = _remove_duplicates(df)

    df = handle_missing_numeric(df)

    df = handle_missing_categorical(df)

    return df

# =========================================================
# TRAIN DATA CLEANING
# =========================================================

def clean_train_data(
    train: pd.DataFrame
) -> pd.DataFrame:
    """
    Clean training dataset.
    """

    df = clean_dataframe(train)

    # Sales cannot be negative
    if "sales" in df.columns:

        df = df[df["sales"] >= 0]

    # Promotion cannot be negative
    if "onpromotion" in df.columns:

        df["onpromotion"] = (
            df["onpromotion"]
            .clip(lower=0)
            .astype(int)
        )

    df.reset_index(
        drop=True,
        inplace=True
    )

    return df


# =========================================================
# TEST DATA CLEANING
# =========================================================

def clean_test_data(
    test: pd.DataFrame
) -> pd.DataFrame:
    """
    Clean testing dataset.
    """

    df = clean_dataframe(test)

    if "onpromotion" in df.columns:

        df["onpromotion"] = (
            df["onpromotion"]
            .clip(lower=0)
            .astype(int)
        )

    df.reset_index(
        drop=True,
        inplace=True
    )

    return df


# =========================================================
# STORE DATA CLEANING
# =========================================================

def clean_store_data(
    stores: pd.DataFrame
) -> pd.DataFrame:
    """
    Clean store metadata.
    """

    df = clean_dataframe(stores)

    object_columns = [

        "city",

        "state",

        "type"

    ]

    for col in object_columns:

        if col in df.columns:

            df[col] = (

                df[col]

                .astype(str)

                .str.strip()

                .str.title()

            )

    if "cluster" in df.columns:

        df["cluster"] = (
            df["cluster"]
            .astype(int)
        )

    return df


# =========================================================
# OIL DATA CLEANING
# =========================================================

def clean_oil_data(
    oil: pd.DataFrame
) -> pd.DataFrame:
    """
    Clean oil price dataset.
    """

    df = clean_dataframe(oil)

    if "dcoilwtico" in df.columns:

        df["dcoilwtico"] = (

            df["dcoilwtico"]

            .ffill()

            .bfill()

        )

    return df


# =========================================================
# HOLIDAY DATA CLEANING
# =========================================================

def clean_holiday_data(
    holidays: pd.DataFrame
) -> pd.DataFrame:
    """
    Clean holiday dataset.
    """

    df = clean_dataframe(holidays)

    text_columns = [

        "type",

        "locale",

        "locale_name",

        "description"

    ]

    for col in text_columns:

        if col in df.columns:

            df[col] = (

                df[col]

                .astype(str)

                .str.strip()

            )

    if "transferred" in df.columns:

        df["transferred"] = (

            df["transferred"]

            .astype(bool)

        )

    return df


# =========================================================
# DATA TYPE VALIDATION
# =========================================================

def validate_data_types(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Ensure important columns have correct types.
    """

    df = df.copy()

    integer_columns = [

        "store_nbr",

        "onpromotion",

        "cluster"

    ]

    for col in integer_columns:

        if col in df.columns:

            df[col] = df[col].astype(int)

    if "sales" in df.columns:

        df["sales"] = (
            df["sales"]
            .astype(float)
        )

    return df

# =========================================================
# BUSINESS RULE VALIDATION
# =========================================================

def validate_business_rules(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Validate business rules.
    """

    df = df.copy()

    # -----------------------------------
    # Sales
    # -----------------------------------

    if "sales" in df.columns:

        df = df[df["sales"] >= 0]

    # -----------------------------------
    # Promotion
    # -----------------------------------

    if "onpromotion" in df.columns:

        df["onpromotion"] = (

            df["onpromotion"]

            .fillna(0)

            .clip(lower=0)

            .astype(int)

        )

    # -----------------------------------
    # Store Number
    # -----------------------------------

    if "store_nbr" in df.columns:

        df = df[df["store_nbr"] > 0]

    df.reset_index(

        drop=True,

        inplace=True

    )

    return df


# =========================================================
# DATASET VALIDATION REPORT
# =========================================================

def validation_report(
    datasets: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """
    Generate validation summary.
    """

    report = []

    for name, df in datasets.items():

        report.append({

            "Dataset":
                name,

            "Rows":
                len(df),

            "Columns":
                df.shape[1],

            "Missing Values":
                int(df.isna().sum().sum()),

            "Duplicate Rows":
                int(df.duplicated().sum())

        })

    return pd.DataFrame(report)


# =========================================================
# CLEAN ALL DATASETS
# =========================================================

def clean_all_datasets(
    datasets: Dict[str, pd.DataFrame]
) -> Dict[str, pd.DataFrame]:
    """
    Execute cleaning pipeline for
    every dataset.
    """

    cleaned = {

        "train":
            validate_business_rules(
                validate_data_types(
                    clean_train_data(
                        datasets["train"]
                    )
                )
            ),

        "test":
            validate_business_rules(
                validate_data_types(
                    clean_test_data(
                        datasets["test"]
                    )
                )
            ),

        "stores":
            validate_data_types(
                clean_store_data(
                    datasets["stores"]
                )
            ),

        "oil":
            clean_oil_data(
                datasets["oil"]
            ),

        "holidays":
            clean_holiday_data(
                datasets["holidays"]
            )

    }

    return cleaned


# =========================================================
# MASTER CLEANING FUNCTION
# =========================================================

def run_cleaning(
    datasets: Dict[str, pd.DataFrame]
) -> Dict[str, pd.DataFrame]:
    """
    Execute complete data cleaning pipeline.
    """

    cleaned = clean_all_datasets(
        datasets
    )

    print("\n" + "=" * 70)
    print("DATA CLEANING SUMMARY")
    print("=" * 70)

    print(
        validation_report(cleaned)
    )

    print("\n" + "=" * 70)
    print("DATA CLEANING COMPLETED")
    print("=" * 70)

    return cleaned

