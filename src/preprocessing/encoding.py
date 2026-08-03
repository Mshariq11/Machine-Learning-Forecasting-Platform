"""
=========================================================
Data Encoding Module
=========================================================

Reusable encoding functions for
Store Sales Forecasting project.

Author : Shariq Zia
Project: Store Sales Forecasting
"""

from typing import Dict, Tuple

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
# LABEL ENCODING
# =========================================================

def label_encode(
    df: pd.DataFrame,
    column: str
) -> Tuple[pd.DataFrame, Dict]:
    """
    Label encode a categorical column.

    Returns
    -------
    encoded dataframe
    mapping dictionary
    """

    df = df.copy()

    categories = sorted(
        df[column]
        .astype(str)
        .unique()
    )

    mapping = {

        category: idx

        for idx, category

        in enumerate(categories)

    }

    df[column] = (

        df[column]

        .astype(str)

        .map(mapping)

        .astype(int)

    )

    return df, mapping


# =========================================================
# APPLY LABEL ENCODING
# =========================================================

def apply_label_mapping(
    df: pd.DataFrame,
    column: str,
    mapping: Dict
) -> pd.DataFrame:
    """
    Apply an existing label mapping.
    """

    df = df.copy()

    df[column] = (

        df[column]

        .astype(str)

        .map(mapping)

        .fillna(-1)

        .astype(int)

    )

    return df


# =========================================================
# FREQUENCY ENCODING
# =========================================================

def frequency_encode(
    df: pd.DataFrame,
    column: str
) -> Tuple[pd.DataFrame, Dict]:
    """
    Frequency encode a column.
    """

    df = df.copy()

    frequency = (

        df[column]

        .value_counts()

        .to_dict()

    )

    df[column + "_freq"] = (

        df[column]

        .map(frequency)

    )

    return df, frequency


# =========================================================
# APPLY FREQUENCY ENCODING
# =========================================================

def apply_frequency_mapping(
    df: pd.DataFrame,
    column: str,
    mapping: Dict
) -> pd.DataFrame:
    """
    Apply existing frequency mapping.
    """

    df = df.copy()

    df[column + "_freq"] = (

        df[column]

        .map(mapping)

        .fillna(0)

    )

    return df


# =========================================================
# ONE HOT ENCODING
# =========================================================

def one_hot_encode(
    df: pd.DataFrame,
    columns: list
) -> pd.DataFrame:
    """
    Apply one-hot encoding.
    """

    return pd.get_dummies(

        df,

        columns=columns,

        drop_first=False,

        dtype=int

    )

# =========================================================
# CYCLICAL ENCODING
# =========================================================

def cyclical_encode(
    df: pd.DataFrame,
    column: str,
    max_value: int
) -> pd.DataFrame:
    """
    Apply sine and cosine transformation
    for cyclic features.

    Example:
    - Month
    - Weekday
    - Day of year
    """

    df = df.copy()

    df[column + "_sin"] = (

        np.sin(
            2 * np.pi * df[column] / max_value
        )

    )

    df[column + "_cos"] = (

        np.cos(
            2 * np.pi * df[column] / max_value
        )

    )

    return df


# =========================================================
# DATE FEATURE ENCODING
# =========================================================

def encode_date_features(
    df: pd.DataFrame,
    date_column: str = "date"
) -> pd.DataFrame:
    """
    Create encoded calendar features.
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


    # Week

    df["week_of_year"] = (

        df[date_column]

        .dt.isocalendar()

        .week

        .astype(int)

    )


    # Day

    df["day"] = (
        df[date_column]
        .dt.day
    )


    # Weekday

    df["day_of_week"] = (

        df[date_column]

        .dt.dayofweek

    )


    # Weekend Flag

    df["is_weekend"] = (

        df["day_of_week"]

        >= 5

    ).astype(int)


    # Cyclic transformations

    df = cyclical_encode(

        df,

        "month",

        12

    )


    df = cyclical_encode(

        df,

        "day_of_week",

        7

    )


    df = cyclical_encode(

        df,

        "week_of_year",

        52

    )


    return df


# =========================================================
# BINARY FEATURE ENCODING
# =========================================================

def binary_encode(
    df: pd.DataFrame,
    column: str,
    true_value=1,
    false_value=0
) -> pd.DataFrame:
    """
    Convert boolean categories into binary values.
    """

    df = df.copy()

    df[column] = (

        df[column]

        .map({

            True: true_value,

            False: false_value

        })

        .fillna(false_value)

        .astype(int)

    )

    return df


# =========================================================
# STORE FEATURE ENCODING
# =========================================================

def encode_store_features(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Encode store related features.
    """

    df = df.copy()

    categorical_columns = [

        "city",

        "state",

        "type"

    ]


    available = [

        col

        for col in categorical_columns

        if col in df.columns

    ]


    if available:

        df = one_hot_encode(

            df,

            available

        )


    return df


# =========================================================
# PRODUCT FAMILY ENCODING
# =========================================================

def encode_product_family(
    df: pd.DataFrame
) -> Tuple[pd.DataFrame, Dict]:
    """
    Encode product family using label encoding.
    """

    if "family" not in df.columns:

        return df, {}


    return label_encode(

        df,

        "family"

    )

# =========================================================
# TRAIN / TEST CONSISTENT ENCODING
# =========================================================

def fit_encoding_mapping(
    df: pd.DataFrame,
    columns: list
) -> Dict:
    """
    Create encoding mappings from training data.
    """

    mappings = {}

    for column in columns:

        if column in df.columns:

            mappings[column] = (

                df[column]

                .astype(str)

                .value_counts()

                .to_dict()

            )

    return mappings


# =========================================================
# APPLY ENCODING MAPPINGS
# =========================================================

def apply_encoding_mappings(
    df: pd.DataFrame,
    mappings: Dict
) -> pd.DataFrame:
    """
    Apply saved mappings to dataset.
    """

    df = df.copy()

    for column, mapping in mappings.items():

        if column in df.columns:

            df[column + "_freq"] = (

                df[column]

                .astype(str)

                .map(mapping)

                .fillna(0)

            )

    return df


# =========================================================
# ENCODE TRAIN DATA
# =========================================================

def encode_train_data(
    train: pd.DataFrame
) -> Tuple[pd.DataFrame, Dict]:
    """
    Encode training dataset.
    """

    df = train.copy()

    mappings = {}


    # Date features

    if "date" in df.columns:

        df = encode_date_features(
            df
        )


    # Product family

    if "family" in df.columns:

        df, family_mapping = encode_product_family(
            df
        )

        mappings["family"] = family_mapping


    # Frequency encoding

    frequency_columns = [

        "store_nbr",

        "family"

    ]


    frequency_columns = [

        col

        for col in frequency_columns

        if col in df.columns

    ]


    frequency_mapping = fit_encoding_mapping(

        df,

        frequency_columns

    )


    df = apply_encoding_mappings(

        df,

        frequency_mapping

    )


    mappings["frequency"] = frequency_mapping


    return df, mappings


# =========================================================
# ENCODE TEST DATA
# =========================================================

def encode_test_data(
    test: pd.DataFrame,
    mappings: Dict
) -> pd.DataFrame:
    """
    Encode test dataset using
    training mappings.
    """

    df = test.copy()


    if "date" in df.columns:

        df = encode_date_features(
            df
        )


    if "frequency" in mappings:

        df = apply_encoding_mappings(

            df,

            mappings["frequency"]

        )


    if "family" in mappings:

        if "family" in df.columns:

            df["family"] = (

                df["family"]

                .astype(str)

                .map(
                    mappings["family"]
                )

                .fillna(-1)

                .astype(int)

            )


    return df


# =========================================================
# ENCODE COMPLETE DATASETS
# =========================================================

def encode_all_datasets(
    datasets: Dict[str, pd.DataFrame]
) -> Dict[str, pd.DataFrame]:
    """
    Execute complete encoding pipeline.
    """

    encoded = {}


    train_encoded, mappings = encode_train_data(

        datasets["train"]

    )


    encoded["train"] = train_encoded


    encoded["test"] = encode_test_data(

        datasets["test"],

        mappings

    )


    encoded["stores"] = datasets["stores"].copy()

    encoded["oil"] = datasets["oil"].copy()

    encoded["holidays"] = datasets["holidays"].copy()


    return encoded


# =========================================================
# MASTER ENCODING FUNCTION
# =========================================================

def run_encoding(
    datasets: Dict[str, pd.DataFrame]
) -> Dict[str, pd.DataFrame]:
    """
    Execute encoding pipeline.
    """

    encoded = encode_all_datasets(
        datasets
    )


    print("\n" + "=" * 70)
    print("ENCODING SUMMARY")
    print("=" * 70)


    for name, df in encoded.items():

        print(
            f"{name}: {df.shape}"
        )


    print("\n" + "=" * 70)
    print("ENCODING COMPLETED")
    print("=" * 70)


    return encoded

