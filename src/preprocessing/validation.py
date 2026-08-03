"""
=========================================================
Data Validation Module
=========================================================

Reusable validation functions for
Store Sales Forecasting project.

Author : Shariq Zia
Project: Store Sales Forecasting
"""


from typing import Dict, List

import pandas as pd
import numpy as np

# =========================================================
# REQUIRED COLUMN VALIDATION
# =========================================================

def validate_required_columns(
    df: pd.DataFrame,
    required_columns: List[str]
) -> Dict:
    """
    Check required columns exist.
    """

    missing = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    return {
        "valid": len(missing) == 0,
        "missing_columns": missing
    }

# =========================================================
# MISSING VALUE VALIDATION
# =========================================================

def missing_value_report(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Generate missing value report.
    """

    report = pd.DataFrame({

        "missing_count":
            df.isna().sum(),

        "missing_percentage":
            (
                df.isna().mean()
                *100
            )

    })

    return (
        report
        .sort_values(
            "missing_count",
            ascending=False
        )
    )

# =========================================================
# DUPLICATE VALIDATION
# =========================================================

def duplicate_report(
    df: pd.DataFrame
) -> Dict:
    """
    Check duplicate rows.
    """

    duplicates = df.duplicated().sum()

    return {

        "duplicates": int(duplicates),

        "valid": duplicates == 0

    }

# =========================================================
# DATA TYPE VALIDATION
# =========================================================

def datatype_report(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Return dataframe data types.
    """

    return pd.DataFrame({

        "column":
            df.columns,

        "datatype":
            df.dtypes.astype(str)

    })

# =========================================================
# SALES BUSINESS VALIDATION
# =========================================================

def sales_business_validation(
    df: pd.DataFrame
) -> Dict:
    """
    Validate sales business rules.
    """

    result = {}


    if "sales" in df.columns:

        result["negative_sales"] = int(
            (df["sales"] < 0).sum()
        )


    if "onpromotion" in df.columns:

        result["negative_promotions"] = int(
            (df["onpromotion"] < 0).sum()
        )


    if "store_nbr" in df.columns:

        result["invalid_stores"] = int(
            (df["store_nbr"] <= 0).sum()
        )


    result["valid"] = all(
        value == 0
        for value in result.values()
        if isinstance(value,int)
    )


    return result

# =========================================================
# TRAIN TEST FEATURE VALIDATION
# =========================================================

def compare_train_test_columns(
    train: pd.DataFrame,
    test: pd.DataFrame
) -> Dict:
    """
    Compare train and test columns.
    """

    train_columns = set(train.columns)

    test_columns = set(test.columns)


    return {

        "missing_in_test":
            list(
                train_columns-test_columns
            ),

        "missing_in_train":
            list(
                test_columns-train_columns
            ),

        "compatible":
            train_columns == test_columns

    }

# =========================================================
# FEATURE VALIDATION
# =========================================================

def validate_features(
    df: pd.DataFrame
) -> Dict:
    """
    Validate ML feature dataset.
    """

    numeric_columns = (
        df
        .select_dtypes(
            include=np.number
        )
        .columns
    )


    return {

        "rows":
            len(df),

        "features":
            len(df.columns),

        "numeric_features":
            len(numeric_columns),

        "missing_values":
            int(
                df.isna()
                .sum()
                .sum()
            )

    }

# =========================================================
# MASTER VALIDATION FUNCTION
# =========================================================

def run_validation(
    datasets: Dict[str,pd.DataFrame]
) -> Dict:
    """
    Execute complete validation pipeline.
    """

    report = {}


    for name, df in datasets.items():

        report[name] = {

            "shape":
                df.shape,

            "duplicates":
                duplicate_report(df),

            "missing":
                missing_value_report(df),

            "datatype":
                datatype_report(df)

        }


    print("\n")
    print("="*70)
    print("VALIDATION COMPLETED")
    print("="*70)


    return report

