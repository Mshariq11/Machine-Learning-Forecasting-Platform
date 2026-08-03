"""
=========================================================
Statistical Analysis Module
=========================================================

Reusable statistical analysis functions for
retail demand forecasting.

Author : Shariq Zia
Project: Store Sales Forecasting
"""

from typing import Dict

import numpy as np
import pandas as pd


# =========================================================
# DESCRIPTIVE STATISTICS
# =========================================================

def descriptive_statistics(
    train: pd.DataFrame,
    column: str = "sales"
) -> pd.DataFrame:
    """
    Generate descriptive statistics for a numerical column.

    Parameters
    ----------
    train : pd.DataFrame

    column : str

    Returns
    -------
    pd.DataFrame
    """

    series = train[column]

    summary = {

        "Count": series.count(),

        "Mean": series.mean(),

        "Median": series.median(),

        "Mode": series.mode().iloc[0],

        "Minimum": series.min(),

        "Maximum": series.max(),

        "Range": series.max() - series.min(),

        "Variance": series.var(),

        "Standard Deviation": series.std(),

        "Skewness": series.skew(),

        "Kurtosis": series.kurt(),

        "Missing": series.isna().sum()

    }

    return (
        pd.DataFrame(summary, index=["Value"])
        .T
        .rename(columns={"Value": column})
    )


# =========================================================
# QUANTILES
# =========================================================

def quantile_statistics(
    train: pd.DataFrame,
    column: str = "sales"
) -> pd.DataFrame:
    """
    Calculate distribution quantiles.
    """

    series = train[column]

    summary = {

        "Minimum": series.min(),

        "Q1 (25%)": series.quantile(0.25),

        "Median": series.median(),

        "Q3 (75%)": series.quantile(0.75),

        "Maximum": series.max(),

        "IQR": (
            series.quantile(0.75)
            - series.quantile(0.25)
        )

    }

    return (
        pd.DataFrame(summary, index=["Value"])
        .T
        .rename(columns={"Value": column})
    )


# =========================================================
# ZERO SALES ANALYSIS
# =========================================================

def zero_sales_statistics(
    train: pd.DataFrame
) -> pd.DataFrame:
    """
    Analyze zero-sales observations.
    """

    sales = train["sales"]

    zeros = (sales == 0).sum()

    summary = {

        "Total Observations": len(sales),

        "Zero Sales": zeros,

        "Zero Sales (%)":
            round(
                zeros / len(sales) * 100,
                2
            ),

        "Non-Zero Sales":
            len(sales) - zeros

    }

    return (
        pd.DataFrame(summary, index=["Value"])
        .T
        .rename(columns={"Value": "Sales"})
    )


# =========================================================
# OUTLIER ANALYSIS
# =========================================================

def outlier_statistics(
    train: pd.DataFrame,
    column: str = "sales"
) -> pd.DataFrame:
    """
    Detect outliers using IQR.
    """

    series = train[column]

    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)

    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outliers = (
        (series < lower)
        | (series > upper)
    )

    summary = {

        "Lower Bound": lower,

        "Upper Bound": upper,

        "Outliers": outliers.sum(),

        "Outlier (%)":
            round(
                outliers.mean() * 100,
                2
            )

    }

    return (
        pd.DataFrame(summary, index=["Value"])
        .T
        .rename(columns={"Value": column})
    )

# =========================================================
# CORRELATION MATRIX
# =========================================================

def correlation_matrix(
    train: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute Pearson correlation matrix
    for all numerical variables.
    """

    numeric = train.select_dtypes(
        include="number"
    )

    return numeric.corr()


# =========================================================
# COVARIANCE MATRIX
# =========================================================

def covariance_matrix(
    train: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute covariance matrix
    for numerical variables.
    """

    numeric = train.select_dtypes(
        include="number"
    )

    return numeric.cov()


# =========================================================
# SALES BY STORE STATISTICS
# =========================================================

def sales_by_store_statistics(
    train: pd.DataFrame
) -> pd.DataFrame:
    """
    Sales statistics grouped by store.
    """

    summary = (
        train.groupby("store_nbr")
        .agg(

            Total_Sales=("sales", "sum"),

            Average_Sales=("sales", "mean"),

            Median_Sales=("sales", "median"),

            Std_Sales=("sales", "std"),

            Min_Sales=("sales", "min"),

            Max_Sales=("sales", "max"),

            Transactions=("sales", "count")

        )
        .reset_index()
    )

    return summary


# =========================================================
# SALES BY PRODUCT FAMILY
# =========================================================

def sales_by_family_statistics(
    train: pd.DataFrame
) -> pd.DataFrame:
    """
    Sales statistics grouped by product family.
    """

    summary = (
        train.groupby("family")
        .agg(

            Total_Sales=("sales", "sum"),

            Average_Sales=("sales", "mean"),

            Median_Sales=("sales", "median"),

            Std_Sales=("sales", "std"),

            Transactions=("sales", "count")

        )
        .reset_index()

        .sort_values(
            "Total_Sales",
            ascending=False
        )
    )

    return summary


# =========================================================
# SALES BY DAY OF WEEK
# =========================================================

def weekday_statistics(
    train: pd.DataFrame
) -> pd.DataFrame:
    """
    Sales statistics grouped by weekday.
    """

    df = train.copy()

    df["date"] = pd.to_datetime(
        df["date"]
    )

    df["weekday"] = (
        df["date"]
        .dt.day_name()
    )

    order = [

        "Monday",

        "Tuesday",

        "Wednesday",

        "Thursday",

        "Friday",

        "Saturday",

        "Sunday"

    ]

    summary = (
        df.groupby("weekday")
        .agg(

            Total_Sales=("sales", "sum"),

            Average_Sales=("sales", "mean"),

            Median_Sales=("sales", "median"),

            Transactions=("sales", "count")

        )
        .reindex(order)
        .reset_index()
    )

    return summary


# =========================================================
# PROMOTION STATISTICS
# =========================================================

def promotion_statistics(
    train: pd.DataFrame
) -> pd.DataFrame:
    """
    Statistical comparison between
    promoted and non-promoted products.
    """

    df = train.copy()

    df["promotion"] = (
        df["onpromotion"] > 0
    )

    summary = (
        df.groupby("promotion")
        .agg(

            Total_Sales=("sales", "sum"),

            Average_Sales=("sales", "mean"),

            Median_Sales=("sales", "median"),

            Std_Sales=("sales", "std"),

            Transactions=("sales", "count")

        )
        .reset_index()
    )

    return summary


# =========================================================
# NUMERIC SUMMARY
# =========================================================

def numeric_summary(
    train: pd.DataFrame
) -> pd.DataFrame:
    """
    Summary statistics for all
    numerical columns.
    """

    return (
        train
        .select_dtypes(
            include=np.number
        )
        .describe()
        .T
    )


# =========================================================
# CATEGORICAL SUMMARY
# =========================================================

def categorical_summary(
    train: pd.DataFrame
) -> pd.DataFrame:
    """
    Summary statistics for
    categorical variables.
    """

    categorical = train.select_dtypes(
        include=[
            "object",
            "category"
        ]
    )

    summary = pd.DataFrame({

        "Unique":
            categorical.nunique(),

        "Missing":
            categorical.isna().sum(),

        "Most Frequent":
            categorical.mode().iloc[0]

    })

    return summary
# =========================================================
# ROLLING STATISTICS
# =========================================================

def rolling_statistics(
    train: pd.DataFrame,
    window: int = 7
) -> pd.DataFrame:
    """
    Calculate rolling statistics for daily sales.
    """

    df = train.copy()

    df["date"] = pd.to_datetime(df["date"])

    daily = (
        df.groupby("date", as_index=False)["sales"]
        .sum()
        .rename(columns={"sales": "total_sales"})
    )

    daily["rolling_mean"] = (
        daily["total_sales"]
        .rolling(window=window)
        .mean()
    )

    daily["rolling_std"] = (
        daily["total_sales"]
        .rolling(window=window)
        .std()
    )

    daily["rolling_min"] = (
        daily["total_sales"]
        .rolling(window=window)
        .min()
    )

    daily["rolling_max"] = (
        daily["total_sales"]
        .rolling(window=window)
        .max()
    )

    return daily


# =========================================================
# COEFFICIENT OF VARIATION
# =========================================================

def coefficient_of_variation(
    train: pd.DataFrame,
    column: str = "sales"
) -> pd.DataFrame:
    """
    Calculate coefficient of variation.
    """

    series = train[column]

    cv = (
        series.std()
        / series.mean()
    ) * 100

    summary = {

        "Mean": series.mean(),

        "Standard Deviation": series.std(),

        "Coefficient of Variation (%)": cv

    }

    return (
        pd.DataFrame(summary, index=["Value"])
        .T
        .rename(columns={"Value": column})
    )


# =========================================================
# SALES PERCENTILES
# =========================================================

def sales_percentiles(
    train: pd.DataFrame,
    column: str = "sales"
) -> pd.DataFrame:
    """
    Calculate major sales percentiles.
    """

    percentiles = {

        "P01": train[column].quantile(0.01),

        "P05": train[column].quantile(0.05),

        "P10": train[column].quantile(0.10),

        "P25": train[column].quantile(0.25),

        "P50": train[column].quantile(0.50),

        "P75": train[column].quantile(0.75),

        "P90": train[column].quantile(0.90),

        "P95": train[column].quantile(0.95),

        "P99": train[column].quantile(0.99)

    }

    return (
        pd.DataFrame(percentiles, index=["Value"])
        .T
        .rename(columns={"Value": column})
    )


# =========================================================
# BUSINESS STATISTICS REPORT
# =========================================================

def business_statistics_report(
    train: pd.DataFrame
) -> Dict[str, pd.DataFrame]:
    """
    Complete statistical report.
    """

    report = {

        "Descriptive":
            descriptive_statistics(train),

        "Quantiles":
            quantile_statistics(train),

        "Zero Sales":
            zero_sales_statistics(train),

        "Outliers":
            outlier_statistics(train),

        "Numeric":
            numeric_summary(train),

        "Categorical":
            categorical_summary(train),

        "Correlation":
            correlation_matrix(train),

        "Covariance":
            covariance_matrix(train),

        "Store Statistics":
            sales_by_store_statistics(train),

        "Family Statistics":
            sales_by_family_statistics(train),

        "Weekday Statistics":
            weekday_statistics(train),

        "Promotion Statistics":
            promotion_statistics(train),

        "Rolling Statistics":
            rolling_statistics(train),

        "Coefficient of Variation":
            coefficient_of_variation(train),

        "Percentiles":
            sales_percentiles(train)

    }

    return report


# =========================================================
# MASTER FUNCTION
# =========================================================

def run_statistics(
    train: pd.DataFrame
) -> None:
    """
    Execute complete statistical analysis.
    """

    print("\n" + "=" * 70)
    print("DESCRIPTIVE STATISTICS")
    print("=" * 70)
    print(descriptive_statistics(train))

    print("\n" + "=" * 70)
    print("QUANTILE STATISTICS")
    print("=" * 70)
    print(quantile_statistics(train))

    print("\n" + "=" * 70)
    print("ZERO SALES")
    print("=" * 70)
    print(zero_sales_statistics(train))

    print("\n" + "=" * 70)
    print("OUTLIER ANALYSIS")
    print("=" * 70)
    print(outlier_statistics(train))

    print("\n" + "=" * 70)
    print("COEFFICIENT OF VARIATION")
    print("=" * 70)
    print(coefficient_of_variation(train))

    print("\n" + "=" * 70)
    print("WEEKDAY STATISTICS")
    print("=" * 70)
    print(weekday_statistics(train))

    print("\n" + "=" * 70)
    print("PROMOTION STATISTICS")
    print("=" * 70)
    print(promotion_statistics(train))

    print("\n" + "=" * 70)
    print("STATISTICAL ANALYSIS COMPLETED")
    print("=" * 70)