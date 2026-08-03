"""
=========================================================
Exploratory Data Analysis (EDA)
=========================================================

Reusable exploratory data analysis functions for
retail demand forecasting.

Author : Shariq Zia
Project: Store Sales Forecasting
"""

from typing import Dict, Optional

import matplotlib.pyplot as plt
import pandas as pd


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def _prepare_sales_data(
    train: pd.DataFrame
) -> pd.DataFrame:
    """
    Prepare sales dataset for exploratory analysis.

    Parameters
    ----------
    train : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """

    df = train.copy()

    df["date"] = pd.to_datetime(df["date"])

    return df


def _merge_store_data(
    train: pd.DataFrame,
    stores: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge sales with store metadata.

    Parameters
    ----------
    train : pd.DataFrame

    stores : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """

    df = _prepare_sales_data(train)

    df = df.merge(
        stores,
        how="left",
        on="store_nbr"
    )

    return df


# =========================================================
# DAILY SALES TREND
# =========================================================

def daily_sales_trend(
    train: pd.DataFrame
) -> pd.DataFrame:
    """
    Aggregate daily sales.
    """

    df = _prepare_sales_data(train)

    daily = (
        df.groupby(
            "date",
            as_index=False
        )["sales"]
        .sum()
        .rename(
            columns={
                "sales": "total_sales"
            }
        )
    )

    return daily


# =========================================================
# WEEKLY SALES TREND
# =========================================================

def weekly_sales_trend(
    train: pd.DataFrame
) -> pd.DataFrame:
    """
    Aggregate weekly sales.
    """

    df = _prepare_sales_data(train)

    weekly = (
        df.set_index("date")
        .resample("W")["sales"]
        .sum()
        .reset_index()
        .rename(
            columns={
                "sales": "total_sales"
            }
        )
    )

    return weekly


# =========================================================
# MONTHLY SALES TREND
# =========================================================

def monthly_sales_trend(
    train: pd.DataFrame
) -> pd.DataFrame:
    """
    Aggregate monthly sales.
    """

    df = _prepare_sales_data(train)

    monthly = (
        df.set_index("date")
        .resample("M")["sales"]
        .sum()
        .reset_index()
        .rename(
            columns={
                "sales": "total_sales"
            }
        )
    )

    return monthly


# =========================================================
# YEARLY SALES TREND
# =========================================================

def yearly_sales_trend(
    train: pd.DataFrame
) -> pd.DataFrame:
    """
    Aggregate yearly sales.
    """

    df = _prepare_sales_data(train)

    yearly = (
        df.set_index("date")
        .resample("Y")["sales"]
        .sum()
        .reset_index()
        .rename(
            columns={
                "sales": "total_sales"
            }
        )
    )

    return yearly


# =========================================================
# SALES GROWTH
# =========================================================

def sales_growth(
    train: pd.DataFrame,
    frequency: str = "M"
) -> pd.DataFrame:
    """
    Calculate sales growth.

    Parameters
    ----------
    train : pd.DataFrame

    frequency : {"D","W","M","Y"}

    Returns
    -------
    pd.DataFrame
    """

    df = _prepare_sales_data(train)

    frequency_map = {
        "D": "D",
        "W": "W",
        "M": "M",
        "Y": "Y"
    }

    sales = (
        df.set_index("date")
        .resample(
            frequency_map[frequency]
        )["sales"]
        .sum()
        .reset_index()
    )

    sales["growth_pct"] = (
        sales["sales"]
        .pct_change()
        * 100
    )

    return sales


# =========================================================
# SALES SUMMARY
# =========================================================

def sales_summary(
    train: pd.DataFrame
) -> pd.DataFrame:
    """
    Executive sales summary.
    """

    df = _prepare_sales_data(train)

    summary = {

        "Total Sales":
            df["sales"].sum(),

        "Average Daily Sales":
            daily_sales_trend(train)[
                "total_sales"
            ].mean(),

        "Maximum Daily Sales":
            daily_sales_trend(train)[
                "total_sales"
            ].max(),

        "Minimum Daily Sales":
            daily_sales_trend(train)[
                "total_sales"
            ].min(),

        "Transactions":
            len(df),

        "Stores":
            df["store_nbr"].nunique(),

        "Families":
            df["family"].nunique()

    }

    return (
        pd.DataFrame(summary, index=["Value"])
        .T
        .rename(columns={"Value": "Metric"})
    )

# =========================================================
# PRODUCT FAMILY ANALYSIS
# =========================================================

def product_family_analysis(
    train: pd.DataFrame
) -> pd.DataFrame:
    """
    Aggregate sales by product family.
    """

    df = train.copy()

    summary = (
        df.groupby(
            "family",
            as_index=False
        )["sales"]
        .sum()
        .sort_values(
            "sales",
            ascending=False
        )
        .rename(
            columns={
                "sales": "total_sales"
            }
        )
    )

    return summary


# =========================================================
# TOP PRODUCT FAMILIES
# =========================================================

def top_product_families(
    train: pd.DataFrame,
    top_n: int = 10
) -> pd.DataFrame:
    """
    Top selling product families.
    """

    summary = product_family_analysis(train)

    return summary.head(top_n)


# =========================================================
# STORE PERFORMANCE
# =========================================================

def store_performance(
    train: pd.DataFrame
) -> pd.DataFrame:
    """
    Aggregate sales by store.
    """

    summary = (
        train.groupby(
            "store_nbr",
            as_index=False
        )["sales"]
        .sum()
        .sort_values(
            "sales",
            ascending=False
        )
        .rename(
            columns={
                "sales": "total_sales"
            }
        )
    )

    return summary


# =========================================================
# TOP STORES
# =========================================================

def top_stores(
    train: pd.DataFrame,
    top_n: int = 10
) -> pd.DataFrame:
    """
    Highest performing stores.
    """

    summary = store_performance(train)

    return summary.head(top_n)


# =========================================================
# STORE TYPE ANALYSIS
# =========================================================

def store_type_analysis(
    train: pd.DataFrame,
    stores: pd.DataFrame
) -> pd.DataFrame:
    """
    Sales by store type.
    """

    df = _merge_store_data(
        train,
        stores
    )

    summary = (
        df.groupby(
            "type",
            as_index=False
        )["sales"]
        .sum()
        .sort_values(
            "sales",
            ascending=False
        )
        .rename(
            columns={
                "sales": "total_sales"
            }
        )
    )

    return summary


# =========================================================
# STORE CLUSTER ANALYSIS
# =========================================================

def cluster_analysis(
    train: pd.DataFrame,
    stores: pd.DataFrame
) -> pd.DataFrame:
    """
    Sales by store cluster.
    """

    df = _merge_store_data(
        train,
        stores
    )

    summary = (
        df.groupby(
            "cluster",
            as_index=False
        )["sales"]
        .sum()
        .sort_values(
            "sales",
            ascending=False
        )
        .rename(
            columns={
                "sales": "total_sales"
            }
        )
    )

    return summary


# =========================================================
# CITY ANALYSIS
# =========================================================

def city_analysis(
    train: pd.DataFrame,
    stores: pd.DataFrame
) -> pd.DataFrame:
    """
    Sales by city.
    """

    df = _merge_store_data(
        train,
        stores
    )

    summary = (
        df.groupby(
            "city",
            as_index=False
        )["sales"]
        .sum()
        .sort_values(
            "sales",
            ascending=False
        )
        .rename(
            columns={
                "sales": "total_sales"
            }
        )
    )

    return summary


# =========================================================
# STATE ANALYSIS
# =========================================================

def state_analysis(
    train: pd.DataFrame,
    stores: pd.DataFrame
) -> pd.DataFrame:
    """
    Sales by state.
    """

    df = _merge_store_data(
        train,
        stores
    )

    summary = (
        df.groupby(
            "state",
            as_index=False
        )["sales"]
        .sum()
        .sort_values(
            "sales",
            ascending=False
        )
        .rename(
            columns={
                "sales": "total_sales"
            }
        )
    )

    return summary


# =========================================================
# STORE COUNT BY CITY
# =========================================================

def stores_per_city(
    stores: pd.DataFrame
) -> pd.DataFrame:
    """
    Number of stores by city.
    """

    summary = (
        stores.groupby(
            "city",
            as_index=False
        )
        .size()
        .rename(
            columns={
                "size": "store_count"
            }
        )
        .sort_values(
            "store_count",
            ascending=False
        )
    )

    return summary


# =========================================================
# STORE COUNT BY STATE
# =========================================================

def stores_per_state(
    stores: pd.DataFrame
) -> pd.DataFrame:
    """
    Number of stores by state.
    """

    summary = (
        stores.groupby(
            "state",
            as_index=False
        )
        .size()
        .rename(
            columns={
                "size": "store_count"
            }
        )
        .sort_values(
            "store_count",
            ascending=False
        )
    )

    return summary

# =========================================================
# PROMOTION ANALYSIS
# =========================================================

def promotion_analysis(
    train: pd.DataFrame
) -> pd.DataFrame:
    """
    Analyze sales by promotion status.

    Parameters
    ----------
    train : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """

    df = train.copy()

    df["promotion_flag"] = df["onpromotion"] > 0

    summary = (
        df.groupby(
            "promotion_flag",
            as_index=False
        )
        .agg(

            total_sales=("sales", "sum"),

            average_sales=("sales", "mean"),

            total_transactions=("sales", "count"),

            promoted_items=("onpromotion", "sum")

        )
    )

    return summary


# =========================================================
# HOLIDAY ANALYSIS
# =========================================================

def holiday_analysis(
    train: pd.DataFrame,
    holidays: pd.DataFrame
) -> pd.DataFrame:
    """
    Analyze holiday impact on sales.
    """

    df = _prepare_sales_data(train)

    holiday = holidays.copy()

    holiday["date"] = pd.to_datetime(
        holiday["date"]
    )

    holiday = holiday[
        ["date", "type", "locale", "description"]
    ].drop_duplicates()

    merged = df.merge(

        holiday,

        how="left",

        on="date"

    )

    merged["holiday"] = (
        merged["description"]
        .notna()
    )

    summary = (
        merged.groupby(
            "holiday",
            as_index=False
        )
        .agg(

            total_sales=("sales", "sum"),

            average_sales=("sales", "mean"),

            observations=("sales", "count")

        )
    )

    return summary


# =========================================================
# OIL PRICE ANALYSIS
# =========================================================

def oil_price_analysis(
    train: pd.DataFrame,
    oil: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge daily sales with oil prices.
    """

    sales = daily_sales_trend(train)

    oil_df = oil.copy()

    oil_df["date"] = pd.to_datetime(
        oil_df["date"]
    )

    oil_df = oil_df.rename(
        columns={
            "dcoilwtico": "oil_price"
        }
    )

    merged = sales.merge(

        oil_df,

        how="left",

        on="date"

    )

    merged["oil_price"] = (
        merged["oil_price"]
        .ffill()
        .bfill()
    )

    return merged


# =========================================================
# SALES DISTRIBUTION
# =========================================================

def sales_distribution(
    train: pd.DataFrame
) -> pd.DataFrame:
    """
    Summary statistics of sales distribution.
    """

    sales = train["sales"]

    summary = {

        "Minimum":
            sales.min(),

        "Maximum":
            sales.max(),

        "Mean":
            sales.mean(),

        "Median":
            sales.median(),

        "Standard Deviation":
            sales.std(),

        "Variance":
            sales.var(),

        "Skewness":
            sales.skew(),

        "Kurtosis":
            sales.kurt(),

        "Zero Sales":
            (sales == 0).sum(),

        "Zero Sales (%)":
            round(
                (sales == 0).mean() * 100,
                2
            )

    }

    return (
        pd.DataFrame(
            summary,
            index=["Value"]
        )
        .T
        .rename(
            columns={
                "Value": "Metric"
            }
        )
    )


# =========================================================
# CORRELATION ANALYSIS
# =========================================================

def correlation_analysis(
    train: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate correlation matrix for numerical variables.
    """

    numeric = train.select_dtypes(
        include="number"
    )

    return numeric.corr()


# =========================================================
# WEEKDAY ANALYSIS
# =========================================================

def weekday_analysis(
    train: pd.DataFrame
) -> pd.DataFrame:
    """
    Sales by weekday.
    """

    df = _prepare_sales_data(train)

    df["weekday"] = df["date"].dt.day_name()

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
        df.groupby(
            "weekday",
            as_index=False
        )["sales"]
        .sum()
        .rename(
            columns={
                "sales": "total_sales"
            }
        )
    )

    summary["weekday"] = pd.Categorical(

        summary["weekday"],

        categories=order,

        ordered=True

    )

    summary = summary.sort_values(
        "weekday"
    )

    return summary


# =========================================================
# WEEKEND ANALYSIS
# =========================================================

def weekend_analysis(
    train: pd.DataFrame
) -> pd.DataFrame:
    """
    Compare weekday and weekend sales.
    """

    df = _prepare_sales_data(train)

    df["is_weekend"] = (
        df["date"]
        .dt
        .dayofweek
        >= 5
    )

    summary = (
        df.groupby(
            "is_weekend",
            as_index=False
        )
        .agg(

            total_sales=("sales", "sum"),

            average_sales=("sales", "mean")

        )
    )

    return summary

# =========================================================
# GENERIC LINE PLOT
# =========================================================

def plot_sales_trend(
    data: pd.DataFrame,
    x: str = "date",
    y: str = "total_sales",
    title: str = "Sales Trend",
    figsize: tuple = (14, 6)
) -> None:
    """
    Generic line chart.
    """

    plt.figure(figsize=figsize)

    plt.plot(
        data[x],
        data[y],
        linewidth=2
    )

    plt.title(title)

    plt.xlabel(x.replace("_", " ").title())

    plt.ylabel(y.replace("_", " ").title())

    plt.grid(alpha=0.30)

    plt.tight_layout()

    plt.show()


# =========================================================
# BAR CHART
# =========================================================

def plot_bar(
    data: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    figsize: tuple = (12, 6)
) -> None:
    """
    Generic bar chart.
    """

    plt.figure(figsize=figsize)

    plt.bar(
        data[x].astype(str),
        data[y]
    )

    plt.title(title)

    plt.xticks(rotation=45)

    plt.grid(axis="y", alpha=0.30)

    plt.tight_layout()

    plt.show()


# =========================================================
# HISTOGRAM
# =========================================================

def plot_histogram(
    train: pd.DataFrame,
    column: str = "sales",
    bins: int = 50,
    figsize: tuple = (10, 6)
) -> None:
    """
    Histogram plot.
    """

    plt.figure(figsize=figsize)

    plt.hist(
        train[column],
        bins=bins
    )

    plt.title(f"{column.title()} Distribution")

    plt.xlabel(column.title())

    plt.ylabel("Frequency")

    plt.tight_layout()

    plt.show()


# =========================================================
# BOXPLOT
# =========================================================

def plot_boxplot(
    train: pd.DataFrame,
    column: str = "sales",
    figsize: tuple = (8, 5)
) -> None:
    """
    Generic boxplot.
    """

    plt.figure(figsize=figsize)

    plt.boxplot(
        train[column],
        vert=True
    )

    plt.title(f"{column.title()} Boxplot")

    plt.tight_layout()

    plt.show()


# =========================================================
# SCATTER PLOT
# =========================================================

def plot_scatter(
    data: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    figsize: tuple = (8, 6)
) -> None:
    """
    Generic scatter plot.
    """

    plt.figure(figsize=figsize)

    plt.scatter(
        data[x],
        data[y],
        alpha=0.60
    )

    plt.title(title)

    plt.xlabel(x)

    plt.ylabel(y)

    plt.tight_layout()

    plt.show()


# =========================================================
# CORRELATION HEATMAP
# =========================================================

def plot_heatmap(
    corr: pd.DataFrame,
    figsize: tuple = (10, 8)
) -> None:
    """
    Correlation heatmap.
    """

    plt.figure(figsize=figsize)

    plt.imshow(
        corr,
        aspect="auto"
    )

    plt.colorbar()

    plt.xticks(
        range(len(corr.columns)),
        corr.columns,
        rotation=90
    )

    plt.yticks(
        range(len(corr.columns)),
        corr.columns
    )

    plt.title("Correlation Matrix")

    plt.tight_layout()

    plt.show()


# =========================================================
# MASTER EDA FUNCTION
# =========================================================

def run_eda(
    train: pd.DataFrame,
    stores: Optional[pd.DataFrame] = None,
    holidays: Optional[pd.DataFrame] = None,
    oil: Optional[pd.DataFrame] = None
) -> None:
    """
    Execute complete exploratory data analysis.

    Parameters
    ----------
    train : pd.DataFrame

    stores : pd.DataFrame, optional

    holidays : pd.DataFrame, optional

    oil : pd.DataFrame, optional
    """

    print("\n" + "=" * 70)
    print("SALES SUMMARY")
    print("=" * 70)
    print(sales_summary(train))

    print("\n" + "=" * 70)
    print("PRODUCT PERFORMANCE")
    print("=" * 70)
    print(top_product_families(train))

    print("\n" + "=" * 70)
    print("STORE PERFORMANCE")
    print("=" * 70)
    print(top_stores(train))

    print("\n" + "=" * 70)
    print("SALES DISTRIBUTION")
    print("=" * 70)
    print(sales_distribution(train))

    print("\n" + "=" * 70)
    print("PROMOTION ANALYSIS")
    print("=" * 70)
    print(promotion_analysis(train))

    print("\n" + "=" * 70)
    print("WEEKDAY ANALYSIS")
    print("=" * 70)
    print(weekday_analysis(train))

    print("\n" + "=" * 70)
    print("WEEKEND ANALYSIS")
    print("=" * 70)
    print(weekend_analysis(train))

    if stores is not None:

        print("\n" + "=" * 70)
        print("STORE TYPE ANALYSIS")
        print("=" * 70)
        print(store_type_analysis(train, stores))

        print("\n" + "=" * 70)
        print("CITY ANALYSIS")
        print("=" * 70)
        print(city_analysis(train, stores))

        print("\n" + "=" * 70)
        print("STATE ANALYSIS")
        print("=" * 70)
        print(state_analysis(train, stores))

    if holidays is not None:

        print("\n" + "=" * 70)
        print("HOLIDAY ANALYSIS")
        print("=" * 70)
        print(holiday_analysis(train, holidays))

    if oil is not None:

        print("\n" + "=" * 70)
        print("OIL PRICE ANALYSIS")
        print("=" * 70)
        print(oil_price_analysis(train, oil).head())

    print("\n" + "=" * 70)
    print("EDA COMPLETED SUCCESSFULLY")
    print("=" * 70)

