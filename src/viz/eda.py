import matplotlib.pyplot as plt

from .style import *
from .utils import apply_style


def plot_sales_trend(df):

    fig, ax = plt.subplots(figsize=FIGSIZE)

    ax.plot(
        df["date"],
        df["sales"],
        color=PRIMARY,
        linewidth=LINE_WIDTH
    )

    apply_style(
        ax,
        title="Daily Sales Trend",
        xlabel="Date",
        ylabel="Sales"
    )

    return fig

def plot_monthly_sales(df):

    fig, ax = plt.subplots(figsize=FIGSIZE)

    ax.bar(
        df["month"],
        df["sales"],
        color=PRIMARY
    )

    apply_style(
        ax,
        title="Average Monthly Sales",
        xlabel="Month",
        ylabel="Sales"
    )

    return fig

def plot_weekday_sales(df):

    fig, ax = plt.subplots(figsize=FIGSIZE)

    ax.bar(
        df["day_of_week"],
        df["sales"],
        color=PRIMARY
    )

    apply_style(
        ax,
        title="Average Sales by Weekday",
        xlabel="Weekday",
        ylabel="Sales"
    )

    return fig

def plot_family_sales(df):

    fig, ax = plt.subplots(figsize=(12,7))

    ax.barh(
        df["family"],
        df["sales"],
        color=PRIMARY
    )

    apply_style(
        ax,
        title="Top Product Families",
        xlabel="Sales",
        ylabel=""
    )

    return fig

def plot_store_sales(df):

    fig, ax = plt.subplots(figsize=(14,6))

    ax.bar(
        df["store_nbr"],
        df["sales"],
        color=PRIMARY
    )

    apply_style(
        ax,
        title="Sales by Store",
        xlabel="Store",
        ylabel="Sales"
    )

    return fig

def plot_promotion_effect(df):

    fig, ax = plt.subplots(figsize=FIGSIZE)

    ax.bar(
        df["onpromotion"],
        df["sales"],
        color=PRIMARY
    )

    apply_style(
        ax,
        title="Promotion Impact",
        xlabel="Promotion Count",
        ylabel="Average Sales"
    )

    return fig

def plot_oil_price(df):

    fig, ax = plt.subplots(figsize=FIGSIZE)

    ax.plot(
        df["date"],
        df["dcoilwtico"],
        color=SECONDARY,
        linewidth=LINE_WIDTH
    )

    apply_style(
        ax,
        title="Oil Price Trend",
        xlabel="Date",
        ylabel="Oil Price"
    )

    return fig

def plot_holiday_frequency(df):

    fig, ax = plt.subplots(figsize=FIGSIZE)

    ax.bar(
        df["type"],
        df["count"],
        color=SECONDARY
    )

    apply_style(
        ax,
        title="Holiday Distribution",
        xlabel="Holiday Type",
        ylabel="Count"
    )

    return fig

def plot_correlation(df):

    fig, ax = plt.subplots(figsize=(10,8))

    image = ax.imshow(
        df,
        cmap="Blues"
    )

    plt.colorbar(image)

    ax.set_xticks(range(len(df.columns)))
    ax.set_xticklabels(
        df.columns,
        rotation=90
    )

    ax.set_yticks(range(len(df.columns)))
    ax.set_yticklabels(df.columns)

    apply_style(
        ax,
        title="Correlation Matrix"
    )

    return fig

