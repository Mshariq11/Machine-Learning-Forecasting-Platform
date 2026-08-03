"""
=========================================================
Executive Dashboard
=========================================================

Executive Dashboard

Author : Shariq Zia
Project: Store Sales Forecasting
"""

import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import PROCESSED_DATA_DIR


# ==========================================================
# LOAD DATA
# ==========================================================

@st.cache_data
def load_data():

    train = pd.read_parquet(
        PROCESSED_DATA_DIR /
        "train_features.parquet"
    )

    return train


# ==========================================================
# EXECUTIVE DASHBOARD
# ==========================================================

def show_executive_dashboard():

    df = load_data()

    st.title(
        " Executive Dashboard"
    )

    st.caption(
        "Strategic Business Performance Overview"
    )

    st.divider()

    # =====================================================
    # FILTERS
    # =====================================================

    c1, c2, c3 = st.columns(3)

    years = sorted(
        df["year"].unique()
    )

    selected_year = c1.selectbox(
        "Year",
        years,
        index=len(years)-1
    )

    months = sorted(
        df["month"].unique()
    )

    selected_month = c2.selectbox(
        "Month",
        months
    )

    stores = sorted(
        df["store_nbr"].unique()
    )

    selected_store = c3.selectbox(
        "Store",
        ["All"] + stores
    )

    executive = df.copy()

    executive = executive[
        executive["year"] == selected_year
    ]

    executive = executive[
        executive["month"] == selected_month
    ]

    if selected_store != "All":

        executive = executive[
            executive["store_nbr"] == selected_store
        ]

    st.divider()

    # =====================================================
    # KPI SECTION
    # =====================================================

    revenue = executive["sales"].sum()

    avg_sales = executive["sales"].mean()

    stores = executive["store_nbr"].nunique()

    families = executive["family"].nunique()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Revenue",
        f"${revenue:,.0f}"
    )

    c2.metric(
        "Average Sale",
        f"${avg_sales:,.2f}"
    )

    c3.metric(
        "Stores",
        stores
    )

    c4.metric(
        "Product Families",
        families
    )

    st.divider()

    # =====================================================
    # DAILY SALES
    # =====================================================

    st.subheader(
        "Daily Sales Trend"
    )

    daily = (

        executive.groupby("date")

        ["sales"]

        .sum()

        .reset_index()

    )

    fig = px.line(

        daily,

        x="date",

        y="sales",

        markers=True,

        title="Daily Revenue"

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

    st.divider()

    # =====================================================
    # PRODUCT FAMILY
    # =====================================================

    st.subheader(
        "Top Product Families"
    )

    families = (

        executive.groupby("family")

        ["sales"]

        .sum()

        .sort_values(
            ascending=False
        )

        .head(15)

        .reset_index()

    )

    fig = px.bar(

        families,

        x="family",

        y="sales",

        title="Revenue by Product Family"

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

    st.divider()

    # =====================================================
    # STORE PERFORMANCE
    # =====================================================

    st.subheader(
        "Top Stores"
    )

    store_sales = (

        executive.groupby("store_nbr")

        ["sales"]

        .sum()

        .sort_values(
            ascending=False
        )

        .head(15)

        .reset_index()

    )

    fig = px.bar(

        store_sales,

        x="store_nbr",

        y="sales",

        title="Top Revenue Stores"

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

    st.divider()

    # =====================================================
    # PROMOTIONS
    # =====================================================

    st.subheader(
        "Promotion Impact"
    )

    promotion = (

        executive.groupby("promotion_flag")

        ["sales"]

        .mean()

        .reset_index()

    )

    promotion["promotion_flag"] = promotion[
        "promotion_flag"
    ].map({

        0: "No Promotion",

        1: "Promotion"

    })

    fig = px.bar(

        promotion,

        x="promotion_flag",

        y="sales",

        color="promotion_flag",

        title="Average Sales by Promotion"

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

    st.divider()

    # =====================================================
    # HOLIDAY IMPACT
    # =====================================================

    if "is_holiday" in executive.columns:

        st.subheader(
            "Holiday Performance"
        )

        holiday = (

            executive.groupby("is_holiday")

            ["sales"]

            .mean()

            .reset_index()

        )

        holiday["is_holiday"] = holiday[
            "is_holiday"
        ].map({

            0: "Normal Day",

            1: "Holiday"

        })

        fig = px.bar(

            holiday,

            x="is_holiday",

            y="sales",

            color="is_holiday",

            title="Average Sales"

        )

        st.plotly_chart(

            fig,

            use_container_width=True

        )

        st.divider()

    # =====================================================
    # EXECUTIVE SUMMARY
    # =====================================================

    st.subheader(
        "Executive Summary"
    )

    summary = (

        executive.groupby("store_nbr")

        .agg(

            Revenue=("sales", "sum"),

            Average_Sale=("sales", "mean"),

            Transactions=("sales", "count")

        )

        .sort_values(

            "Revenue",

            ascending=False

        )

        .reset_index()

    )

    st.dataframe(

        summary,

        hide_index=True,

        use_container_width=True

    )

    st.divider()

    st.info(
        """
Executive Insight

• Revenue performance can be monitored
  daily and monthly.

• Product families contribute differently
  to overall business revenue.

• Promotion effectiveness can be evaluated
  directly.

• Store comparison helps prioritize
  investment and inventory allocation.
        """
    )

    st.caption(
        "Executive Dashboard | Store Sales Forecasting"
    )