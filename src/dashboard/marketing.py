"""
=========================================================
Marketing Dashboard
=========================================================

Marketing Campaign & Promotion Dashboard

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
# MARKETING DASHBOARD
# ==========================================================

def show_marketing_dashboard():

    df = load_data()

    st.title(
        "📣 Marketing Dashboard"
    )

    st.caption(
        "Promotion Performance & Product Marketing Analysis"
    )

    st.divider()

    # =====================================================
    # FILTERS
    # =====================================================

    c1, c2 = st.columns(2)

    years = sorted(
        df["year"].unique()
    )

    selected_year = c1.selectbox(
        "Year",
        years,
        index=len(years) - 1
    )

    months = sorted(
        df["month"].unique()
    )

    selected_month = c2.selectbox(
        "Month",
        months
    )

    marketing = df.copy()

    marketing = marketing[
        marketing["year"] == selected_year
    ]

    marketing = marketing[
        marketing["month"] == selected_month
    ]

    st.divider()

    # =====================================================
    # KPI SECTION
    # =====================================================

    total_sales = marketing["sales"].sum()

    avg_sales = marketing["sales"].mean()

    promoted_items = marketing[
        "promotion_flag"
    ].sum()

    promoted_pct = (

        promoted_items /

        len(marketing)

    ) * 100

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Marketing Revenue",
        f"${total_sales:,.0f}"
    )

    c2.metric(
        "Average Sale",
        f"${avg_sales:,.2f}"
    )

    c3.metric(
        "Promotion Records",
        f"{promoted_items:,}"
    )

    c4.metric(
        "Promotion Rate",
        f"{promoted_pct:.1f}%"
    )

    st.divider()

    # =====================================================
    # PROMOTION IMPACT
    # =====================================================

    st.subheader(
        "Promotion Performance"
    )

    promotion = (

        marketing.groupby(
            "promotion_flag"
        )

        .agg(

            Revenue=("sales", "sum"),

            Average_Sales=("sales", "mean")

        )

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

        y="Revenue",

        color="promotion_flag",

        title="Revenue by Promotion"

    )

    st.plotly_chart(

        fig,

        width="stretch"

    )

    st.divider()

    # =====================================================
    # PRODUCT FAMILY
    # =====================================================

    st.subheader(
        "Top Product Families"
    )

    families = (

        marketing.groupby("family")

        ["sales"]

        .sum()

        .sort_values(
            ascending=False
        )

        .head(20)

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

        width="stretch"

    )

    st.divider()

    # =====================================================
    # WEEKDAY SALES
    # =====================================================

    st.subheader(
        "Sales by Weekday"
    )

    weekday = (

        marketing.groupby(
            "day_of_week"
        )

        ["sales"]

        .sum()

        .reset_index()

    )

    fig = px.line(

        weekday,

        x="day_of_week",

        y="sales",

        markers=True,

        title="Weekly Marketing Performance"

    )

    st.plotly_chart(

        fig,

        width="stretch"

    )

    st.divider()

    # =====================================================
    # HOLIDAY PERFORMANCE
    # =====================================================

    if "is_holiday" in marketing.columns:

        st.subheader(
            "Holiday Marketing"
        )

        holiday = (

            marketing.groupby(
                "is_holiday"
            )

            ["sales"]

            .mean()

            .reset_index()

        )

        holiday["is_holiday"] = holiday[
            "is_holiday"
        ].map({

            0: "Regular Day",

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

            width="stretch"

        )

        st.divider()

    # =====================================================
    # PRODUCT TABLE
    # =====================================================

    st.subheader(
        "Marketing Performance Summary"
    )

    summary = (

        marketing.groupby(
            "family"
        )

        .agg(

            Revenue=("sales", "sum"),

            Average_Sales=("sales", "mean"),

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

        width="stretch"

    )

    st.divider()

    st.success(
        """
Marketing Insights

• Promotion effectiveness can be monitored using revenue.

• Product families with the highest sales deserve marketing priority.

• Holiday periods can be leveraged for campaigns.

• Weekly sales trends help schedule promotional activities.
        """
    )

    st.caption(
        "Marketing Dashboard | Store Sales Forecasting"
    )