"""
=========================================================
Forecast Dashboard
=========================================================

Forecast Dashboard

Author : Shariq Zia
Project: Store Sales Forecasting
"""

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import (
    PREDICTION_DIR,
    REPORT_DIR
)


# ==========================================================
# LOAD FORECAST
# ==========================================================

@st.cache_data
def load_forecast():

    prediction_file = (
        PREDICTION_DIR /
        "sales_prediction.csv"
    )

    if prediction_file.exists():

        return pd.read_csv(
            prediction_file
        )

    return pd.DataFrame()


# ==========================================================
# FORECAST DASHBOARD
# ==========================================================

def show_forecast_dashboard():

    forecast = load_forecast()

    st.title(
        " Forecast Dashboard"
    )

    st.caption(
        "Machine Learning Demand Forecast & Business Planning"
    )

    st.divider()

    if forecast.empty:

        st.warning(
            "Forecast file not found."
        )

        return

    # =====================================================
    # FILTERS
    # =====================================================

    c1, c2 = st.columns(2)

    stores = sorted(
        forecast["store_nbr"].unique()
    )

    selected_store = c1.selectbox(

        "Store",

        ["All"] + stores

    )

    families = sorted(
        forecast["family"].astype(str).unique()
    )

    selected_family = c2.selectbox(

        "Product Family",

        ["All"] + families

    )

    df = forecast.copy()

    if selected_store != "All":

        df = df[
            df["store_nbr"] == selected_store
        ]

    if selected_family != "All":

        df = df[
            df["family"].astype(str)
            == selected_family
        ]

    st.divider()

    # =====================================================
    # KPI SECTION
    # =====================================================

    total_forecast = df["Forecast"].sum()

    avg_forecast = df["Forecast"].mean()

    highest_forecast = df["Forecast"].max()

    records = len(df)

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(

        "Forecast Demand",

        f"{total_forecast:,.0f}"

    )

    c2.metric(

        "Average Forecast",

        f"{avg_forecast:,.2f}"

    )

    c3.metric(

        "Highest Forecast",

        f"{highest_forecast:,.0f}"

    )

    c4.metric(

        "Forecast Records",

        f"{records:,}"

    )

    st.divider()

    # =====================================================
    # DAILY FORECAST
    # =====================================================

    if "date" in df.columns:

        st.subheader(
            "Forecast Trend"
        )

        daily = (

            df.groupby("date")

            ["Forecast"]

            .sum()

            .reset_index()

        )

        fig = px.line(

            daily,

            x="date",

            y="Forecast",

            markers=True,

            title="Forecast Trend"

        )

        st.plotly_chart(

            fig,

            use_container_width=True

        )

        st.divider()

    # =====================================================
    # STORE FORECAST
    # =====================================================

    st.subheader(
        "Forecast by Store"
    )

    stores = (

        df.groupby("store_nbr")

        ["Forecast"]

        .sum()

        .sort_values(
            ascending=False
        )

        .reset_index()

    )

    fig = px.bar(

        stores,

        x="store_nbr",

        y="Forecast",

        title="Store Forecast"

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
        "Forecast by Product Family"
    )

    families = (

        df.groupby("family")

        ["Forecast"]

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

        y="Forecast",

        title="Product Family Forecast"

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

    st.divider()

    # =====================================================
    # FORECAST DISTRIBUTION
    # =====================================================

    st.subheader(
        "Forecast Distribution"
    )

    fig = px.histogram(

        df,

        x="Forecast",

        nbins=40,

        title="Forecast Distribution"

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

    st.divider()

    # =====================================================
    # TOP FORECAST ITEMS
    # =====================================================

    st.subheader(
        "Highest Forecast Items"
    )

    columns = [

        col for col in [

            "date",
            "store_nbr",
            "family",
            "Forecast"

        ]

        if col in df.columns

    ]

    top = (

        df[columns]

        .sort_values(

            "Forecast",

            ascending=False

        )

        .head(50)

    )

    st.dataframe(

        top,

        use_container_width=True,

        hide_index=True

    )

    st.divider()

    # =====================================================
    # DOWNLOAD REPORT
    # =====================================================

    prediction_file = (

        PREDICTION_DIR /

        "sales_prediction.csv"

    )

    if prediction_file.exists():

        with open(

            prediction_file,

            "rb"

        ) as f:

            st.download_button(

                "📥 Download Forecast CSV",

                data=f,

                file_name="sales_prediction.csv",

                mime="text/csv"

            )

    report = (

        REPORT_DIR /

        "Demand_Forecast_Report.xlsx"

    )

    if report.exists():

        with open(

            report,

            "rb"

        ) as f:

            st.download_button(

                "📥 Download Excel Forecast Report",

                data=f,

                file_name=report.name,

                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

            )

    st.divider()

    st.success(
        """
Forecast Insights

• Forecast demand supports inventory replenishment.

• High-demand products should receive procurement priority.

• Store-level forecasts improve distribution planning.

• Product family forecasts support purchasing decisions.

• Downloadable reports are available for business users.
        """
    )

    st.caption(
        "Forecast Dashboard | Store Sales Forecasting"
    )