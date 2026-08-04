"""
=========================================================
Planning Dashboard
=========================================================

Planning Dashboard

Author : Shariq Zia
Project: Store Sales Forecasting
"""

import urllib.request

from src.models.predict import run_prediction

import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import (
    PROCESSED_DATA_DIR,
    MODEL_DIR,
    PREDICTION_DIR,
    REPORT_DIR,
)


# ==========================================================
# LOAD DATA
# ==========================================================

@st.cache_data
def load_forecast():

    prediction_file = (
        PREDICTION_DIR /
        "sales_prediction.csv"
    )

    try:

        if not prediction_file.exists():

            test_file = (
                PROCESSED_DATA_DIR /
                "test_features.parquet"
            )

            if not test_file.exists():

                urllib.request.urlretrieve(
                    "https://huggingface.co/datasets/ShawRickZia/machine-learning-forecasting-data/resolve/main/test_features.parquet",
                    test_file
                )

            test = pd.read_parquet(
                test_file
            )

            run_prediction(
                test=test,
                model_path=MODEL_DIR / "xgboost_model.pkl",
                feature_path=MODEL_DIR / "feature_columns.pkl",
                output_path=prediction_file
            )

        return pd.read_csv(
            prediction_file
        )

    except Exception as e:

        st.error(
            f"Unable to generate forecast: {e}"
        )

        return pd.DataFrame()


# ==========================================================
# PLANNING DASHBOARD
# ==========================================================

def show_planning_dashboard():

    forecast = load_forecast()

    st.title(
        " Planning Dashboard"
    )

    st.caption(
        "Demand Forecast & Inventory Planning"
    )

    st.divider()

    if forecast.empty:

        st.warning(
            "No forecast data available."
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

    filtered = forecast.copy()

    if selected_store != "All":

        filtered = filtered[
            filtered["store_nbr"] == selected_store
        ]

    if selected_family != "All":

        filtered = filtered[
            filtered["family"].astype(str)
            == selected_family
        ]

    st.divider()

    # =====================================================
    # KPI SECTION
    # =====================================================

    total_forecast = filtered[
        "Forecast"
    ].sum()

    avg_forecast = filtered[
        "Forecast"
    ].mean()

    stores_count = filtered[
        "store_nbr"
    ].nunique()

    family_count = filtered[
        "family"
    ].nunique()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(

        "Forecast Demand",

        f"{total_forecast:,.0f}"

    )

    c2.metric(

        "Average Demand",

        f"{avg_forecast:,.2f}"

    )

    c3.metric(

        "Stores",

        stores_count

    )

    c4.metric(

        "Families",

        family_count

    )

    st.divider()

    # =====================================================
    # DAILY FORECAST
    # =====================================================

    if "date" in filtered.columns:

        st.subheader(
            "Forecast Trend"
        )

        daily = (

            filtered.groupby("date")

            ["Forecast"]

            .sum()

            .reset_index()

        )

        fig = px.line(

            daily,

            x="date",

            y="Forecast",

            markers=True,

            title="Daily Demand Forecast"

        )

        st.plotly_chart(

            fig,

            width="stretch"

        )

        st.divider()

    # =====================================================
    # STORE DEMAND
    # =====================================================

    st.subheader(
        "Demand by Store"
    )

    stores = (

        filtered.groupby("store_nbr")

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

        title="Forecast by Store"

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
        "Demand by Product Family"
    )

    products = (

        filtered.groupby("family")

        ["Forecast"]

        .sum()

        .sort_values(
            ascending=False
        )

        .head(20)

        .reset_index()

    )

    fig = px.bar(

        products,

        x="family",

        y="Forecast",

        title="Forecast by Product Family"

    )

    st.plotly_chart(

        fig,

        width="stretch"

    )

    st.divider()

    # =====================================================
    # INVENTORY PLAN
    # =====================================================

    st.subheader(
        "Inventory Planning"
    )

    inventory = (

        filtered.groupby(

            [

                "store_nbr",

                "family"

            ]

        )

        .agg(

            Forecast_Demand=(

                "Forecast",

                "sum"

            )

        )

        .reset_index()

    )

    inventory["Recommended_Stock"] = (

        inventory["Forecast_Demand"] * 1.15

    ).round()

    inventory["Safety_Stock"] = (

        inventory["Forecast_Demand"] * 0.15

    ).round()

    st.dataframe(

        inventory,

        width="stretch",

        hide_index=True

    )

    st.divider()

    # =====================================================
    # DOWNLOAD REPORT
    # =====================================================

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

                "📥 Download Planning Report",

                data=f,

                file_name=report.name,

                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

            )

    st.divider()

    st.success(
        """
Planning Insight

• Forecast demand supports replenishment planning.

• Safety stock is automatically calculated.

• Product-level forecasts assist procurement teams.

• Store-wise demand helps optimize distribution.
        """
    )

    st.caption(
        "Planning Dashboard | Store Sales Forecasting"
    )