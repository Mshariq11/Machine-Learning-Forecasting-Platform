"""
=========================================================
Inventory Dashboard
=========================================================

Inventory Planning Dashboard

Author : Shariq Zia
Project: Store Sales Forecasting
"""

import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import REPORT_DIR
from src.utils.forecast_loader import load_forecast

# ==========================================================
# INVENTORY DASHBOARD
# ==========================================================

def show_inventory_dashboard():

    forecast = load_forecast()

    st.title(
        " Inventory Dashboard"
    )

    st.caption(
        "Inventory Planning & Stock Optimization"
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
    # INVENTORY CALCULATIONS
    # =====================================================

    inventory = (

        df.groupby(

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

    inventory["Safety_Stock"] = (

        inventory["Forecast_Demand"] * 0.15

    ).round()

    inventory["Recommended_Stock"] = (

        inventory["Forecast_Demand"]

        +

        inventory["Safety_Stock"]

    ).round()

    inventory["Reorder_Point"] = (

        inventory["Forecast_Demand"] * 0.50

    ).round()

    inventory["Maximum_Stock"] = (

        inventory["Recommended_Stock"] * 1.20

    ).round()

    inventory["Inventory_Status"] = inventory[
        "Forecast_Demand"
    ].apply(

        lambda x:

        "High Demand"

        if x >= 500

        else

        "Medium Demand"

        if x >= 150

        else

        "Low Demand"

    )

    # =====================================================
    # KPI SECTION
    # =====================================================

    total_stock = inventory[
        "Recommended_Stock"
    ].sum()

    total_safety = inventory[
        "Safety_Stock"
    ].sum()

    total_reorder = inventory[
        "Reorder_Point"
    ].sum()

    sku_count = len(inventory)

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(

        "Recommended Stock",

        f"{total_stock:,.0f}"

    )

    c2.metric(

        "Safety Stock",

        f"{total_safety:,.0f}"

    )

    c3.metric(

        "Reorder Points",

        f"{total_reorder:,.0f}"

    )

    c4.metric(

        "Inventory Items",

        sku_count

    )

    st.divider()

    # =====================================================
    # INVENTORY STATUS
    # =====================================================

    st.subheader(
        "Inventory Status"
    )

    status = (

        inventory.groupby(

            "Inventory_Status"

        )

        .size()

        .reset_index(

            name="Items"

        )

    )

    fig = px.pie(

        status,

        names="Inventory_Status",

        values="Items",

        hole=0.45,

        title="Inventory Classification"

    )

    st.plotly_chart(

        fig,

        width="stretch"

    )

    st.divider()

    # =====================================================
    # RECOMMENDED STOCK
    # =====================================================

    st.subheader(
        "Top Recommended Stock"
    )

    top_stock = (

        inventory

        .sort_values(

            "Recommended_Stock",

            ascending=False

        )

        .head(20)

    )

    fig = px.bar(

        top_stock,

        x="family",

        y="Recommended_Stock",

        color="Inventory_Status",

        title="Recommended Stock"

    )

    st.plotly_chart(

        fig,

        width="stretch"

    )

    st.divider()

    # =====================================================
    # STORE INVENTORY
    # =====================================================

    st.subheader(
        "Store Inventory Requirements"
    )

    store_inventory = (

        inventory.groupby(

            "store_nbr"

        )

        [

            "Recommended_Stock"

        ]

        .sum()

        .reset_index()

    )

    fig = px.bar(

        store_inventory,

        x="store_nbr",

        y="Recommended_Stock",

        title="Recommended Stock by Store"

    )

    st.plotly_chart(

        fig,

        width="stretch"

    )

    st.divider()

    # =====================================================
    # INVENTORY TABLE
    # =====================================================

    st.subheader(
        "Inventory Planning Table"
    )

    st.dataframe(

        inventory.sort_values(

            "Recommended_Stock",

            ascending=False

        ),

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

                "📥 Download Inventory Report",

                data=f,

                file_name=report.name,

                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

            )

    st.divider()

    st.info(
        """
Inventory Recommendations

• Recommended Stock = Forecast + Safety Stock

• Safety Stock = 15% of forecast demand

• Reorder Point = 50% of forecast demand

• Maximum Stock = 120% of recommended stock

• High-demand products should receive replenishment priority.
        """
    )

    st.caption(
        "Inventory Dashboard | Store Sales Forecasting"
    )