"""
=========================================================
Operations Dashboard
=========================================================

Operations Dashboard

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
# OPERATIONS DASHBOARD
# ==========================================================

def show_operations_dashboard():

    df = load_data()

    st.title(
        " Operations Dashboard"
    )

    st.caption(
        "Store Operations & Performance Monitoring"
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

    operations = df.copy()

    operations = operations[
        operations["year"] == selected_year
    ]

    operations = operations[
        operations["month"] == selected_month
    ]

    if selected_store != "All":

        operations = operations[
            operations["store_nbr"] == selected_store
        ]

    st.divider()

    # =====================================================
    # KPI SECTION
    # =====================================================

    total_sales = operations["sales"].sum()

    avg_sales = operations["sales"].mean()

    total_transactions = len(operations)

    active_stores = operations[
        "store_nbr"
    ].nunique()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Revenue",
        f"${total_sales:,.0f}"
    )

    c2.metric(
        "Average Sale",
        f"${avg_sales:,.2f}"
    )

    c3.metric(
        "Transactions",
        f"{total_transactions:,}"
    )

    c4.metric(
        "Active Stores",
        active_stores
    )

    st.divider()

    # =====================================================
    # DAILY SALES
    # =====================================================

    st.subheader(
        "Daily Store Operations"
    )

    daily = (

        operations.groupby("date")

        ["sales"]

        .sum()

        .reset_index()

    )

    fig = px.line(

        daily,

        x="date",

        y="sales",

        markers=True,

        title="Daily Sales"

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
        "Store Performance"
    )

    stores = (

        operations.groupby("store_nbr")

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

    fig = px.bar(

        stores,

        x="store_nbr",

        y="Revenue",

        title="Store Revenue"

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

    st.divider()

    # =====================================================
    # CITY PERFORMANCE
    # =====================================================

    if "city" in operations.columns:

        st.subheader(
            "City Performance"
        )

        city = (

            operations.groupby("city")

            ["sales"]

            .sum()

            .sort_values(
                ascending=False
            )

            .head(20)

            .reset_index()

        )

        fig = px.bar(

            city,

            x="city",

            y="sales",

            title="Revenue by City"

        )

        st.plotly_chart(

            fig,

            use_container_width=True

        )

        st.divider()

    # =====================================================
    # STORE TYPE
    # =====================================================

    if "type_x" in operations.columns:

        st.subheader(
            "Store Type Performance"
        )

        store_type = (

            operations.groupby("type_x")

            ["sales"]

            .sum()

            .reset_index()

        )

        fig = px.pie(

            store_type,

            names="type_x",

            values="sales",

            hole=0.45,

            title="Revenue by Store Type"

        )

        st.plotly_chart(

            fig,

            use_container_width=True

        )

        st.divider()

    # =====================================================
    # CLUSTER PERFORMANCE
    # =====================================================

    if "cluster" in operations.columns:

        st.subheader(
            "Cluster Performance"
        )

        cluster = (

            operations.groupby("cluster")

            ["sales"]

            .sum()

            .reset_index()

        )

        fig = px.bar(

            cluster,

            x="cluster",

            y="sales",

            title="Revenue by Cluster"

        )

        st.plotly_chart(

            fig,

            use_container_width=True

        )

        st.divider()

    # =====================================================
    # OPERATIONS SUMMARY
    # =====================================================

    st.subheader(
        "Operational Summary"
    )

    st.dataframe(

        stores,

        use_container_width=True,

        hide_index=True

    )

    st.divider()

    st.info(
        """
Operations Insights

• Monitor store productivity and operational efficiency.

• Compare revenue across stores and cities.

• Evaluate performance by store type and cluster.

• Track operational trends for daily decision-making.
        """
    )

    st.caption(
        "Operations Dashboard | Store Sales Forecasting"
    )