"""
=========================================================
Finance Dashboard
=========================================================

Finance Dashboard

Author : Shariq Zia
Project: Store Sales Forecasting
"""

import pandas as pd
import plotly.express as px
import streamlit as st
from src.utils.data_loader import load_train_data

# ==========================================================
# FINANCE DASHBOARD
# ==========================================================

def show_finance_dashboard():

    df = load_train_data()

    st.title(
        " Finance Dashboard"
    )

    st.caption(
        "Revenue Analysis & Financial Performance"
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
        index=len(years)-1
    )

    months = sorted(
        df["month"].unique()
    )

    selected_month = c2.selectbox(
        "Month",
        months
    )

    finance = df.copy()

    finance = finance[
        finance["year"] == selected_year
    ]

    finance = finance[
        finance["month"] == selected_month
    ]

    st.divider()

    # =====================================================
    # KPI SECTION
    # =====================================================

    revenue = finance["sales"].sum()

    avg_sale = finance["sales"].mean()

    transactions = len(finance)

    revenue_per_transaction = (

        revenue /

        transactions

        if transactions > 0

        else 0

    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Revenue",
        f"${revenue:,.0f}"
    )

    c2.metric(
        "Average Sale",
        f"${avg_sale:,.2f}"
    )

    c3.metric(
        "Transactions",
        f"{transactions:,}"
    )

    c4.metric(
        "Revenue / Transaction",
        f"${revenue_per_transaction:,.2f}"
    )

    st.divider()

    # =====================================================
    # DAILY REVENUE
    # =====================================================

    st.subheader(
        "Daily Revenue"
    )

    daily = (

        finance.groupby("date")

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

        width="stretch"

    )

    st.divider()

    # =====================================================
    # MONTHLY REVENUE
    # =====================================================

    st.subheader(
        "Monthly Revenue"
    )

    monthly = (

        finance.groupby("month")

        ["sales"]

        .sum()

        .reset_index()

    )

    fig = px.bar(

        monthly,

        x="month",

        y="sales",

        title="Monthly Revenue"

    )

    st.plotly_chart(

        fig,

        width="stretch"

    )

    st.divider()

    # =====================================================
    # STORE REVENUE
    # =====================================================

    st.subheader(
        "Top Revenue Stores"
    )

    stores = (

        finance.groupby("store_nbr")

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

        stores.head(20),

        x="store_nbr",

        y="Revenue",

        title="Revenue by Store"

    )

    st.plotly_chart(

        fig,

        width="stretch"

    )

    st.divider()

    # =====================================================
    # PRODUCT REVENUE
    # =====================================================

    st.subheader(
        "Revenue by Product Family"
    )

    products = (

        finance.groupby("family")

        ["sales"]

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

        y="sales",

        title="Product Revenue"

    )

    st.plotly_chart(

        fig,

        width="stretch"

    )

    st.divider()

    # =====================================================
    # STORE TYPE
    # =====================================================

    if "type_x" in finance.columns:

        st.subheader(
            "Revenue by Store Type"
        )

        store_type = (

            finance.groupby("type_x")

            ["sales"]

            .sum()

            .reset_index()

        )

        fig = px.pie(

            store_type,

            names="type_x",

            values="sales",

            hole=0.45,

            title="Store Type Revenue"

        )

        st.plotly_chart(

            fig,

            width="stretch"

        )

        st.divider()

    # =====================================================
    # FINANCIAL SUMMARY
    # =====================================================

    st.subheader(
        "Financial Summary"
    )

    st.dataframe(

        stores,

        width="stretch",

        hide_index=True

    )

    st.divider()

    st.success(
        """
Financial Insights

• Revenue trends help monitor business growth.

• Store-wise revenue highlights top-performing locations.

• Product family revenue identifies the most profitable categories.

• Average sales and transaction metrics support financial planning.

• Monthly revenue assists budgeting and forecasting decisions.
        """
    )

    st.caption(
        "Finance Dashboard | Store Sales Forecasting"
    )