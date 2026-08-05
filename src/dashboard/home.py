"""
=========================================================
Home Dashboard
=========================================================

Landing page for Store Sales Forecasting System

Author : ShawRick
Project: Store Sales Forecasting
"""

import streamlit as st
import pandas as pd

from src.utils.data_loader import load_train_data
from src.config import (
    MODEL_DIR,
    REPORT_DIR,
    PREDICTION_DIR,
)

# ==========================================================
# HOME PAGE
# ==========================================================

def show_home():

    df = load_train_data()

    st.title(
        " Store Sales Forecasting System"
    )

    st.caption(
        "Business Intelligence • Machine Learning • Demand Forecasting"
    )

    st.divider()

    # =====================================================
    # KPI SECTION
    # =====================================================

    total_sales = df["sales"].sum()

    total_stores = df["store_nbr"].nunique()

    total_products = df["family"].nunique()

    total_records = len(df)

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Total Sales",
        f"${total_sales:,.0f}"
    )

    c2.metric(
        "Stores",
        total_stores
    )

    c3.metric(
        "Product Families",
        total_products
    )

    c4.metric(
        "Transactions",
        f"{total_records:,}"
    )

    st.divider()

    # =====================================================
    # PROJECT OVERVIEW
    # =====================================================

    left, right = st.columns([2, 1])

    with left:

        st.subheader(
            "Project Overview"
        )

        st.markdown(
            """
This solution provides an end-to-end **Retail Demand Forecasting**
platform designed to support business decision-making.

### Key Capabilities

- Historical sales analytics
- Demand forecasting using Machine Learning
- Inventory planning
- Promotion analysis
- Store performance monitoring
- Financial reporting
- Automated Excel reports
- Interactive dashboards
            """
        )

    with right:

        st.subheader(
            "Technology Stack"
        )

        st.markdown(
            """
- Python
- Pandas
- XGBoost
- LightGBM
- Plotly
- Streamlit
- OpenPyXL
- Joblib
            """
        )

    st.divider()

    # =====================================================
    # PROJECT STATUS
    # =====================================================

    st.subheader(
        "Project Status"
    )

    status = pd.DataFrame({

        "Module":[

            "Data Audit",
            "EDA",
            "Statistics",
            "Cleaning",
            "Encoding",
            "Validation",
            "Feature Engineering",
            "Model Training",
            "Evaluation",
            "Prediction",
            "Excel Reporting",
            "Dashboard"

        ],

        "Status":[

            "Completed",
            "Completed",
            "Completed",
            "Completed",
            "Completed",
            "Completed",
            "Completed",
            "Completed",
            "Completed",
            "Completed",
            "Completed",
            "Completed"

        ]

    })

    st.dataframe(

        status,

        hide_index=True,

        width="stretch"

    )

    st.divider()

    # =====================================================
    # SYSTEM FILES
    # =====================================================

    st.subheader(
        "Generated Project Assets"
    )

    files = [

        (
            "XGBoost Model",
            MODEL_DIR / "xgboost_model.pkl"
        ),

        (
            "LightGBM Model",
            MODEL_DIR / "lightgbm_model.pkl"
        ),

        (
            "Feature Columns",
            MODEL_DIR / "feature_columns.pkl"
        ),

        (
            "Training Metadata",
            MODEL_DIR / "training_metadata.pkl"
        ),

        (
            "Forecast Report",
            REPORT_DIR / "Demand_Forecast_Report.xlsx"
        ),

        (
            "Prediction File",
            PREDICTION_DIR / "sales_prediction.csv"
        )

    ]

    rows = []

    for name, path in files:

        rows.append({

            "Asset": name,

            "Available": "✅" if path.exists() else "❌",

            "Location": str(path)

        })

    st.dataframe(

        pd.DataFrame(rows),

        hide_index=True,

        width="stretch"

    )

    st.divider()

    # =====================================================
    # BUSINESS OBJECTIVES
    # =====================================================

    st.subheader(
        "Business Objectives"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.success(
            """
✓ Improve forecasting accuracy

✓ Reduce stock shortages

✓ Reduce overstock inventory

✓ Optimize promotions
            """
        )

    with col2:

        st.success(
            """
✓ Support executive decisions

✓ Automate reporting

✓ Improve operational efficiency

✓ Increase revenue visibility
            """
        )

    st.divider()

    st.caption(
        "Store Sales Forecasting System | Version 1.0"
    )