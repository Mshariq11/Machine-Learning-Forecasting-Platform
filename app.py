"""
=========================================================
Store Sales Forecasting Dashboard
=========================================================

Main Streamlit Application

Author : ShawRick
Project: Store Sales Forecasting
"""

import streamlit as st

from src.dashboard.home import show_home
from src.dashboard.executive import show_executive_dashboard
from src.dashboard.planning import show_planning_dashboard
from src.dashboard.inventory import show_inventory_dashboard
from src.dashboard.marketing import show_marketing_dashboard
from src.dashboard.operations import show_operations_dashboard
from src.dashboard.finance import show_finance_dashboard
from src.dashboard.forecast import show_forecast_dashboard


# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(

    page_title="Store Sales Forecasting",

    page_icon="📈",

    layout="wide",

    initial_sidebar_state="expanded"

)


# ==========================================================
# CUSTOM STYLE
# ==========================================================

st.markdown(
    """
<style>

#MainMenu {
    visibility:hidden;
}

footer {
    visibility:hidden;
}

header {
    visibility:hidden;
}

.block-container{
    padding-top:1rem;
    padding-bottom:2rem;
    padding-left:2rem;
    padding-right:2rem;
}

section[data-testid="stSidebar"]{
    background:#0F172A;
}

section[data-testid="stSidebar"] *{
    color:white;
}

.sidebar-title{
    font-size:28px;
    font-weight:bold;
    text-align:center;
    color:white;
}

.sidebar-subtitle{
    text-align:center;
    color:#CBD5E1;
    font-size:14px;
}

.dashboard-title{
    font-size:34px;
    font-weight:bold;
    color:#0F172A;
}

.kpi-card{

    background:#F8FAFC;

    padding:20px;

    border-radius:12px;

    border:1px solid #E2E8F0;

}

</style>
""",
    unsafe_allow_html=True
)


# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:

    st.markdown(
        '<p class="sidebar-title"> Retail BI</p>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<p class="sidebar-subtitle">'
        'Store Sales Forecasting System'
        '</p>',
        unsafe_allow_html=True
    )

    st.divider()

    page = st.radio(

        "Navigation",

        [

            "Home",

            "Executive",

            "Planning",

            "Inventory",

            "Marketing",

            "Operations",

            "Finance",

            "Forecast"

        ]

    )

    st.divider()

    st.markdown("### Project")

    st.markdown("""
    - Retail Demand Forecasting
    - Machine Learning
    - Business Intelligence
    - Automated Reporting
    """)

    st.divider()

    st.caption(
        "Version 1.0"
    )

    st.caption(
        "© 2026 Shariq Zia"
    )


# ==========================================================
# PAGE ROUTER
# ==========================================================

PAGES = {
    "Home": show_home,
    "Executive": show_executive_dashboard,
    "Planning": show_planning_dashboard,
    "Inventory": show_inventory_dashboard,
    "Marketing": show_marketing_dashboard,
    "Operations": show_operations_dashboard,
    "Finance": show_finance_dashboard,
    "Forecast": show_forecast_dashboard,
}

# ==========================================================
# RUN SELECTED PAGE
# ==========================================================

PAGES[page]()