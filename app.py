
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st

from src.ui import inject_css, load_data

st.set_page_config(page_title="RetailPulse", page_icon=":material/insights:", layout="wide")
inject_css()

# ---- login gate: until someone is signed in, the only page that exists is the login / sign-up page
if not st.session_state.get("user"):
    st.navigation([st.Page("views/login.py", title="Log in", icon=":material/lock:")], position="hidden").run()
    st.stop()

pages = {
    "Understand": [
        st.Page("views/p01_overview.py", title="Executive Overview", icon=":material/dashboard:", default=True),
        st.Page("views/p03_sales.py", title="Sales Analytics", icon=":material/show_chart:"),
        st.Page("views/p13_regional.py", title="Regional Sales", icon=":material/public:"),
        st.Page("views/p14_profit_revenue.py", title="Profit & Revenue", icon=":material/payments:"),
        st.Page("views/p04_products.py", title="Product Performance", icon=":material/category:"),
    ],
    "Customers": [
        st.Page("views/p05_segmentation.py", title="Customer Segmentation", icon=":material/groups:"),
        st.Page("views/p06_churn.py", title="Churn Prediction", icon=":material/person_off:"),
    ],
    "Plan": [
        st.Page("views/p07_forecast.py", title="Demand Forecasting", icon=":material/trending_up:"),
        st.Page("views/p08_models.py", title="Model Comparison", icon=":material/compare_arrows:"),
        st.Page("views/p09_inventory.py", title="Inventory Optimization", icon=":material/inventory_2:"),
    ],
    "Operate": [
        st.Page("views/p15_realtime.py", title="Real-time Analytics", icon=":material/bolt:"),
        st.Page("views/p11_alerts.py", title="Alerts & Reports", icon=":material/notifications_active:"),
    ],
}
nav = st.navigation(pages)

df, source = load_data()
user = st.session_state["user"]
if name := st.session_state.pop("_welcome", None):
    st.toast(f"Welcome, {name}!", icon=":material/waving_hand:")

with st.sidebar:
    st.markdown("**Global filters**")
    lo, hi = df["InvoiceDate"].min().date(), df["InvoiceDate"].max().date()
    st.date_input("Date range", value=(lo, hi), min_value=lo, max_value=hi, key="f_dates")
    st.multiselect("Region", sorted(df["Region"].unique()), key="f_region", placeholder="All regions")
    st.multiselect("Channel", sorted(df["Channel"].unique()), key="f_channel", placeholder="All channels")
    st.caption("Filters apply to Overview, Sales, Regional Sales, Profit & Revenue and Products. Customer, forecast and inventory pages always use the full history.")
    st.divider()
    st.caption(source)
    st.divider()
    st.markdown(f":material/account_circle: **{user['full_name']}**  \n<span style='opacity:.65'>@{user['username']}</span>", unsafe_allow_html=True)
    if st.button("Log out", icon=":material/logout:", width="stretch"):
        st.session_state.clear()
        st.rerun()

nav.run()
