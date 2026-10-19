import pandas as pd
import plotly.express as px
import streamlit as st

from src.services import inventory_base
from src.inventory import compute_policy
from src.ui import TEAL, filtered, header, money, need_data

header("Executive Overview", "Revenue, profit and customer health at a glance. Use the sidebar filters to slice the business.")
df = filtered()
if need_data(df):
    s = df[~df["Return_Flag"]]
    ret = df[df["Return_Flag"]]
    orders = s["InvoiceNo"].nunique()
    c = st.columns(6)
    c[0].metric("Net revenue", money(s["Net_Revenue"].sum()))
    c[1].metric("Gross profit", money(s["Profit"].sum()), f"{s['Profit'].sum() / s['Net_Revenue'].sum():.1%} margin")
    c[2].metric("Orders", f"{orders:,}")
    c[3].metric("Customers", f"{s['CustomerID'].nunique():,}")
    c[4].metric("Avg order value", money(s["Net_Revenue"].sum() / max(orders, 1)))
    c[5].metric("Return rate", f"{ret['InvoiceNo'].nunique() / max(df['InvoiceNo'].nunique(), 1):.1%}")

    m = s.groupby(pd.Grouper(key="InvoiceDate", freq="MS")).agg(Revenue=("Net_Revenue", "sum"), Profit=("Profit", "sum")).reset_index()
    st.subheader("Monthly revenue and profit")
    st.plotly_chart(px.bar(m, x="InvoiceDate", y=["Revenue", "Profit"], barmode="group",
                           color_discrete_sequence=[TEAL, "#F59E0B"], labels={"value": "GBP", "InvoiceDate": ""}), width="stretch")

    a, b = st.columns(2)
    with a:
        st.subheader("Revenue by category")
        cat = s.groupby("Category")["Net_Revenue"].sum().sort_values().reset_index()
        st.plotly_chart(px.bar(cat, x="Net_Revenue", y="Category", orientation="h", color_discrete_sequence=[TEAL],
                               labels={"Net_Revenue": "Net revenue (GBP)"}), width="stretch")
    with b:
        st.subheader("Revenue by channel and region")
        ch = s.groupby(["Region", "Channel"])["Net_Revenue"].sum().reset_index()
        st.plotly_chart(px.bar(ch, x="Region", y="Net_Revenue", color="Channel", barmode="stack",
                               labels={"Net_Revenue": "Net revenue (GBP)"}), width="stretch")

    inv = compute_policy(inventory_base())
    risk = int((inv["Status"] == "Stockout Risk").sum())
    reo = int((inv["Status"] == "Reorder Now").sum())
    over = int((inv["Status"] == "Overstock").sum())
    st.subheader("Where to look next")
    x = st.columns(3)
    x[0].metric("Products at stockout risk", risk, help="Stock on hand is below demand during supplier lead time.")
    x[1].metric("Products to reorder", reo)
    x[2].metric("Overstocked products", over, help="More than 90 days of cover.")
    st.info("Open **Inventory Optimization** to place orders, **Churn Prediction** to see customers likely to leave, or **Alerts & Reports** to see everything that needs attention.")
