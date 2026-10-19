import pandas as pd
import plotly.express as px
import streamlit as st

from src.data import COLUMN_DICT, quality_checks
from src.ui import TEAL, filtered, header, load_data, need_data

header("Data Explorer", "What data the platform uses, which columns came from the public dataset and which were engineered.")
df = filtered()
if need_data(df):
    full, source = load_data()
    c = st.columns(4)
    c[0].metric("Rows", f"{len(df):,}")
    c[1].metric("Columns", df.shape[1])
    c[2].metric("Products", f"{df['StockCode'].nunique():,}")
    c[3].metric("Date span", f"{(df['InvoiceDate'].max() - df['InvoiceDate'].min()).days} days")
    st.caption(f"Source: {source}")

    t1, t2, t3, t4, t5 = st.tabs(["Column dictionary", "Data quality", "Sample rows", "Distributions", "Correlations"])
    with t1:
        d = pd.DataFrame(COLUMN_DICT, columns=["Column", "Origin", "Description"])
        n_orig = (d["Origin"] == "Original").sum()
        st.write(f"**{n_orig} original columns** from the public dataset were extended with **{len(d) - n_orig} groups of engineered columns**.")
        st.dataframe(d, width="stretch", hide_index=True,
                     column_config={"Origin": st.column_config.TextColumn(width="small")})
    with t2:
        q = quality_checks(full)
        st.dataframe(q.style.map(lambda v: "color:#0E7C7B;font-weight:600" if v == "Pass" else ("color:#E4572E;font-weight:600" if v == "Fail" else ""), subset=["Status"]),
                     width="stretch", hide_index=True)
        miss = df.isna().mean().mul(100).round(2)
        miss = miss[miss > 0].rename("Missing %").reset_index().rename(columns={"index": "Column"})
        if len(miss):
            st.write("**Missing values**")
            st.dataframe(miss, hide_index=True, width="stretch")
    with t3:
        st.dataframe(df.head(500), width="stretch", hide_index=True)
    with t4:
        num = [c for c in ["Quantity", "UnitPrice", "Revenue", "Net_Revenue", "Profit", "Discount_Pct", "Satisfaction", "Lead_Time_Days"] if c in df]
        col = st.selectbox("Numeric column", num)
        clip = st.checkbox("Trim the top 1% outliers", value=True)
        x = df[col]
        if clip:
            x = x[x <= x.quantile(0.99)]
        st.plotly_chart(px.histogram(x, nbins=50, color_discrete_sequence=[TEAL]).update_layout(showlegend=False, xaxis_title=col), width="stretch")
        cat = st.selectbox("Categorical column", ["Category", "Channel", "Region", "Payment_Method", "Loyalty_Tier", "Age_Group", "Season_Event", "Supplier"])
        v = df[cat].value_counts().reset_index()
        st.plotly_chart(px.bar(v, x=cat, y="count", color_discrete_sequence=[TEAL]), width="stretch")
    with t5:
        cols = ["Quantity", "UnitPrice", "Discount_Pct", "Net_Revenue", "Profit", "Satisfaction", "Lead_Time_Days", "Hour", "DayOfWeek"]
        corr = df[cols].corr().round(2)
        st.plotly_chart(px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto"), width="stretch")
