import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.inventory import abc_xyz
from src.ui import CORAL, TEAL, TEAL_SCALE, filtered, header, money, need_data

header("Product Performance", "Which products drive revenue and profit, and how stable their demand is.")
df = filtered()
if need_data(df):
    s = df[~df["Return_Flag"]]
    t1, t2, t3 = st.tabs(["Top products", "ABC-XYZ analysis", "Margin and pricing"])
    with t1:
        n = st.slider("Number of products", 5, 50, 15)
        by = st.radio("Rank by", ["Net_Revenue", "Profit", "Quantity"], horizontal=True)
        top = s.groupby(["StockCode", "Description", "Category"]).agg(Net_Revenue=("Net_Revenue", "sum"), Profit=("Profit", "sum"), Quantity=("Quantity", "sum")).reset_index().nlargest(n, by)
        st.plotly_chart(px.bar(top.sort_values(by), x=by, y="Description", color="Category", orientation="h", height=max(380, n * 24)), width="stretch")
        st.dataframe(top, hide_index=True, width="stretch")
    with t2:
        t = abc_xyz(df)
        agg = t.reset_index().rename(columns={"index": "StockCode"})
        pareto = t.reset_index(drop=True)
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_bar(x=list(range(len(pareto))), y=pareto["Revenue"], name="Revenue", marker_color=TEAL, secondary_y=False)
        fig.add_scatter(x=list(range(len(pareto))), y=pareto["Cum_Share"] * 100, name="Cumulative %", line=dict(color=CORAL), secondary_y=True)
        fig.update_layout(xaxis_title="Products ranked by revenue", yaxis_title="Revenue", yaxis2_title="Cumulative share %")
        st.plotly_chart(fig, width="stretch")
        m = t.groupby(["ABC", "XYZ"]).size().unstack(fill_value=0).reindex(index=["A", "B", "C"], columns=["X", "Y", "Z"], fill_value=0)
        c1, c2 = st.columns([1, 1])
        with c1:
            st.write("**Products per ABC-XYZ class**")
            st.plotly_chart(px.imshow(m, text_auto=True, color_continuous_scale=TEAL_SCALE, labels=dict(x="Demand variability (X stable, Z erratic)", y="Value class")), width="stretch")
        with c2:
            st.markdown("- **A** products make 80% of revenue; **B** the next 15%; **C** the last 5%.\n- **X** products have steady monthly demand, **Y** moderate, **Z** erratic.\n- **AX** items deserve tight, automated replenishment; **CZ** items are candidates for delisting or made-to-order.")
    with t3:
        cat = s.groupby("Category").agg(Revenue=("Net_Revenue", "sum"), Profit=("Profit", "sum")).reset_index()
        cat["Margin %"] = cat["Profit"] / cat["Revenue"] * 100
        st.plotly_chart(px.bar(cat.sort_values("Margin %"), x="Margin %", y="Category", orientation="h", color_discrete_sequence=[TEAL]), width="stretch")
        p = s.groupby("StockCode").agg(Price=("UnitPrice", "median"), Units=("Quantity", "sum"), Category=("Category", "first")).reset_index()
        st.write("**Price versus units sold** (log scales)")
        st.plotly_chart(px.scatter(p, x="Price", y="Units", color="Category", log_x=True, log_y=True, opacity=0.6), width="stretch")
