import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.ui import CORAL, TEAL, TEAL_SCALE, filtered, header, money, need_data

header("Sales Analytics", "Trends, seasonality and the effect of promotions and holidays.")
df = filtered()
if need_data(df):
    s = df[~df["Return_Flag"]]
    t1, t2, t3, t4 = st.tabs(["Trend", "Breakdown", "Seasonality", "Promotions & holidays"])
    with t1:
        a, b = st.columns(2)
        gran = a.radio("Granularity", ["Daily", "Weekly", "Monthly"], index=1, horizontal=True)
        metric = b.radio("Metric", ["Net_Revenue", "Profit", "Quantity"], horizontal=True)
        freq = {"Daily": "D", "Weekly": "W", "Monthly": "MS"}[gran]
        ts = s.groupby(pd.Grouper(key="InvoiceDate", freq=freq))[metric].sum().reset_index()
        ts["Rolling avg"] = ts[metric].rolling(4 if gran != "Daily" else 7, min_periods=1).mean()
        fig = go.Figure()
        fig.add_scatter(x=ts["InvoiceDate"], y=ts[metric], name=metric, line=dict(color=TEAL, width=1.5))
        fig.add_scatter(x=ts["InvoiceDate"], y=ts["Rolling avg"], name="Rolling average", line=dict(color=CORAL, width=2.5))
        st.plotly_chart(fig, width="stretch")
    with t2:
        dim = st.selectbox("Break down by", ["Category", "Country", "Region", "Channel", "Payment_Method", "Age_Group", "Gender", "Loyalty_Tier", "Acquisition_Channel", "Supplier"])
        g = s.groupby(dim).agg(Revenue=("Net_Revenue", "sum"), Profit=("Profit", "sum"), Orders=("InvoiceNo", "nunique")).reset_index()
        g["Margin %"] = (g["Profit"] / g["Revenue"] * 100).round(1)
        g = g.sort_values("Revenue", ascending=False).head(15)
        st.plotly_chart(px.bar(g, x=dim, y="Revenue", color="Margin %", color_continuous_scale=TEAL_SCALE), width="stretch")
        st.dataframe(g, hide_index=True, width="stretch",
                     column_config={"Revenue": st.column_config.NumberColumn(format="£%.0f"), "Profit": st.column_config.NumberColumn(format="£%.0f")})
    with t3:
        order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        h = s.pivot_table(index="DayName", columns="Hour", values="Net_Revenue", aggfunc="sum").reindex(order)
        st.write("**Revenue by weekday and hour**")
        st.plotly_chart(px.imshow(h.fillna(0), color_continuous_scale=TEAL_SCALE, aspect="auto", labels=dict(color="GBP")), width="stretch")
        se = s.groupby("Season_Event").apply(lambda x: x["Net_Revenue"].sum() / x["InvoiceDate"].dt.normalize().nunique(), include_groups=False).rename("Avg daily revenue").reset_index()
        st.write("**Average daily revenue by season / event**")
        st.plotly_chart(px.bar(se.sort_values("Avg daily revenue"), x="Avg daily revenue", y="Season_Event", orientation="h", color_discrete_sequence=[TEAL]), width="stretch")
    with t4:
        day = s.groupby(s["InvoiceDate"].dt.normalize()).agg(Rev=("Net_Revenue", "sum"), Promo=("Promo_Flag", "max"), Holiday=("Holiday_Flag", "max")).reset_index()
        c = st.columns(3)
        p, n = day[day["Promo"]]["Rev"].mean(), day[~day["Promo"]]["Rev"].mean()
        c[0].metric("Avg daily revenue on promo days", money(p) if p == p else "n/a")
        c[1].metric("Avg daily revenue otherwise", money(n) if n == n else "n/a")
        c[2].metric("Promo uplift", f"{(p / n - 1):+.1%}" if p == p and n == n and n else "n/a")
        d = s[s["Promo_Flag"]].groupby("Discount_Pct").agg(Orders=("InvoiceNo", "nunique"), Revenue=("Net_Revenue", "sum")).reset_index()
        if len(d):
            st.plotly_chart(px.bar(d, x="Discount_Pct", y="Orders", color_discrete_sequence=[TEAL], labels={"Discount_Pct": "Discount %"}), width="stretch")
        hol = day.groupby("Holiday")["Rev"].mean().rename({True: "Bank holiday", False: "Regular day"}).reset_index()
        st.plotly_chart(px.bar(hol, x="Holiday", y="Rev", color_discrete_sequence=[CORAL], labels={"Rev": "Avg daily revenue", "Holiday": ""}), width="stretch")
        st.caption("Promotions and discounts are simulated attributes, so uplift here illustrates the method rather than a real business effect.")
