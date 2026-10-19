import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.ui import AMBER, CORAL, INDIGO, SAND, TEAL, TEAL_SCALE, filtered, header, money, need_data, prior_period

GBP = {"Revenue": st.column_config.NumberColumn(format="£%.0f"), "Cost": st.column_config.NumberColumn(format="£%.0f"),
       "Profit": st.column_config.NumberColumn(format="£%.0f"), "Margin %": st.column_config.NumberColumn(format="%.1f%%"),
       "Share of profit": st.column_config.NumberColumn(format="%.1f%%")}


def pct_delta(now: float, before: float) -> str | None:
    return f"{now / before - 1:+.1%} vs previous period" if before else None


header("Profit & Revenue", "Where the money comes from, what it costs, and how much of it you keep.")
df = filtered()
if need_data(df):
    s, ret = df[~df["Return_Flag"]], df[df["Return_Flag"]]
    gross, net, cogs, profit = s["Revenue"].sum(), s["Net_Revenue"].sum(), s["Cost"].sum(), s["Profit"].sum()
    discounts, ret_value, ret_profit = gross - net, -ret["Net_Revenue"].sum(), ret["Profit"].sum()
    margin = profit / net if net else 0.0

    pv = prior_period()
    pv = pv[~pv["Return_Flag"]]
    p_net, p_profit = pv["Net_Revenue"].sum(), pv["Profit"].sum()
    p_margin = p_profit / p_net if p_net else None

    k = st.columns(5)
    k[0].metric("Net revenue", money(net), pct_delta(net, p_net), help="Sales after discounts, before returns.")
    k[1].metric("Gross profit", money(profit), pct_delta(profit, p_profit), help="Net revenue minus the cost of goods sold.")
    k[2].metric("Profit margin", f"{margin:.1%}", f"{(margin - p_margin) * 100:+.1f} pts vs previous period" if p_margin is not None else None)
    k[3].metric("Discounts given", money(discounts), help="Revenue given away on promotion days.")
    k[4].metric("Returned goods", money(ret_value), help="Value of refunded returns.")
    if pv.empty:
        st.caption("Tip: choose a shorter date range in the sidebar to see how each number changed against the period just before it.")

    t1, t2, t3, t4, t5 = st.tabs(["Revenue vs profit", "Profit bridge", "Profitability by", "Products", "Discounts & returns"])

    with t1:
        a, b = st.columns(2)
        gran = a.radio("Group by", ["Weekly", "Monthly", "Quarterly"], index=1, horizontal=True)
        freq = {"Weekly": "W", "Monthly": "MS", "Quarterly": "QS"}[gran]
        ts = s.groupby(pd.Grouper(key="InvoiceDate", freq=freq)).agg(Revenue=("Net_Revenue", "sum"), Profit=("Profit", "sum")).reset_index()
        ts["Margin %"] = ts["Profit"] / ts["Revenue"].where(ts["Revenue"] != 0) * 100
        ts["Growth %"] = ts["Revenue"].pct_change() * 100
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_bar(x=ts["InvoiceDate"], y=ts["Revenue"], name="Net revenue", marker_color=TEAL)
        fig.add_bar(x=ts["InvoiceDate"], y=ts["Profit"], name="Gross profit", marker_color=AMBER)
        fig.add_scatter(x=ts["InvoiceDate"], y=ts["Margin %"], name="Margin %", mode="lines+markers", line=dict(color=CORAL, width=2.5), secondary_y=True)
        fig.update_layout(barmode="group", title=f"{gran} revenue, profit and margin", legend=dict(orientation="h", y=-0.15))
        fig.update_yaxes(title_text="GBP", secondary_y=False)
        fig.update_yaxes(title_text="Margin %", secondary_y=True, showgrid=False)
        st.plotly_chart(fig, width="stretch")
        g = ts.dropna(subset=["Growth %"]).assign(Direction=lambda x: ["Growth" if v >= 0 else "Decline" for v in x["Growth %"]])
        if len(g):
            unit = {"Weekly": "week", "Monthly": "month", "Quarterly": "quarter"}[gran]
            gf = px.bar(g, x="InvoiceDate", y="Growth %", color="Direction", color_discrete_map={"Growth": TEAL, "Decline": CORAL},
                        title=f"Revenue growth vs previous {unit}", labels={"InvoiceDate": ""})
            gf.update_layout(showlegend=False)
            st.plotly_chart(gf, width="stretch")

    with t2:
        steps = [("Gross sales", gross, "absolute"), ("Discounts", -discounts, "relative"), ("Net revenue", net, "total"),
                 ("Cost of goods", -cogs, "relative"), ("Gross profit", profit, "total"),
                 ("Returns impact", ret_profit, "relative"), ("Profit after returns", profit + ret_profit, "total")]
        wf = go.Figure(go.Waterfall(x=[x[0] for x in steps], y=[x[1] for x in steps], measure=[x[2] for x in steps],
                                    text=[money(x[1]) for x in steps], textposition="outside",
                                    increasing=dict(marker=dict(color=TEAL)), decreasing=dict(marker=dict(color=CORAL)),
                                    totals=dict(marker=dict(color=INDIGO)), connector=dict(line=dict(color="#64748B"))))
        wf.update_layout(title="From gross sales to profit after returns", showlegend=False, yaxis_title="GBP", height=480)
        st.plotly_chart(wf, width="stretch")
        c = st.columns(3)
        c[0].metric("Cost of goods sold", money(cogs), f"{cogs / net:.0%} of net revenue" if net else None, delta_color="off")
        c[1].metric("Discounts as % of gross", f"{discounts / gross:.1%}" if gross else "n/a")
        c[2].metric("Profit after returns", money(profit + ret_profit), f"{ret_profit / profit:+.1%} vs gross profit" if profit else None, delta_color="inverse")
        st.caption("Net revenue and gross profit match the Executive Overview (returns excluded). The last two bars take refunded returns off the profit.")

    with t3:
        dims = ["Category", "Region", "Country", "Channel", "Supplier", "Price_Band", "Payment_Method", "Loyalty_Tier", "Age_Group", "Customer_Type"]
        dim = st.selectbox("Break profit down by", [d for d in dims if d in s.columns])
        g = s.groupby(dim).agg(Revenue=("Net_Revenue", "sum"), Cost=("Cost", "sum"), Profit=("Profit", "sum"), Orders=("InvoiceNo", "nunique")).reset_index()
        g["Margin %"] = g["Profit"] / g["Revenue"] * 100
        g["Share of profit"] = g["Profit"] / g["Profit"].sum() * 100
        g = g.sort_values("Profit", ascending=False)
        a, b = st.columns(2)
        a.plotly_chart(px.bar(g.head(15).sort_values("Profit"), x="Profit", y=dim, orientation="h", color="Margin %", color_continuous_scale=TEAL_SCALE,
                              title=f"Profit by {dim} (colour = margin)"), width="stretch")
        b.plotly_chart(px.scatter(g.head(30), x="Revenue", y="Margin %", size="Orders", color=dim if len(g) <= 12 else None, hover_name=dim,
                                  title="Big and profitable is top-right", color_discrete_sequence=[TEAL, AMBER, INDIGO, CORAL, SAND]), width="stretch")
        st.dataframe(g[[dim, "Revenue", "Cost", "Profit", "Margin %", "Share of profit", "Orders"]], hide_index=True, width="stretch", column_config=GBP)

    with t4:
        p = df.groupby("Description").agg(Revenue=("Net_Revenue", "sum"), Cost=("Cost", "sum"), Profit=("Profit", "sum")).reset_index()
        p = p[p["Revenue"] != 0]
        p["Margin %"] = p["Profit"] / p["Revenue"] * 100
        n = st.slider("Products to list", 5, 25, 10)
        a, b = st.columns(2)
        top, bot = p.nlargest(n, "Profit").sort_values("Profit"), p.nsmallest(n, "Profit").sort_values("Profit", ascending=False)
        a.plotly_chart(px.bar(top, x="Profit", y="Description", orientation="h", color_discrete_sequence=[TEAL], title=f"Top {n} products by profit"), width="stretch")
        b.plotly_chart(px.bar(bot, x="Profit", y="Description", orientation="h", color_discrete_sequence=[CORAL], title=f"Bottom {n} products by profit"), width="stretch")
        loss = int((p["Profit"] < 0).sum())
        st.metric("Loss-making products", f"{loss:,}", help="Products whose returns outweigh their sales profit in the selected period.")
        st.caption("Product profit here includes returns, so a product that is returned more than it sells shows a loss.")
        st.dataframe(p.sort_values("Profit", ascending=False), hide_index=True, width="stretch", column_config=GBP)

    with t5:
        a, b = st.columns(2)
        dd = s[s["Discount_Pct"] > 0].assign(Given=lambda x: x["Revenue"] - x["Net_Revenue"])
        if len(dd):
            by = dd.groupby("Category")["Given"].sum().sort_values().reset_index()
            a.plotly_chart(px.bar(by, x="Given", y="Category", orientation="h", color_discrete_sequence=[AMBER], labels={"Given": "Discount given (GBP)"},
                                  title="Discounts given by category"), width="stretch")
        else:
            a.info("No promotion discounts in the current selection.")
        if len(ret) and "Return_Reason" in ret.columns:
            rr = ret.assign(Value=-ret["Net_Revenue"]).groupby("Return_Reason")["Value"].sum().sort_values().reset_index()
            b.plotly_chart(px.bar(rr, x="Value", y="Return_Reason", orientation="h", color_discrete_sequence=[CORAL], labels={"Value": "Returned value (GBP)", "Return_Reason": ""},
                                  title="Returned value by reason"), width="stretch")
        else:
            b.info("No returns in the current selection.")
        st.caption("Discounts and return reasons are simulated attributes, so these charts illustrate the method rather than a real business effect.")
