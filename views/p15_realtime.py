import time

import plotly.express as px
import streamlit as st
from plotly.subplots import make_subplots

from src.services import live_stream
from src.ui import AMBER, CORAL, INDIGO, REGION_COLORS, TEAL, header, money

header("Real-time Analytics", "A live order stream: sales, alerts and stock levels refresh on their own every few seconds.")
stream = live_stream()

with st.expander("How this works"):
    st.markdown(
        "- The public dataset has no live shop feed, so the stream **replays real historical orders** (real products, prices, countries "
        "and basket sizes) at the rate you choose. The pipeline after that point is the same you would use on a real feed: "
        "**ingest, running totals, alerts, stock depletion, dashboard**.\n"
        "- **Time is compressed.** The history averages about 65 orders per trading day, so 6 orders per minute is roughly one trading day "
        "every 11 minutes (12 per minute: one every 5). Stock therefore drains much faster than in a real shop.\n"
        "- **Stock** starts from the Inventory page's stock levels and drops with every order. Crossing a product's reorder point or "
        "hitting zero raises an alert. Use *Place purchase orders* to replenish.\n"
        "- **Demand spike** alerts compare the last minute of orders with the 5 minutes before it (needs about 4+ orders per minute to be meaningful). "
        "Press *Simulate demand surge* to see one fire.\n"
        "- The stream is shared by everyone using the app and lives in memory (it restarts when the app restarts).")

c = st.columns([0.8, 2, 1.1, 1.4, 1.2])
live = c[0].toggle("Live", value=True, key="rt_live")
rate = c[1].slider("Orders per minute", 2, 120, stream.rate, key="rt_rate")
every = c[2].selectbox("Refresh every", [2, 3, 5, 10], index=1, format_func=lambda s: f"{s} seconds", key="rt_every")
if c[3].button("Simulate demand surge", icon=":material/bolt:", width="stretch", key="rt_surge"):
    stream.surge()
if c[4].button("Reset stream", icon=":material/restart_alt:", width="stretch", key="rt_reset"):
    stream.reset()
stream.set_rate(rate)
stream.start() if live else stream.pause()

SEV = {"critical": "red", "warning": "orange", "info": "blue"}
STOCK_KINDS, FLOW_KINDS = {"Stockout", "Reorder point", "Purchase orders"}, {"Demand spike", "High-value order", "Simulation"}


def alert_list(kinds: set[str], empty: str) -> None:
    rows = stream.recent_alerts(8, kinds)
    if not rows:
        st.caption(empty)
    for r in rows:
        st.markdown(f":{SEV[r['severity']]}[**{r['type']}**] {r['message']}  \n:gray[{r['time']}]")


def panel():
    stream.advance()
    k = stream.kpis()
    spike = stream.recent_alerts(1, {"Demand spike"})
    if spike and time.time() - spike[0]["ts"] < 120:
        st.error(f"**Demand spike at {spike[0]['time']}**: {spike[0]['message']}", icon=":material/warning:")
    m = st.columns(6)
    m[0].metric("Orders", f"{k['orders']:,}")
    m[1].metric("Revenue", money(k["revenue"]), f"{k['rev5'] / k['prev5'] - 1:+.0%} last 5 min vs before" if k["prev5"] else None)
    m[2].metric("Gross profit", money(k["profit"]))
    m[3].metric("Units sold", f"{k['units']:,}")
    m[4].metric("Avg order value", money(k["aov"]))
    m[5].metric("Orders / minute", f"{k['per_min']:.0f}", help="Average over the last 2 minutes.")

    a, b = st.columns([2.2, 1])
    s = stream.series(60, 30)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_bar(x=s["Time"], y=s["Revenue"], name="Revenue per minute", marker_color=TEAL)
    fig.add_scatter(x=s["Time"], y=s["Orders"], name="Orders per minute", mode="lines+markers", line=dict(color=AMBER, width=2), secondary_y=True)
    fig.update_layout(title="Last 30 minutes", height=340, margin=dict(t=50, b=10), legend=dict(orientation="h", y=-0.2))
    fig.update_yaxes(title_text="GBP", secondary_y=False)
    fig.update_yaxes(title_text="Orders", secondary_y=True, showgrid=False)
    a.plotly_chart(fig, width="stretch", key="rt_series")
    a.caption("The newest bar fills up as the minute progresses.")

    with b:
        with st.container(height=190, border=True):
            st.markdown("**Demand & order alerts**")
            alert_list(FLOW_KINDS, "Nothing unusual yet.")
        with st.container(height=190, border=True):
            st.markdown("**Stock alerts**")
            alert_list(STOCK_KINDS, "No stock alerts yet.")

    x, y, z = st.columns(3)
    tp = stream.top_products(8).sort_values("Units")
    x.plotly_chart(px.bar(tp, x="Units", y="Product", orientation="h", color_discrete_sequence=[TEAL], title="Top products by units sold", height=320),
                   width="stretch", key="rt_top")
    bc = stream.by_category(8).sort_values("Revenue")
    y.plotly_chart(px.bar(bc, x="Revenue", y="Category", orientation="h", color_discrete_sequence=[INDIGO], title="Revenue by category", height=320),
                   width="stretch", key="rt_cat")
    br = stream.by_region()
    if len(br):
        z.plotly_chart(px.pie(br, names="Region", values="Revenue", hole=0.55, color="Region", color_discrete_map=REGION_COLORS,
                              title="Revenue by region", height=320), width="stretch", key="rt_reg")

    t1, t2 = st.tabs(["Live orders", "Stock watchlist"])
    with t1:
        o = stream.recent_orders(20)
        st.dataframe(o, hide_index=True, width="stretch", column_config={"Revenue": st.column_config.NumberColumn(format="£%.2f")})
        st.download_button("Download the last 500 orders (CSV)", stream.recent_orders(500).to_csv(index=False), "live_orders.csv", "text/csv", key="rt_dl")
    with t2:
        ss = stream.stock_summary()
        q = st.columns(4)
        q[0].metric("Products tracked", f"{ss['tracked']:,}")
        q[1].metric("At or below reorder point", f"{ss['low']:,}")
        q[2].metric("Stockouts", f"{ss['out']:,}")
        if q[3].button("Place purchase orders", icon=":material/local_shipping:", width="stretch", key="rt_restock",
                       help="Order every flagged product up to its reorder point plus EOQ (instant delivery in this demo)."):
            st.toast(f"Stock ordered for {stream.restock():,} products", icon=":material/check_circle:")
        w = stream.watchlist(15)
        if w.empty:
            st.caption("Products appear here as they start selling.")
        else:
            st.dataframe(w, hide_index=True, width="stretch", column_config={
                "Pct_Of_Reorder_Point": st.column_config.ProgressColumn("Stock vs reorder point", min_value=0, max_value=300, format="%d%%"),
                "Days_Of_Cover": st.column_config.NumberColumn("Days of cover", format="%.1f"), "Reorder_Point": "Reorder point"})
            st.caption("Products that sold since the stream started, closest to running out first. Stock levels are simulated (the public dataset has no inventory table).")


st.fragment(run_every=f"{every}s" if live else None)(panel)()
