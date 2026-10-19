import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.inventory import compute_policy
from src.services import inventory_base
from src.ui import CORAL, STATUS_COLORS, TEAL, header, money

header("Inventory Optimization", "Safety stock, reorder points and order quantities driven by demand and supplier lead time.")

base = inventory_base()
with st.sidebar:
    st.markdown("**Inventory policy**")
    sl = st.slider("Service level", 0.80, 0.995, 0.95, 0.005, format="%.3f", help="Probability of not stocking out during lead time.")
    oc = st.number_input("Cost per order (GBP)", 1.0, 500.0, 25.0)
    hr = st.slider("Annual holding cost (% of unit cost)", 5, 40, 20) / 100
    mult = st.slider("Demand scenario (x forecast)", 0.5, 2.0, 1.0, 0.05, help="Use the uplift you tested on the forecasting page.")

inv = compute_policy(base, sl, oc, hr, mult)
vc = inv["Status"].value_counts()
k = st.columns(5)
k[0].metric("Products tracked", f"{len(inv):,}")
k[1].metric("Stockout risk", int(vc.get("Stockout Risk", 0)), help="Stock on hand will run out before a new order can arrive.")
k[2].metric("Reorder now", int(vc.get("Reorder Now", 0)))
k[3].metric("Overstocked", int(vc.get("Overstock", 0)), help="More than 90 days of cover.")
k[4].metric("Capital tied in excess stock", money(inv["Excess_Capital"].sum()))

t1, t2, t3 = st.tabs(["Recommendations", "Stock vs reorder point", "Purchase orders"])
with t1:
    a, b = st.columns([1, 2])
    status = a.multiselect("Status", list(STATUS_COLORS)[:4], default=["Stockout Risk", "Reorder Now"])
    cat = b.multiselect("Category", sorted(inv["Category"].unique()))
    v = inv[inv["Status"].isin(status)]
    if cat:
        v = v[v["Category"].isin(cat)]
    cols = ["StockCode", "Description", "Category", "Supplier", "Stock_On_Hand", "AvgDailyDemand", "Lead_Time_Days", "Safety_Stock", "Reorder_Point", "EOQ", "Days_Of_Cover", "Order_Qty", "Status"]
    st.dataframe(v.sort_values("Revenue", ascending=False)[cols], hide_index=True, width="stretch",
                 column_config={"AvgDailyDemand": st.column_config.NumberColumn("Daily demand", format="%.1f")})
    st.plotly_chart(px.pie(inv, names="Status", hole=0.5, color="Status", color_discrete_map=STATUS_COLORS, title="Stock health of the range"), width="stretch")
with t2:
    n = st.slider("Top products by revenue", 10, 60, 25)
    top = inv.nlargest(n, "Revenue").sort_values("Revenue", ascending=False)
    lab = top["StockCode"] + " " + top["Description"].str.slice(0, 18)
    f = go.Figure()
    f.add_bar(x=lab, y=top["Stock_On_Hand"], name="Stock on hand", marker_color=TEAL)
    f.add_scatter(x=lab, y=top["Reorder_Point"], name="Reorder point", mode="markers", marker=dict(color=CORAL, size=10, symbol="diamond"))
    f.update_layout(height=470, xaxis_tickangle=-60)
    st.plotly_chart(f, width="stretch")
    st.caption("Bars below the red marker should be reordered. Stock on hand is simulated because the public dataset has no inventory table.")
with t3:
    po = inv[inv["Order_Qty"] > 0].copy()
    po["Order_Value"] = po["Order_Qty"] * po["UnitCost"]
    sup = po.groupby("Supplier").agg(Lines=("StockCode", "count"), Units=("Order_Qty", "sum"), Order_Value=("Order_Value", "sum")).reset_index()
    st.write("**Orders to place, grouped by supplier**")
    st.dataframe(sup, hide_index=True, width="stretch", column_config={"Order_Value": st.column_config.NumberColumn("Order value", format="£%.0f")})
    st.download_button("Download purchase orders (CSV)", po[["Supplier", "StockCode", "Description", "Order_Qty", "UnitCost", "Order_Value", "Lead_Time_Days"]].sort_values("Supplier").to_csv(index=False).encode(), "purchase_orders.csv", "text/csv")
    with st.expander("How the numbers are calculated"):
        st.markdown("- **Safety stock** = z x daily demand std x sqrt(lead time), where z comes from the service level.\n- **Reorder point** = daily demand x lead time + safety stock.\n- **EOQ** = sqrt(2 x annual demand x order cost / (holding rate x unit cost)).\n- **Order quantity** = reorder point + EOQ - stock on hand, when stock is at or below the reorder point.")
