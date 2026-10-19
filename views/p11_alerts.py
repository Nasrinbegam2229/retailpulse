import io

import pandas as pd
import streamlit as st

from src.drift import drift_report
from src.inventory import compute_policy
from src.services import churn_bundle, inventory_base
from src.ui import header, load_data, money

header("Alerts & Reports", "Everything that needs attention today, with exports for the team.")

df, _ = load_data()
inv = compute_policy(inventory_base())
crit = inv[inv["Status"] == "Stockout Risk"].sort_values("Revenue", ascending=False)
reo = inv[inv["Status"] == "Reorder Now"].sort_values("Revenue", ascending=False)
over = inv[inv["Status"] == "Overstock"].sort_values("Excess_Capital", ascending=False)
sc = churn_bundle(90)["scored"].reset_index()
hv = sc[(sc["Risk"] == "High") & (sc["Recency"] <= 180)].sort_values("Revenue_At_Risk", ascending=False)
cut = df["InvoiceDate"].min() + (df["InvoiceDate"].max() - df["InvoiceDate"].min()) * 0.7
rep = drift_report(df[df["InvoiceDate"] < cut], df[df["InvoiceDate"] >= cut])
drifting = rep[rep["Status"] != "Stable"]

k = st.columns(4)
k[0].metric("Stockout risk", len(crit))
k[1].metric("Reorder now", len(reo))
k[2].metric("High-risk active customers", len(hv), money(hv["Revenue_At_Risk"].sum()) + " at risk", delta_color="off")
k[3].metric("Drifting features", len(drifting))

t1, t2, t3, t4 = st.tabs(["Inventory", "Customers", "Data drift", "Reports"])
with t1:
    if len(crit):
        st.error(f"{len(crit)} products will run out before a new order can arrive.")
    st.dataframe(pd.concat([crit, reo]).head(100)[["StockCode", "Description", "Supplier", "Stock_On_Hand", "Reorder_Point", "Lead_Time_Days", "Order_Qty", "Status"]], hide_index=True, width="stretch")
    if len(over):
        st.warning(f"{money(over['Excess_Capital'].sum())} is tied up in {len(over)} overstocked products.")
with t2:
    st.dataframe(hv.head(100)[["CustomerID", "Country", "Loyalty_Tier", "Recency", "Monetary", "Churn_Prob", "Revenue_At_Risk"]], hide_index=True, width="stretch",
                 column_config={"Churn_Prob": st.column_config.ProgressColumn("Churn probability", min_value=0, max_value=1, format="percent")})
with t3:
    if len(drifting):
        st.warning("Features outside the stable range (reference = first 70% of history, current = last 30%):")
        st.dataframe(drifting[["Feature", "PSI", "Status"]], hide_index=True, width="stretch")
    else:
        st.success("No drift detected between the reference and current windows.")
with t4:
    s = df[~df["Return_Flag"]]
    summary = pd.DataFrame({"Metric": ["Net revenue", "Gross profit", "Orders", "Customers", "Stockout-risk products", "Reorder products", "Overstock capital", "High-risk customers", "Revenue at risk"],
                            "Value": [round(s["Net_Revenue"].sum()), round(s["Profit"].sum()), s["InvoiceNo"].nunique(), s["CustomerID"].nunique(), len(crit), len(reo), round(over["Excess_Capital"].sum()), len(hv), round(hv["Revenue_At_Risk"].sum())]})
    st.dataframe(summary, hide_index=True, width="stretch")
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        summary.to_excel(xw, sheet_name="Summary", index=False)
        pd.concat([crit, reo]).to_excel(xw, sheet_name="Inventory alerts", index=False)
        hv.to_excel(xw, sheet_name="At-risk customers", index=False)
        rep.to_excel(xw, sheet_name="Drift", index=False)
    st.download_button("Download full report (Excel)", buf.getvalue(), "retailpulse_report.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", type="primary")
    st.download_button("Download inventory alerts (CSV)", pd.concat([crit, reo]).to_csv(index=False).encode(), "inventory_alerts.csv", "text/csv")
