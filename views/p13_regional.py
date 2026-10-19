import pandas as pd
import plotly.express as px
import streamlit as st

from src.ui import INDIGO, REGION_COLORS, SAND, TEAL, TEAL_SCALE, filtered, header, money, need_data

ISO3 = {"Australia": "AUS", "Austria": "AUT", "Bahrain": "BHR", "Belgium": "BEL", "Bermuda": "BMU", "Brazil": "BRA", "Canada": "CAN",
        "Cyprus": "CYP", "Czech Republic": "CZE", "Denmark": "DNK", "EIRE": "IRL", "Finland": "FIN", "France": "FRA", "Germany": "DEU",
        "Greece": "GRC", "Hong Kong": "HKG", "Iceland": "ISL", "Israel": "ISR", "Italy": "ITA", "Japan": "JPN", "Korea": "KOR",
        "Lebanon": "LBN", "Lithuania": "LTU", "Malta": "MLT", "Netherlands": "NLD", "Nigeria": "NGA", "Norway": "NOR", "Poland": "POL",
        "Portugal": "PRT", "RSA": "ZAF", "Saudi Arabia": "SAU", "Singapore": "SGP", "Spain": "ESP", "Sweden": "SWE",
        "Switzerland": "CHE", "Thailand": "THA", "USA": "USA", "United Arab Emirates": "ARE", "United Kingdom": "GBR"}
MONEY = {"Revenue": st.column_config.NumberColumn(format="£%.0f"), "Profit": st.column_config.NumberColumn(format="£%.0f"),
         "Avg order value": st.column_config.NumberColumn(format="£%.0f"), "Revenue share": st.column_config.NumberColumn(format="%.1f%%"),
         "Margin %": st.column_config.NumberColumn(format="%.1f%%"), "Return rate": st.column_config.NumberColumn(format="%.1f%%")}

header("Regional Sales", "How the UK, Europe and the rest of the world compare on revenue, profit, customers and delivery.")
df = filtered()
if need_data(df):
    s = df[~df["Return_Flag"]]
    reg = s.groupby("Region").agg(Revenue=("Net_Revenue", "sum"), Profit=("Profit", "sum"), Orders=("InvoiceNo", "nunique"),
                                  Customers=("CustomerID", "nunique")).reset_index()
    reg["Margin %"] = reg["Profit"] / reg["Revenue"] * 100
    reg["Avg order value"] = reg["Revenue"] / reg["Orders"].clip(lower=1)
    reg["Revenue share"] = reg["Revenue"] / reg["Revenue"].sum() * 100
    all_orders = df.groupby("Region")["InvoiceNo"].nunique()
    ret_orders = df[df["Return_Flag"]].groupby("Region")["InvoiceNo"].nunique()
    reg["Return rate"] = reg["Region"].map(ret_orders).fillna(0) / reg["Region"].map(all_orders).clip(lower=1) * 100
    reg = reg.sort_values("Revenue", ascending=False).reset_index(drop=True)

    cols = st.columns(len(reg))
    for c, (_, r) in zip(cols, reg.iterrows()):
        c.metric(r["Region"], money(r["Revenue"]), help="Net revenue from sales (returns excluded).")
        c.caption(f"{r['Revenue share']:.1f}% of revenue · {r['Margin %']:.1f}% margin · {int(r['Customers']):,} customers")

    t1, t2, t3, t4, t5 = st.tabs(["Region comparison", "Countries", "Region mix", "World map", "Delivery & shipping"])

    with t1:
        a, b = st.columns([1, 1.4])
        a.plotly_chart(px.pie(reg, names="Region", values="Revenue", hole=0.55, color="Region", color_discrete_map=REGION_COLORS,
                              title="Share of revenue"), width="stretch")
        m = s.groupby([pd.Grouper(key="InvoiceDate", freq="MS"), "Region"])["Net_Revenue"].sum().reset_index()
        b.plotly_chart(px.area(m, x="InvoiceDate", y="Net_Revenue", color="Region", color_discrete_map=REGION_COLORS,
                               labels={"Net_Revenue": "Net revenue (GBP)", "InvoiceDate": ""}, title="Monthly revenue by region"), width="stretch")
        st.dataframe(reg[["Region", "Revenue", "Profit", "Margin %", "Orders", "Customers", "Avg order value", "Revenue share", "Return rate"]],
                     hide_index=True, width="stretch", column_config=MONEY)

    with t2:
        g = s.groupby(["Region", "Country"]).agg(Revenue=("Net_Revenue", "sum"), Profit=("Profit", "sum"), Orders=("InvoiceNo", "nunique"),
                                                  Customers=("CustomerID", "nunique")).reset_index()
        g["Margin %"] = g["Profit"] / g["Revenue"] * 100
        g = g.sort_values("Revenue", ascending=False)
        excl = st.toggle("Exclude United Kingdom from the charts (it dominates the scale)", value=True)
        gv = g[g["Country"] != "United Kingdom"] if excl else g
        n = st.slider("Countries to show", 5, 25, min(12, max(5, len(gv))))
        top = gv.head(n).sort_values("Revenue")
        a, b = st.columns(2)
        a.plotly_chart(px.bar(top, x="Revenue", y="Country", orientation="h", color="Region", color_discrete_map=REGION_COLORS,
                              labels={"Revenue": "Net revenue (GBP)"}, title=f"Top {len(top)} countries by revenue"), width="stretch")
        b.plotly_chart(px.treemap(gv, path=["Region", "Country"], values="Revenue", color="Region", color_discrete_map=REGION_COLORS,
                                  title="Revenue split (area = revenue)"), width="stretch")
        st.dataframe(g, hide_index=True, width="stretch", column_config=MONEY)

    with t3:
        a, b = st.columns(2)
        dim = a.selectbox("Compare regions by", ["Category", "Channel", "Payment_Method", "Loyalty_Tier", "Age_Group", "Supplier"])
        metric = b.radio("Heatmap shows", ["Revenue", "Profit", "Margin %"], horizontal=True)
        x = s.groupby(["Region", dim]).agg(Revenue=("Net_Revenue", "sum"), Profit=("Profit", "sum")).reset_index()
        x["Margin %"] = x["Profit"] / x["Revenue"] * 100
        heat = x.pivot(index="Region", columns=dim, values=metric)
        st.plotly_chart(px.imshow(heat, text_auto=".3s" if metric != "Margin %" else ".1f", aspect="auto", color_continuous_scale=TEAL_SCALE,
                                  labels=dict(color=metric), title=f"{metric} by region and {dim}"), width="stretch")
        mix = px.bar(x, x="Region", y="Revenue", color=dim, labels={"Revenue": "% of the region's revenue"},
                     title=f"How each region's revenue splits by {dim}")
        mix.update_layout(barnorm="percent")
        st.plotly_chart(mix, width="stretch")

    with t4:
        c = s.groupby("Country")["Net_Revenue"].sum().reset_index()
        c["ISO"] = c["Country"].map(ISO3)
        lost = c[c["ISO"].isna()]
        c = c.dropna(subset=["ISO"])
        if st.toggle("Exclude United Kingdom from the colour scale", value=True, key="map_excl"):
            c = c[c["ISO"] != "GBR"]
        if c.empty:
            st.info("No countries to plot with the current filters.")
        else:
            fig = px.choropleth(c, locations="ISO", color="Net_Revenue", hover_name="Country", color_continuous_scale=TEAL_SCALE,
                                labels={"Net_Revenue": "Net revenue (GBP)"})
            fig.update_geos(showframe=False, showcoastlines=False, showcountries=True, countrycolor="#334155", landcolor="#1E293B",
                            bgcolor="rgba(0,0,0,0)", projection_type="natural earth")
            fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=480)
            st.plotly_chart(fig, width="stretch")
        if len(lost):
            st.caption("Not shown on the map (no single country): " + ", ".join(f"{a} ({money(b)})" for a, b in zip(lost["Country"], lost["Net_Revenue"])))

    with t5:
        if "Delivery_Status" not in df.columns:
            st.info("This view needs the delivery columns. Rebuild the dataset with `python -m src.data --build`.")
        else:
            d = s[s["Delivery_Status"].isin(["On Time", "Late"])].drop_duplicates("InvoiceNo")
            if d.empty:
                st.info("No home deliveries in the current selection (in-store orders are collected, not delivered).")
            else:
                q = d.groupby("Region").agg(Orders=("InvoiceNo", "nunique"), Avg_days=("Delivery_Days", "mean"),
                                            Late=("Delivery_Status", lambda v: (v == "Late").mean() * 100),
                                            Shipping_income=("Shipping_Cost", "sum"), Avg_shipping=("Shipping_Cost", "mean")).reset_index()
                k = st.columns(3)
                k[0].metric("Delivered orders", f"{len(d):,}")
                k[1].metric("Average delivery time", f"{d['Delivery_Days'].mean():.1f} days")
                k[2].metric("Late deliveries", f"{(d['Delivery_Status'] == 'Late').mean():.1%}")
                a, b = st.columns(2)
                a.plotly_chart(px.bar(q, x="Region", y="Avg_days", color="Region", color_discrete_map=REGION_COLORS,
                                      labels={"Avg_days": "Average days to deliver"}, title="Delivery time by region"), width="stretch")
                b.plotly_chart(px.bar(q, x="Region", y="Late", color="Region", color_discrete_map=REGION_COLORS,
                                      labels={"Late": "% of orders late"}, title="Late deliveries by region"), width="stretch")
                st.dataframe(q.rename(columns={"Avg_days": "Avg delivery days", "Late": "Late %", "Shipping_income": "Shipping income",
                                               "Avg_shipping": "Avg shipping fee"}), hide_index=True, width="stretch",
                             column_config={"Avg delivery days": st.column_config.NumberColumn(format="%.1f"), "Late %": st.column_config.NumberColumn(format="%.1f%%"),
                                            "Shipping income": st.column_config.NumberColumn(format="£%.0f"), "Avg shipping fee": st.column_config.NumberColumn(format="£%.2f")})
                st.caption("Delivery days, delivery status and shipping fees are simulated attributes (the public dataset has none), so this tab demonstrates the method.")
