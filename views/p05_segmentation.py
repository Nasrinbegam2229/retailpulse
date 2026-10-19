import plotly.express as px
import streamlit as st

from src.services import rfm_table, segments, silhouette
from src.ui import TEAL, header, money

header("Customer Segmentation", "RFM features clustered with K-Means, then named by business meaning.")

PLAYBOOK = {
    "Champions": "Reward them: early access, loyalty perks, referral asks.",
    "Loyal Customers": "Upsell higher-margin products and bundle offers.",
    "Potential Loyalists": "Offer membership and a second-purchase incentive.",
    "Promising": "Nurture with product recommendations and onboarding emails.",
    "Needs Attention": "Send limited-time offers before they go quiet.",
    "At Risk": "Run win-back campaigns with personalised discounts.",
    "Hibernating": "Low-cost reactivation email; do not over-invest.",
    "Lost": "Last-chance survey or exclude from paid campaigns.",
}

rfm = rfm_table()
k = st.slider("Number of segments (k)", 3, 8, 5, help="Guideline target is 6-8 meaningful segments; the silhouette chart helps you choose.")
seg = segments(k)

a, b = st.columns([1, 2])
with a:
    sc = silhouette()
    st.write("**Silhouette score by k** (higher means cleaner clusters)")
    st.plotly_chart(px.line(sc, x="k", y="Silhouette", markers=True, color_discrete_sequence=[TEAL]).update_layout(height=280), width="stretch")
with b:
    st.write("**Customers by recency and spend** (log scales)")
    st.plotly_chart(px.scatter(seg.reset_index(), x="Recency", y="Monetary", color="Segment", log_y=True, opacity=0.65,
                               hover_data=["CustomerID", "Frequency"]).update_layout(height=320), width="stretch")

prof = seg.groupby("Segment").agg(Customers=("Recency", "size"), Recency=("Recency", "mean"), Frequency=("Frequency", "mean"),
                                   Avg_Spend=("Monetary", "mean"), Total_Revenue=("Monetary", "sum"), Avg_Satisfaction=("Satisfaction", "mean")).reset_index()
prof["Revenue share"] = prof["Total_Revenue"] / prof["Total_Revenue"].sum()
prof = prof.sort_values("Total_Revenue", ascending=False)
prof["Playbook"] = prof["Segment"].map(PLAYBOOK)
st.subheader("Segment profiles")
st.dataframe(prof, hide_index=True, width="stretch", column_config={
    "Recency": st.column_config.NumberColumn("Recency (days)", format="%.0f"), "Frequency": st.column_config.NumberColumn("Orders", format="%.1f"),
    "Avg_Spend": st.column_config.NumberColumn("Avg spend", format="£%.0f"), "Total_Revenue": st.column_config.NumberColumn("Total revenue", format="£%.0f"),
    "Avg_Satisfaction": st.column_config.NumberColumn("Satisfaction", format="%.2f"),
    "Revenue share": st.column_config.ProgressColumn("Revenue share", format="percent", min_value=0, max_value=1)})

c1, c2 = st.columns(2)
c1.plotly_chart(px.pie(prof, names="Segment", values="Customers", hole=0.5, title="Customers per segment"), width="stretch")
c2.plotly_chart(px.pie(prof, names="Segment", values="Total_Revenue", hole=0.5, title="Revenue per segment"), width="stretch")

st.subheader("Explore one segment")
pick = st.selectbox("Segment", prof["Segment"])
sub = seg[seg["Segment"] == pick]
m = st.columns(3)
m[0].metric("Customers", f"{len(sub):,}")
m[1].metric("Revenue", money(sub["Monetary"].sum()))
m[2].metric("Avg order value", money(sub["AvgOrderValue"].mean()))
st.info(PLAYBOOK[pick])
st.dataframe(sub.reset_index().sort_values("Monetary", ascending=False).head(200), hide_index=True, width="stretch")
st.download_button("Download this segment (CSV)", sub.reset_index().to_csv(index=False).encode(), f"segment_{pick.replace(' ', '_')}.csv", "text/csv")
