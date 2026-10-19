import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.features import CHURN_FEATURES
from src.services import churn_bundle
from src.ui import CORAL, STATUS_COLORS, TEAL, header, money

header("Churn Prediction", "Which customers are likely to stop buying in the next period, and how much revenue is at stake.")

horizon = st.select_slider("Churn definition: no purchase in the next ... days", [60, 90, 120], value=90)
res = churn_bundle(horizon)
m = res["metrics"]

st.caption(f"Training setup: features are built from data up to {res['cutoff']:%d %b %Y}; the label is 'did not buy in the following {horizon} days'. "
           "Recency is computed before the label window, so the model cannot see the answer.")
c = st.columns(5)
c[0].metric("AUC-ROC", f"{m['AUC-ROC']:.3f}", "meets 0.88 target" if m["AUC-ROC"] >= 0.88 else "target 0.88", delta_color="normal" if m["AUC-ROC"] >= 0.88 else "off")
c[1].metric("Precision @ top 20%", f"{m['Precision@Top20%']:.2f}", "meets 0.75 target" if m["Precision@Top20%"] >= 0.75 else "target 0.75", delta_color="normal" if m["Precision@Top20%"] >= 0.75 else "off")
c[2].metric("F1", f"{m['F1']:.2f}")
c[3].metric("Precision", f"{m['Precision']:.2f}")
c[4].metric("Churn rate", f"{m['Churn rate']:.0%}")

t1, t2, t3, t4 = st.tabs(["Model quality", "At-risk customers", "Score one customer", "Explainability (SHAP)"])
with t1:
    a, b = st.columns(2)
    roc = res["roc"]
    f = go.Figure()
    f.add_scatter(x=roc["FPR"], y=roc["TPR"], name="Model", line=dict(color=TEAL, width=3))
    f.add_scatter(x=[0, 1], y=[0, 1], name="Random", line=dict(color="#94A3B8", dash="dash"))
    f.update_layout(title="ROC curve", xaxis_title="False positive rate", yaxis_title="True positive rate", height=360)
    a.plotly_chart(f, width="stretch")
    imp = res["importance"].sort_values("Importance")
    b.plotly_chart(px.bar(imp, x="Importance", y="Feature", orientation="h", color_discrete_sequence=[TEAL],
                          title="Permutation importance (drop in AUC when shuffled)").update_layout(height=360), width="stretch")
with t2:
    sc = res["scored"].reset_index()
    a, b, d = st.columns(3)
    risk = a.multiselect("Risk level", ["High", "Medium", "Low"], default=["High"])
    max_rec = b.slider("Bought within the last (days)", 30, 365, 180, help="Customers silent for longer are probably already lost.")
    tiers = d.multiselect("Loyalty tier", sorted(sc["Loyalty_Tier"].dropna().unique()), default=[])
    v = sc[sc["Risk"].isin(risk) & (sc["Recency"] <= max_rec)]
    if tiers:
        v = v[v["Loyalty_Tier"].isin(tiers)]
    x = st.columns(3)
    x[0].metric("Customers listed", f"{len(v):,}")
    x[1].metric("Revenue at risk", money(v["Revenue_At_Risk"].sum()))
    x[2].metric("Avg churn probability", f"{v['Churn_Prob'].mean():.0%}" if len(v) else "n/a")
    dist = sc["Risk"].value_counts().reindex(["Low", "Medium", "High"]).reset_index()
    st.plotly_chart(px.bar(dist, x="Risk", y="count", color="Risk", color_discrete_map=STATUS_COLORS).update_layout(height=260, showlegend=False), width="stretch")
    cols = ["CustomerID", "Country", "Loyalty_Tier", "Recency", "Frequency", "Monetary", "Churn_Prob", "Risk", "Revenue_At_Risk"]
    st.dataframe(v.sort_values("Revenue_At_Risk", ascending=False)[cols].head(300), hide_index=True, width="stretch", column_config={
        "Churn_Prob": st.column_config.ProgressColumn("Churn probability", min_value=0, max_value=1, format="percent"),
        "Monetary": st.column_config.NumberColumn(format="£%.0f"), "Revenue_At_Risk": st.column_config.NumberColumn("Revenue at risk", format="£%.0f")})
    st.download_button("Download at-risk list (CSV)", v[cols].to_csv(index=False).encode(), "at_risk_customers.csv", "text/csv")
with t3:
    sc = res["scored"]
    cid = st.selectbox("Customer", sc.sort_values("Revenue_At_Risk", ascending=False).index[:500])
    row = sc.loc[cid]
    p = float(row["Churn_Prob"])
    g = go.Figure(go.Indicator(mode="gauge+number", value=p * 100, number={"suffix": "%"}, title={"text": "Churn probability"},
                               gauge={"axis": {"range": [0, 100]}, "bar": {"color": CORAL if p > 0.7 else "#F59E0B" if p > 0.4 else TEAL}}))
    g.update_layout(height=280)
    a, b = st.columns([1, 1])
    a.plotly_chart(g, width="stretch")
    b.dataframe(row[CHURN_FEATURES].astype(float).round(2).rename("Value").reset_index().rename(columns={"index": "Feature"}), hide_index=True, width="stretch")
with t4:
    try:
        import matplotlib.pyplot as plt
        import shap
        Xs = res["X_test"].sample(min(300, len(res["X_test"])), random_state=0)
        sv = shap.TreeExplainer(res["model"]).shap_values(Xs)
        shap.summary_plot(sv, Xs, show=False)
        st.pyplot(plt.gcf(), clear_figure=True)
        st.caption("Each dot is a customer. Red = high feature value; right of centre = pushes the churn probability up.")
    except ImportError:
        st.info("Install SHAP to see this view:  `pip install shap`  (listed in requirements.txt).")
