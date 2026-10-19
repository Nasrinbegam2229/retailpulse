import pandas as pd
import plotly.express as px
import streamlit as st

from src.data import CATEGORY_KEYWORDS
from src.services import forecast_bundle
from src.ui import CORAL, TEAL, header

header("Model Comparison", "Backtest every forecasting model on the same 30 held-out days and pick the winner.")

a, b = st.columns(2)
metric = a.radio("Series", ["Net_Revenue", "Quantity"], format_func=lambda x: "Revenue" if x == "Net_Revenue" else "Units", horizontal=True)
cat = b.selectbox("Category", ["All"] + sorted(list(CATEGORY_KEYWORDS) + ["General Merchandise"]))
res = forecast_bundle(metric, cat, 30)
m = res["metrics"].copy()
best = m.iloc[0]
st.success(f"Best model: **{best['Model']}** with MAPE {best['MAPE %']:.1f}% and WAPE {best['WAPE %']:.1f}% on the last 30 days.")

l, r = st.columns(2)
l.plotly_chart(px.bar(m.sort_values("MAPE %", ascending=False), x="MAPE %", y="Model", orientation="h", color_discrete_sequence=[TEAL], title="MAPE (lower is better)")
               .add_vline(x=12, line_dash="dash", line_color=CORAL, annotation_text="12% target"), width="stretch")
r.plotly_chart(px.bar(m.sort_values("RMSE", ascending=False), x="RMSE", y="Model", orientation="h", color_discrete_sequence=["#6366F1"], title="RMSE (lower is better)"), width="stretch")
st.dataframe(m, hide_index=True, width="stretch", column_config={"MAPE %": st.column_config.NumberColumn(format="%.1f"), "WAPE %": st.column_config.NumberColumn(format="%.1f"), "RMSE": st.column_config.NumberColumn(format="%.0f")})

st.subheader("Predicted versus actual on the backtest window")
test = res["test"]
long = pd.DataFrame({"Actual": test.values, **{n: res["val"][n] for n in ["Hybrid", res["best"]] if n in res["val"]}}, index=test.index).reset_index(names="Date").melt("Date", var_name="Series", value_name="Value")
st.plotly_chart(px.line(long, x="Date", y="Value", color="Series"), width="stretch")
st.caption("Hybrid blends the two best non-baseline models, weighted by inverse backtest error: " + ", ".join(f"{n} {w:.0%}" for n, w in res["weights"].items()))

st.subheader("Leaderboard across categories")
st.caption("Trains every model for every category, which takes a minute or two the first time.")
if st.button("Run leaderboard"):
    rows = []
    bar = st.progress(0.0)
    cats = sorted(list(CATEGORY_KEYWORDS) + ["General Merchandise"])
    for i, c in enumerate(cats):
        try:
            r_ = forecast_bundle(metric, c, 30)
            top = r_["metrics"].iloc[0]
            rows.append({"Category": c, "Best model": top["Model"], "MAPE %": top["MAPE %"], "WAPE %": top["WAPE %"]})
        except Exception:
            pass
        bar.progress((i + 1) / len(cats))
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
