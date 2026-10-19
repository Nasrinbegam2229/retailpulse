import plotly.graph_objects as go
import streamlit as st

from src.data import CATEGORY_KEYWORDS
from src.services import forecast_bundle
from src.ui import CORAL, TEAL, header, money

header("Demand Forecasting", "Forecast daily demand by category, compare models, and test what-if scenarios.")

a, b, c = st.columns(3)
metric = a.radio("Forecast", ["Net_Revenue", "Quantity"], format_func=lambda x: "Revenue (GBP)" if x == "Net_Revenue" else "Units", horizontal=True)
cat = b.selectbox("Category", ["All"] + sorted(list(CATEGORY_KEYWORDS) + ["General Merchandise"]))
horizon = c.slider("Horizon (days)", 7, 60, 30)

try:
    res = forecast_bundle(metric, cat, horizon)
except ValueError as e:
    st.error(str(e))
    st.stop()

models = list(res["metrics"]["Model"])
model = st.selectbox("Model shown", models, index=models.index(res["best"]), help="The best model on the last 30 days is preselected.")
w1, w2 = st.columns(2)
uplift = w1.slider("What-if: demand change (%)", -30, 50, 0, help="Scales the forecast, e.g. for a planned campaign or a supply shock.")
promo = w2.slider("What-if: extra promo days in horizon", 0, 14, 0, help="Adds a 20% lift on that many days.")

fut = res["future"][model].copy()
fut = fut * (1 + uplift / 100)
if promo:
    fut[:promo] = fut[:promo] * 1.2
band = 1.28 * res["resid_std"][model]
unit = money if metric == "Net_Revenue" else (lambda v: f"{v:,.0f}")
series, test = res["series"], res["test"]
prev = series.iloc[-horizon:].sum()
row = res["metrics"].set_index("Model").loc[model]

k = st.columns(4)
k[0].metric(f"Forecast, next {horizon} days", unit(fut.sum()))
k[1].metric("vs previous period", f"{fut.sum() / prev - 1:+.1%}" if prev else "n/a")
k[2].metric("Backtest MAPE (last 30 days)", f"{row['MAPE %']:.1f}%", help="MAPE ignores days with zero sales (the dataset has no Saturday trading).")
k[3].metric("Backtest WAPE", f"{row['WAPE %']:.1f}%", help="Weighted error: total absolute error divided by total actual demand. More stable than MAPE for daily data.")

hist = series.iloc[-120:]
fig = go.Figure()
fig.add_scatter(x=hist.index, y=hist.values, name="Actual", line=dict(color="#CBD5E1", width=1.5))
fig.add_scatter(x=test.index, y=res["val"][model], name="Backtest prediction", line=dict(color="#F59E0B", dash="dot"))
fig.add_scatter(x=res["future_dates"], y=fut + band, line=dict(width=0), showlegend=False, hoverinfo="skip")
fig.add_scatter(x=res["future_dates"], y=(fut - band).clip(min=0), fill="tonexty", fillcolor="rgba(14,124,123,0.15)", line=dict(width=0), name="80% band", hoverinfo="skip")
fig.add_scatter(x=res["future_dates"], y=fut, name="Forecast", line=dict(color=TEAL, width=3))
fig.update_layout(height=430, yaxis_title="Revenue (GBP)" if metric == "Net_Revenue" else "Units", legend=dict(orientation="h", y=1.1))
st.plotly_chart(fig, width="stretch")

out = {"date": res["future_dates"], "forecast": fut.round(1)}
st.download_button("Download forecast (CSV)", __import__("pandas").DataFrame(out).to_csv(index=False).encode(), "forecast.csv", "text/csv")
if model == "Hybrid":
    st.caption("Hybrid = " + " + ".join(f"{w:.0%} {n}" for n, w in res["weights"].items()) + " (weights are inversely proportional to backtest error).")
