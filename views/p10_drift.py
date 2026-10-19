import pandas as pd
import plotly.express as px
import streamlit as st

from src.drift import drift_report, invoice_level, simulate_drift
from src.ui import STATUS_COLORS, TEAL, header, load_data

header("Drift Monitoring", "Compare a reference window with a current window. Large shifts mean models need retraining.")

df, _ = load_data()
lo, hi = df["InvoiceDate"].min().date(), df["InvoiceDate"].max().date()
split = st.slider("Reference window ends / current window starts", lo, hi, value=(lo + (hi - lo) * 0.7))
ref, cur = df[df["InvoiceDate"] < pd.Timestamp(split)], df[df["InvoiceDate"] >= pd.Timestamp(split)]
if len(ref) < 500 or len(cur) < 500:
    st.warning("Each window needs at least 500 rows. Move the split date towards the middle of the range.")
    st.stop()

with st.expander("Drift simulator: inject artificial drift into the current window", expanded=False):
    on = st.checkbox("Enable simulation")
    s = st.columns(4)
    ps = s[0].slider("Price change", -0.5, 1.0, 0.0, 0.05, format="%.2f")
    qs = s[1].slider("Quantity change", -0.5, 1.0, 0.0, 0.05)
    ds = s[2].slider("Extra discount (pts)", 0, 30, 0)
    ms = s[3].slider("Share moved to mobile app", 0.0, 0.8, 0.0, 0.05)
    if on:
        cur = simulate_drift(cur, ps, qs, ds, ms)

rep = drift_report(ref, cur)
worst = rep.loc[rep["PSI"].idxmax()]
n_drift, n_warn = int((rep["Status"] == "Drift").sum()), int((rep["Status"] == "Warning").sum())
k = st.columns(4)
k[0].metric("Reference rows", f"{len(ref):,}")
k[1].metric("Current rows", f"{len(cur):,}")
k[2].metric("Features drifting", n_drift, f"{n_warn} warning")
k[3].metric("Highest PSI", f"{worst['PSI']:.3f}", worst["Feature"], delta_color="off")
st.session_state["drift_summary"] = {"drift": n_drift, "warn": n_warn, "worst": worst["Feature"], "psi": float(worst["PSI"])}

if n_drift:
    st.error(f"Retraining recommended: {n_drift} feature(s) have PSI of 0.25 or more. Largest shift is in **{worst['Feature']}**.")
elif n_warn:
    st.warning(f"{n_warn} feature(s) show moderate drift (PSI 0.10 to 0.25). Keep watching.")
else:
    st.success("All monitored features are stable.")

st.dataframe(rep, hide_index=True, width="stretch", column_config={
    "PSI": st.column_config.ProgressColumn("PSI", min_value=0, max_value=0.5, format="%.3f"),
    "KS p-value": st.column_config.NumberColumn(format="%.4f"), "Ref mean": st.column_config.NumberColumn(format="%.2f"), "Current mean": st.column_config.NumberColumn(format="%.2f")})
st.caption("PSI below 0.10 is stable, 0.10 to 0.25 is a warning, 0.25 or more is drift. The KS test p-value below 0.05 means the two distributions differ significantly.")

a, b = st.columns(2)
num = a.selectbox("Numeric feature", ["Basket_Value", "Basket_Size", "Avg_Price", "Discount_Pct", "Satisfaction"])
ri, ci = invoice_level(ref), invoice_level(cur)
if num == "Basket_Value" or num == "Basket_Size":
    cap = ri[num].quantile(0.99)
    ri, ci = ri[ri[num] <= cap], ci[ci[num] <= cap]
long = pd.concat([ri[[num]].assign(Window="Reference"), ci[[num]].assign(Window="Current")])
a.plotly_chart(px.histogram(long, x=num, color="Window", barmode="overlay", histnorm="probability density", nbins=40, opacity=0.65,
                            color_discrete_map={"Reference": TEAL, "Current": "#F59E0B"}), width="stretch")
cat = b.selectbox("Categorical feature", ["Category", "Channel", "Region", "Payment_Method"])
share = pd.concat([ref[cat].value_counts(normalize=True).rename("Reference"), cur[cat].value_counts(normalize=True).rename("Current")], axis=1).fillna(0).reset_index().melt(cat, var_name="Window", value_name="Share")
b.plotly_chart(px.bar(share, x=cat, y="Share", color="Window", barmode="group", color_discrete_map={"Reference": TEAL, "Current": "#F59E0B"}), width="stretch")
