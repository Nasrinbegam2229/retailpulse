"""Shared UI helpers: data loading, global filters, formatting."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src.data import build_dataset

# dark-theme palette (bright enough to read on #0B1220)
TEAL, CORAL, SLATE, SAND, INDIGO, AMBER = "#2DD4BF", "#FF6B5B", "#94A3B8", "#E3B341", "#818CF8", "#F59E0B"
TEAL_SCALE = [[0.0, "#134E5E"], [0.5, "#1FA6A0"], [1.0, "#7FF0DC"]]  # dim -> bright, readable on dark
REGION_COLORS = {"UK": TEAL, "Europe": INDIGO, "Rest of World": SAND}
STATUS_COLORS = {"Stockout Risk": CORAL, "Reorder Now": "#F59E0B", "Healthy": TEAL, "Overstock": INDIGO,
                 "High": CORAL, "Medium": "#F59E0B", "Low": TEAL, "Drift": CORAL, "Warning": "#F59E0B", "Stable": TEAL}


@st.cache_resource(show_spinner="Loading data (first run builds the enriched dataset)...")
def load_data() -> tuple[pd.DataFrame, str]:
    return build_dataset()


def filtered() -> pd.DataFrame:
    """Full data narrowed by the sidebar filters (read-only; do not mutate)."""
    df, _ = load_data()
    d = st.session_state.get("f_dates")
    if d and len(d) == 2:
        df = df[(df["InvoiceDate"] >= pd.Timestamp(d[0])) & (df["InvoiceDate"] < pd.Timestamp(d[1]) + pd.Timedelta(days=1))]
    if st.session_state.get("f_region"):
        df = df[df["Region"].isin(st.session_state["f_region"])]
    if st.session_state.get("f_channel"):
        df = df[df["Channel"].isin(st.session_state["f_channel"])]
    return df


def header(title: str, subtitle: str) -> None:
    st.title(title)
    st.caption(subtitle)


def money(x: float) -> str:
    a = abs(x)
    if a >= 1e6:
        return f"£{x / 1e6:,.2f}M"
    if a >= 1e3:
        return f"£{x / 1e3:,.1f}K"
    return f"£{x:,.0f}"


def need_data(df: pd.DataFrame) -> bool:
    if df.empty:
        st.warning("The current filters return no rows. Widen the date range or clear the region / channel filters in the sidebar.")
        return False
    return True


def inject_css() -> None:
    st.markdown("""<style>
    [data-testid="stMetric"]{background:#131C2E;border:1px solid #1F2A3F;border-radius:12px;padding:12px 16px}
    [data-testid="stMetricValue"]{font-size:1.7rem}
    </style>""", unsafe_allow_html=True)


def prior_period() -> pd.DataFrame:
    """Rows from the window of equal length just before the selected date range (same region / channel filters)."""
    df, _ = load_data()
    d = st.session_state.get("f_dates")
    if not (d and len(d) == 2):
        return df.iloc[0:0]
    start, end = pd.Timestamp(d[0]), pd.Timestamp(d[1]) + pd.Timedelta(days=1)
    df = df[(df["InvoiceDate"] >= start - (end - start)) & (df["InvoiceDate"] < start)]
    if st.session_state.get("f_region"):
        df = df[df["Region"].isin(st.session_state["f_region"])]
    if st.session_state.get("f_channel"):
        df = df[df["Channel"].isin(st.session_state["f_channel"])]
    return df
