"""Cached, no-argument wrappers around the heavy computations (so every page reuses the same results)."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src import features, forecast, inventory
from src.realtime import LiveStream
from src.ui import load_data


@st.cache_resource(show_spinner="Building customer features...")
def rfm_table() -> pd.DataFrame:
    df, _ = load_data()
    return features.customer_features(df, df["InvoiceDate"].max().normalize() + pd.Timedelta(days=1))


@st.cache_data(show_spinner="Clustering customers...")
def segments(k: int) -> pd.DataFrame:
    return features.segment_customers(rfm_table(), k)


@st.cache_data(show_spinner="Scanning cluster counts...")
def silhouette() -> pd.DataFrame:
    return features.silhouette_scan(rfm_table())


@st.cache_resource(show_spinner="Training churn model...")
def churn_bundle(horizon: int = 90) -> dict:
    df, _ = load_data()
    res = features.train_churn(df, horizon)
    res["scored"] = features.score_customers(res["model"], rfm_table())
    return res


@st.cache_data(show_spinner="Training forecasting models (about 10 seconds)...")
def forecast_bundle(metric: str, category: str, horizon: int) -> dict:
    df, _ = load_data()
    return forecast.run_forecast(forecast.daily_series(df, metric, category), horizon)


@st.cache_data(show_spinner="Computing demand statistics...")
def inventory_base() -> pd.DataFrame:
    df, _ = load_data()
    return inventory.product_table(df)


@st.cache_data(show_spinner="Classifying products...")
def abc_table() -> pd.DataFrame:
    df, _ = load_data()
    return inventory.abc_xyz(df)


@st.cache_resource(show_spinner="Starting the live order stream...")
def live_stream() -> LiveStream:
    """One stream shared by every browser session (state lives in memory until the app restarts)."""
    df, _ = load_data()
    return LiveStream(df, inventory.compute_policy(inventory_base()))
