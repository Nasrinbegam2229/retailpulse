import numpy as np
import pandas as pd
import pytest

from src.data import clean, enrich, make_demo, quality_checks
from src.drift import psi, simulate_drift, drift_report
from src.features import customer_features, segment_customers, train_churn
from src.forecast import daily_series, mape, run_forecast
from src.inventory import compute_policy, product_table


@pytest.fixture(scope="module")
def df():
    return enrich(clean(make_demo(n_customers=300, n_products=120)))


def test_enrichment_adds_columns(df):
    for c in ["Category", "Channel", "Promo_Flag", "Net_Revenue", "Profit", "Loyalty_Tier", "Lead_Time_Days", "Satisfaction",
              "Margin_Pct", "Price_Band", "Basket_Value", "Items_Per_Invoice", "Order_Size_Band", "Customer_Type",
              "Days_Since_Prev_Order", "Shipping_Cost", "Delivery_Days", "Delivery_Status", "Return_Reason"]:
        assert c in df.columns
    assert (df.loc[~df.Return_Flag, "Net_Revenue"] <= df.loc[~df.Return_Flag, "Revenue"] + 1e-9).all()


def test_enrichment_is_deterministic():
    raw = clean(make_demo(n_customers=100, n_products=60))
    a, b = enrich(raw), enrich(raw)
    pd.testing.assert_frame_equal(a, b)


def test_quality_checks_pass(df):
    assert (quality_checks(df)["Status"] == "Pass").all()


def test_churn_has_no_leakage_and_is_better_than_random(df):
    res = train_churn(df, 90)
    assert 0.6 < res["metrics"]["AUC-ROC"] < 0.995


def test_segmentation_names(df):
    rfm = customer_features(df, df.InvoiceDate.max() + pd.Timedelta(days=1))
    seg = segment_customers(rfm, 5)
    assert seg["Segment"].nunique() == 5


def test_forecast_shapes(df):
    res = run_forecast(daily_series(df, "Net_Revenue"), 14)
    assert len(res["future"]["Hybrid"]) == 14 and (res["future"]["Hybrid"] >= 0).all()


def test_inventory_policy(df):
    inv = compute_policy(product_table(df), 0.95)
    assert (inv["Reorder_Point"] >= inv["Safety_Stock"]).all()
    hi = compute_policy(product_table(df), 0.99)
    assert (hi["Safety_Stock"] >= inv["Safety_Stock"]).all()


def test_psi_detects_shift():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 5000)
    assert psi(a, rng.normal(0, 1, 5000)) < 0.1
    assert psi(a, rng.normal(1.5, 1, 5000)) > 0.25


def test_drift_simulator_triggers_alert(df):
    cut = df.InvoiceDate.min() + (df.InvoiceDate.max() - df.InvoiceDate.min()) * 0.7
    ref, cur = df[df.InvoiceDate < cut], df[df.InvoiceDate >= cut]
    rep = drift_report(ref, simulate_drift(cur, price_shift=0.6, qty_shift=0.5))
    assert (rep["Status"] == "Drift").any()
