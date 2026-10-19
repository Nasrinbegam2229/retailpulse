"""Data drift: PSI, KS test, categorical drift and a drift simulator."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def psi(ref, cur, bins: int = 10) -> float:
    ref, cur = np.asarray(ref, float), np.asarray(cur, float)
    ref, cur = ref[~np.isnan(ref)], cur[~np.isnan(cur)]
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.clip(np.histogram(ref, edges)[0] / len(ref), 1e-4, None)
    c = np.clip(np.histogram(cur, edges)[0] / len(cur), 1e-4, None)
    return float(np.sum((c - r) * np.log(c / r)))


def psi_cat(ref: pd.Series, cur: pd.Series) -> float:
    r = ref.value_counts(normalize=True)
    c = cur.value_counts(normalize=True)
    idx = r.index.union(c.index)
    r, c = r.reindex(idx).fillna(0).clip(lower=1e-4), c.reindex(idx).fillna(0).clip(lower=1e-4)
    return float(np.sum((c - r) * np.log(c / r)))


def status(v: float) -> str:
    return "Drift" if v >= 0.25 else "Warning" if v >= 0.10 else "Stable"


def invoice_level(df: pd.DataFrame) -> pd.DataFrame:
    s = df[~df["Return_Flag"]]
    return s.groupby("InvoiceNo").agg(Basket_Value=("Net_Revenue", "sum"), Basket_Size=("Quantity", "sum"),
                                      Avg_Price=("UnitPrice", "mean"), Discount_Pct=("Discount_Pct", "mean"),
                                      Satisfaction=("Satisfaction", "mean"))


NUM_COLS = ["Basket_Value", "Basket_Size", "Avg_Price", "Discount_Pct", "Satisfaction"]
CAT_COLS = ["Category", "Channel", "Region", "Payment_Method"]


def drift_report(ref: pd.DataFrame, cur: pd.DataFrame) -> pd.DataFrame:
    """ref / cur are line-level frames; numeric drift is measured at invoice level."""
    ri, ci = invoice_level(ref), invoice_level(cur)
    rows = []
    for c in NUM_COLS:
        p = psi(ri[c], ci[c])
        rows.append({"Feature": c, "Type": "Numeric", "PSI": p, "KS p-value": float(ks_2samp(ri[c], ci[c]).pvalue),
                     "Ref mean": ri[c].mean(), "Current mean": ci[c].mean(), "Status": status(p)})
    for c in CAT_COLS:
        p = psi_cat(ref[c], cur[c])
        rows.append({"Feature": c, "Type": "Categorical", "PSI": p, "KS p-value": np.nan, "Ref mean": np.nan,
                     "Current mean": np.nan, "Status": status(p)})
    return pd.DataFrame(rows)


def simulate_drift(cur: pd.DataFrame, price_shift: float = 0.0, qty_shift: float = 0.0, extra_discount: float = 0.0,
                   mobile_shift: float = 0.0) -> pd.DataFrame:
    """Inject artificial drift into the current window (for demos)."""
    d = cur.copy()
    sale = ~d["Return_Flag"]
    d.loc[sale, "UnitPrice"] = d.loc[sale, "UnitPrice"] * (1 + price_shift)
    d.loc[sale, "Quantity"] = np.maximum((d.loc[sale, "Quantity"] * (1 + qty_shift)).round(), 1)
    d.loc[sale, "Discount_Pct"] = np.clip(d.loc[sale, "Discount_Pct"] + extra_discount, 0, 60)
    d["Net_Revenue"] = d["Quantity"] * d["UnitPrice"] * (1 - d["Discount_Pct"] / 100)
    if mobile_shift > 0:
        flip = np.random.default_rng(1).random(len(d)) < mobile_shift
        d.loc[flip, "Channel"] = "Mobile App"
    return d
