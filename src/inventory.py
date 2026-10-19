"""Inventory optimisation: safety stock, reorder point, EOQ, ABC classification."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.data import _map_u


def product_table(df: pd.DataFrame, lookback: int = 90) -> pd.DataFrame:
    s = df[~df["Return_Flag"]]
    end = s["InvoiceDate"].max().normalize()
    start = end - pd.Timedelta(days=lookback - 1)
    r = s[s["InvoiceDate"] >= start]
    daily = (r.groupby(["StockCode", r["InvoiceDate"].dt.normalize()])["Quantity"].sum().unstack(fill_value=0)
             .reindex(columns=pd.date_range(start, end), fill_value=0))
    meta = s.groupby("StockCode").agg(Description=("Description", "first"), Category=("Category", "first"),
                                      UnitPrice=("UnitPrice", "median"), UnitCost=("UnitCost", "median"),
                                      Lead_Time_Days=("Lead_Time_Days", "first"), Supplier=("Supplier", "first"),
                                      Revenue=("Net_Revenue", "sum"))
    t = meta.join(pd.DataFrame({"AvgDailyDemand": daily.mean(axis=1), "StdDailyDemand": daily.std(axis=1).fillna(0)}), how="inner")
    t = t[t["AvgDailyDemand"] > 0].copy()
    # No inventory table exists in the public dataset -> simulated stock on hand (deterministic)
    u = _map_u(pd.Series(t.index, index=t.index), "stock")
    t["Stock_On_Hand"] = np.round(t["AvgDailyDemand"] * (2 + u * 75)).astype(int)
    return t.reset_index()


def compute_policy(t: pd.DataFrame, service_level=0.95, order_cost=25.0, holding_rate=0.2, demand_mult=1.0) -> pd.DataFrame:
    t = t.copy()
    z = norm.ppf(service_level)
    d = t["AvgDailyDemand"] * demand_mult
    sigma = t["StdDailyDemand"] * demand_mult
    L = t["Lead_Time_Days"]
    t["Safety_Stock"] = np.ceil(z * sigma * np.sqrt(L))
    t["Reorder_Point"] = np.ceil(d * L + t["Safety_Stock"])
    t["EOQ"] = np.ceil(np.sqrt(2 * d * 365 * order_cost / (holding_rate * t["UnitCost"].clip(lower=0.01))))
    t["Days_Of_Cover"] = (t["Stock_On_Hand"] / d).round(1)
    crit = t["Stock_On_Hand"] < d * L
    reorder = t["Stock_On_Hand"] <= t["Reorder_Point"]
    over = t["Days_Of_Cover"] > 90
    t["Status"] = np.select([crit, reorder, over], ["Stockout Risk", "Reorder Now", "Overstock"], default="Healthy")
    t["Order_Qty"] = np.where(reorder, np.ceil(np.maximum(t["Reorder_Point"] + t["EOQ"] - t["Stock_On_Hand"], 0)), 0).astype(int)
    target = t["Reorder_Point"] + t["EOQ"]
    t["Excess_Units"] = np.where(over, np.maximum(t["Stock_On_Hand"] - target, 0), 0).astype(int)
    t["Excess_Capital"] = (t["Excess_Units"] * t["UnitCost"]).round(2)
    return t


def abc_xyz(df: pd.DataFrame) -> pd.DataFrame:
    s = df[~df["Return_Flag"]]
    rev = s.groupby("StockCode")["Net_Revenue"].sum().sort_values(ascending=False)
    cum = rev.cumsum() / rev.sum()
    abc = pd.Series(np.where(cum <= 0.80, "A", np.where(cum <= 0.95, "B", "C")), index=rev.index)
    monthly = s.groupby(["StockCode", pd.Grouper(key="InvoiceDate", freq="MS")])["Quantity"].sum().unstack(fill_value=0)
    cv = (monthly.std(axis=1) / monthly.mean(axis=1).replace(0, np.nan)).reindex(rev.index)
    xyz = pd.Series(np.where(cv < 0.5, "X", np.where(cv < 1.0, "Y", "Z")), index=rev.index)
    return pd.DataFrame({"Revenue": rev, "Cum_Share": cum, "ABC": abc, "CV": cv, "XYZ": xyz})
