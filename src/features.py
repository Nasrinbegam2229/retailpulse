"""Customer features, RFM segmentation and churn model."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score, precision_score, roc_auc_score, roc_curve, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEGMENT_NAMES = ["Champions", "Loyal Customers", "Potential Loyalists", "Promising",
                 "Needs Attention", "At Risk", "Hibernating", "Lost"]
CHURN_FEATURES = ["Recency", "Frequency", "Monetary", "AvgOrderValue", "Tenure", "PurchaseRate",
                  "ReturnRate", "Categories", "PromoShare", "Satisfaction", "OnlineShare"]


def customer_features(df: pd.DataFrame, snapshot: pd.Timestamp) -> pd.DataFrame:
    """One row per customer using ONLY data up to `snapshot` (no look-ahead)."""
    d = df[(df["InvoiceDate"] <= snapshot) & df["CustomerID"].notna()]
    s = d[~d["Return_Flag"]].assign(is_online=lambda x: (x["Channel"] == "Online").astype(float))
    g = s.groupby("CustomerID")
    f = g.agg(last=("InvoiceDate", "max"), first=("InvoiceDate", "min"), Frequency=("InvoiceNo", "nunique"),
              Monetary=("Net_Revenue", "sum"), Items=("Quantity", "sum"), Categories=("Category", "nunique"),
              PromoShare=("Promo_Flag", "mean"), Satisfaction=("Satisfaction", "mean"), OnlineShare=("is_online", "mean"),
              Profit=("Profit", "sum"))
    f["Recency"] = (snapshot - f["last"]).dt.days
    f["Tenure"] = (snapshot - f["first"]).dt.days
    f["AvgOrderValue"] = f["Monetary"] / f["Frequency"]
    f["PurchaseRate"] = f["Frequency"] / (f["Tenure"] + 30) * 30  # orders per month
    rets = d[d["Return_Flag"]].groupby("CustomerID")["InvoiceNo"].nunique().reindex(f.index).fillna(0)
    f["ReturnRate"] = rets / (rets + f["Frequency"])
    attrs = s.drop_duplicates("CustomerID").set_index("CustomerID")[["Country", "Loyalty_Tier", "Age_Group", "Gender", "Acquisition_Channel"]]
    return f.drop(columns=["last", "first"]).join(attrs)


# ------------------------------------------------------------------ segmentation
def _scaled(rfm: pd.DataFrame) -> np.ndarray:
    return StandardScaler().fit_transform(np.log1p(rfm[["Recency", "Frequency", "Monetary"]].clip(lower=0)))


def silhouette_scan(rfm: pd.DataFrame, ks=range(3, 9)) -> pd.DataFrame:
    X = _scaled(rfm)
    idx = np.random.default_rng(0).choice(len(X), min(3000, len(X)), replace=False)
    rows = []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=5, random_state=42).fit(X)
        rows.append((k, silhouette_score(X[idx], km.labels_[idx]), km.inertia_))
    return pd.DataFrame(rows, columns=["k", "Silhouette", "Inertia"])


def segment_customers(rfm: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(_scaled(rfm))
    out = rfm.copy()
    out["Cluster"] = km.labels_
    prof = out.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean()
    score = prof["Frequency"].rank() + prof["Monetary"].rank() + (-prof["Recency"]).rank()
    order = score.sort_values(ascending=False).index.tolist()
    picks = np.linspace(0, len(SEGMENT_NAMES) - 1, k).round().astype(int)
    names = {c: SEGMENT_NAMES[i] for c, i in zip(order, picks)}
    out["Segment"] = out["Cluster"].map(names)
    return out


# ------------------------------------------------------------------ churn
def build_churn_dataset(df: pd.DataFrame, horizon: int = 90):
    """Features at `cutoff`; label = customer does NOT buy in the next `horizon` days (no leakage)."""
    end = df["InvoiceDate"].max().normalize()
    cutoff = end - pd.Timedelta(days=horizon)
    X = customer_features(df, cutoff)
    future = df[(df["InvoiceDate"] > cutoff) & ~df["Return_Flag"] & df["CustomerID"].notna()]["CustomerID"].unique()
    y = (~X.index.isin(future)).astype(int)
    return X, y, cutoff


def train_churn(df: pd.DataFrame, horizon: int = 90, seed: int = 42) -> dict:
    X, y, cutoff = build_churn_dataset(df, horizon)
    if len(np.unique(y)) < 2:
        raise ValueError("Churn label has a single class; check the horizon / date range.")
    Xtr, Xte, ytr, yte = train_test_split(X[CHURN_FEATURES], y, test_size=0.25, stratify=y, random_state=seed)
    model = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=seed)
    model.fit(Xtr, ytr)
    proba = model.predict_proba(Xte)[:, 1]
    k = max(1, int(0.2 * len(yte)))
    top = np.argsort(-proba)[:k]
    fpr, tpr, _ = roc_curve(yte, proba)
    imp = permutation_importance(model, Xte, yte, scoring="roc_auc", n_repeats=5, random_state=seed)
    importance = pd.DataFrame({"Feature": CHURN_FEATURES, "Importance": imp.importances_mean}).sort_values("Importance", ascending=False)
    metrics = {"AUC-ROC": float(roc_auc_score(yte, proba)), "Precision@Top20%": float(yte[top].mean()),
               "F1": float(f1_score(yte, proba >= 0.5)), "Precision": float(precision_score(yte, proba >= 0.5, zero_division=0)),
               "Churn rate": float(np.mean(y)), "Train size": int(len(Xtr)), "Test size": int(len(Xte))}
    return {"model": model, "metrics": metrics, "importance": importance, "roc": pd.DataFrame({"FPR": fpr, "TPR": tpr}),
            "cutoff": cutoff, "horizon": horizon, "X_test": Xte}


def score_customers(model, feats: pd.DataFrame) -> pd.DataFrame:
    out = feats.copy()
    out["Churn_Prob"] = model.predict_proba(out[CHURN_FEATURES])[:, 1]
    out["Risk"] = pd.cut(out["Churn_Prob"], [-0.01, 0.4, 0.7, 1.01], labels=["Low", "Medium", "High"])
    out["Revenue_At_Risk"] = out["Monetary"] * out["Churn_Prob"]
    return out
