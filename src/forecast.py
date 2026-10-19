
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


def daily_series(df: pd.DataFrame, metric: str = "Net_Revenue", category: str | None = None) -> pd.Series:
    s = df[~df["Return_Flag"]]
    if category and category != "All":
        s = s[s["Category"] == category]
    ser = s.groupby(s["InvoiceDate"].dt.normalize())[metric].sum()
    idx = pd.date_range(ser.index.min(), ser.index.max(), freq="D")
    return ser.reindex(idx, fill_value=0.0).astype(float)


def mape(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    m = y > 0
    return float(np.mean(np.abs((y[m] - p[m]) / y[m])) * 100) if m.any() else float("nan")


def wape(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.abs(y - p).sum() / max(y.sum(), 1e-9) * 100)



def rmse(y, p) -> float:
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(p, float)) ** 2)))


def f_naive(train: pd.Series, h: int) -> np.ndarray:
    return np.resize(train.values[-7:], h)


def f_holt_winters(train: pd.Series, h: int) -> np.ndarray:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    m = ExponentialSmoothing(train.clip(lower=0), trend="add", damped_trend=True, seasonal="add",
                             seasonal_periods=7, initialization_method="estimated").fit()
    return np.clip(m.forecast(h).values, 0, None)


def _lag_forecast(train: pd.Series, h: int, make_model) -> np.ndarray:
    df = pd.DataFrame({"y": train})
    for l in (1, 7, 14, 28):
        df[f"lag{l}"] = df["y"].shift(l)
    df["rm7"] = df["y"].shift(1).rolling(7).mean()
    df["rm28"] = df["y"].shift(1).rolling(28).mean()
    df["dow"], df["month"], df["doy"], df["t"] = df.index.dayofweek, df.index.month, df.index.dayofyear, np.arange(len(df))
    df = df.dropna()
    X, y = df.drop(columns="y"), df["y"]
    model = make_model().fit(X, y)
    hist, out = list(train.values), []
    for d in pd.date_range(train.index[-1] + pd.Timedelta(days=1), periods=h):
        row = {"lag1": hist[-1], "lag7": hist[-7], "lag14": hist[-14], "lag28": hist[-28], "rm7": np.mean(hist[-7:]),
               "rm28": np.mean(hist[-28:]), "dow": d.dayofweek, "month": d.month, "doy": d.dayofyear, "t": len(hist)}
        p = max(0.0, float(model.predict(pd.DataFrame([row])[X.columns])[0]))
        hist.append(p)
        out.append(p)
    return np.array(out)


def f_gbm(train, h):
    return _lag_forecast(train, h, lambda: GradientBoostingRegressor(n_estimators=250, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=42))


def f_mlp(train, h):
    return _lag_forecast(train, h, lambda: TransformedTargetRegressor(
        regressor=make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=600, random_state=42)),
        transformer=StandardScaler()))


def f_prophet(train, h):
    from prophet import Prophet
    m = Prophet(weekly_seasonality=True, yearly_seasonality=len(train) > 365, daily_seasonality=False)
    m.fit(pd.DataFrame({"ds": train.index, "y": train.values}))
    fc = m.predict(pd.DataFrame({"ds": pd.date_range(train.index[-1] + pd.Timedelta(days=1), periods=h)}))
    return np.clip(fc["yhat"].values, 0, None)


def available_models() -> dict:
    models = {"Seasonal Naive": f_naive, "Holt-Winters": f_holt_winters, "Gradient Boosting": f_gbm, "Neural Net (MLP)": f_mlp}
    try:
        import prophet  # noqa: F401
        models["Prophet"] = f_prophet
    except Exception:
        pass
    return models


def run_forecast(series: pd.Series, horizon: int = 30, holdout: int = 30) -> dict:
    if len(series) < holdout + 90:
        raise ValueError("Not enough history for this selection (need at least 120 days).")
    models = available_models()
    train, test = series.iloc[:-holdout], series.iloc[-holdout:]
    val, rows, fut = {}, [], {}
    for name, fn in models.items():
        try:
            val[name] = fn(train, holdout)
            fut[name] = fn(series, horizon)
        except Exception:
            val.pop(name, None)
            fut.pop(name, None)
    ranked = sorted((n for n in val if n != "Seasonal Naive"), key=lambda n: mape(test.values, val[n]))
    cand = ranked[:2] or list(val)  # hybrid = inverse-error weighted blend of the two best models
    w = np.array([1 / max(mape(test.values, val[n]), 1e-6) for n in cand])
    w = w / w.sum()
    val["Hybrid"] = sum(wi * val[n] for wi, n in zip(w, cand))
    fut["Hybrid"] = sum(wi * fut[n] for wi, n in zip(w, cand))
    for n, p in val.items():
        rows.append({"Model": n, "MAPE %": mape(test.values, p), "WAPE %": wape(test.values, p), "RMSE": rmse(test.values, p)})
    metrics = pd.DataFrame(rows).sort_values("MAPE %").reset_index(drop=True)
    return {"series": series, "test": test, "val": val, "future": fut,
            "future_dates": pd.date_range(series.index[-1] + pd.Timedelta(days=1), periods=horizon),
            "metrics": metrics, "weights": dict(zip(cand, w)), "best": metrics.iloc[0]["Model"],
            "resid_std": {n: float(np.std(test.values - p)) for n, p in val.items()}}
