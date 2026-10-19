"""Weekly retraining DAG (reference implementation; deploy to an Airflow instance to run it)."""
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import BranchPythonOperator, PythonOperator

import pandas as pd


def ingest_clean():
    from src.data import build_dataset
    build_dataset(force=True)


def check_drift():
    from src.data import build_dataset
    from src.drift import drift_report
    df, _ = build_dataset()
    cut = df["InvoiceDate"].max() - pd.Timedelta(days=60)
    rep = drift_report(df[df["InvoiceDate"] < cut], df[df["InvoiceDate"] >= cut])
    return "train_models" if (rep["PSI"] >= 0.25).any() or datetime.today().weekday() == 6 else "skip_training"


def train_models():
    from src import tracking
    from src.data import build_dataset
    from src.features import train_churn
    from src.forecast import daily_series, run_forecast
    df, _ = build_dataset()
    ch = train_churn(df, 90)
    tracking.log_run("churn", "GradientBoostingClassifier", {"horizon_days": 90}, ch["metrics"])
    fc = run_forecast(daily_series(df, "Net_Revenue"), 30)
    for _, r in fc["metrics"].iterrows():
        tracking.log_run("forecast", r["Model"], {"horizon_days": 30}, {"MAPE": r["MAPE %"], "RMSE": r["RMSE"]})


with DAG("retailpulse_weekly", start_date=datetime(2026, 5, 1), schedule="0 2 * * 0", catchup=False,
         default_args={"retries": 1, "retry_delay": timedelta(minutes=5)}) as dag:
    ingest = PythonOperator(task_id="ingest_clean", python_callable=ingest_clean)
    branch = BranchPythonOperator(task_id="check_drift", python_callable=check_drift)
    train = PythonOperator(task_id="train_models", python_callable=train_models)
    skip = PythonOperator(task_id="skip_training", python_callable=lambda: None)
    ingest >> branch >> [train, skip]
