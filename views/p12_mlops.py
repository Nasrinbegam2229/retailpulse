import pandas as pd
import streamlit as st

from src import tracking
from src.services import churn_bundle, forecast_bundle
from src.ui import header

header("MLOps & Architecture", "Experiment tracking, model registry, retraining rules and the end-to-end pipeline.")

t1, t2, t3, t4 = st.tabs(["Architecture", "Experiments", "Model registry", "Retraining & CI/CD"])
with t1:
    st.graphviz_chart("""
    digraph G { rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#EEF2F1", color="#0E7C7B", fontname="Helvetica"];
      raw [label="Raw sales data\\n(UCI / Kaggle)"]; enr [label="Cleaning +\\nown columns"]; fe [label="Feature\\nengineering"];
      seg [label="Segmentation\\n(RFM + K-Means)"]; ch [label="Churn model"]; fc [label="Forecasting\\n(Holt-Winters, GBM, MLP, Prophet, Hybrid)"];
      inv [label="Inventory\\noptimisation"]; dash [label="Streamlit dashboard\\n(12 pages)", fillcolor="#0E7C7B", fontcolor="white"];
      mlf [label="Experiment tracking\\n+ registry"]; drift [label="Drift monitoring\\n(PSI, KS)"]; retrain [label="Retraining\\n(Airflow DAG)"];
      raw -> enr -> fe; fe -> seg; fe -> ch; enr -> fc -> inv; seg -> dash; ch -> dash; inv -> dash;
      ch -> mlf; fc -> mlf; enr -> drift -> retrain -> mlf; mlf -> dash; }
    """)
    st.markdown("Each box is a module in `src/`. The Streamlit pages only call those modules, so the same code runs in notebooks, in Airflow and in the app.")
with t2:
    if st.button("Train models and log experiment runs", type="primary"):
        with st.spinner("Training churn and forecasting models..."):
            ch = churn_bundle(90)
            tracking.log_run("churn", "GradientBoostingClassifier", {"horizon_days": 90, "n_estimators": 200, "max_depth": 3}, ch["metrics"])
            fc = forecast_bundle("Net_Revenue", "All", 30)
            for _, r in fc["metrics"].iterrows():
                tracking.log_run("forecast", r["Model"], {"series": "All / Net_Revenue", "horizon_days": 30}, {"MAPE": r["MAPE %"], "WAPE": r["WAPE %"], "RMSE": r["RMSE"]})
        st.success("Logged. Each run records parameters and metrics to artifacts/runs.json (and to MLflow when USE_MLFLOW=1).")
    runs = tracking.load_runs()
    if runs:
        flat = pd.DataFrame([{"run_id": r["run_id"], "time": r["time"], "experiment": r["experiment"], "model": r["model"], **r["metrics"]} for r in runs])
        st.dataframe(flat.sort_values("time", ascending=False), hide_index=True, width="stretch")
    else:
        st.info("No runs yet. Select **Train models and log experiment runs** to create the first entries.")
with t3:
    runs = [r for r in tracking.load_runs() if r["experiment"] == "churn"]
    if not runs:
        st.info("Log a churn run first, then promote it here.")
    else:
        pick = st.selectbox("Run", [f"{r['run_id']}  |  {r['time']}  |  AUC {r['metrics'].get('AUC-ROC', 0):.3f}" for r in runs])
        stage = st.radio("Stage", ["Staging", "Production"], horizontal=True)
        if st.button("Register churn model"):
            tracking.register(pick.split()[0], "churn-model", stage)
            st.rerun()
    reg = tracking.load_registry()
    if reg:
        st.dataframe(pd.DataFrame(reg).sort_values("time", ascending=False), hide_index=True, width="stretch")
with t4:
    st.markdown("""
**Retraining policy**

| Trigger | Threshold | Action |
|---|---|---|
| Data drift | PSI of 0.25 or more on any feature | Retrain churn and forecast models |
| Forecast quality | Rolling MAPE above 12% | Re-tune and retrain forecast |
| Churn quality | AUC-ROC below 0.80 on the latest cohort | Retrain churn model |
| Schedule | Every Sunday 02:00 | Refresh features and models |

**Pipeline files in this repository**
- `airflow/dags/retailpulse_dag.py` runs ingest, validate, train, evaluate and register on a weekly schedule.
- `.github/workflows/ci.yml` lints and tests every push.
- `Dockerfile` builds the app image; the same image is deployed to Streamlit Cloud, Hugging Face Spaces or any container host.
""")
    st.code("""ingest_clean >> build_features >> [train_churn, train_forecast] >> evaluate >> register_model""", language="python")
