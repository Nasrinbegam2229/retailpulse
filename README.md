# RetailPulse: AI-Powered Customer Analytics & Demand Forecasting

A 12-page Streamlit platform (dark theme, with sign-up / log-in) that turns retail sales data into regional and profit insight, demand forecasts, customer segments, churn risk, inventory orders and alerts.
Built for the Zidio Data Science & Analytics project (March 2026 edition).

**Live demo:** _paste your Streamlit Cloud URL here_  
**Demo video:** _paste your YouTube / Loom link here_

## Dashboard pages

| Group | Page | What it answers |
|---|---|---|
| Understand | Executive Overview | How is the business doing right now? |
| | Sales Analytics | Trends, seasonality, promotion and holiday effects |
| | Regional Sales | UK vs Europe vs Rest of World: revenue, countries, world map, delivery |
| | Profit & Revenue | Revenue vs profit trend, profit bridge, profitability by segment, discounts and returns |
| | Product Performance | Top products, ABC-XYZ classes, margins |
| Customers | Customer Segmentation | RFM + K-Means segments with a playbook per segment |
| | Churn Prediction | Who is likely to leave, and how much revenue is at stake |
| Plan | Demand Forecasting | 7-60 day forecast with what-if scenarios |
| | Model Comparison | Holt-Winters vs Gradient Boosting vs MLP (vs Prophet) vs Hybrid |
| | Inventory Optimization | Safety stock, reorder points, EOQ, purchase orders |
| Operate | Real-time Analytics | Live order stream: KPIs, revenue per minute, alerts, demand-spike detection, live stock and purchase orders |
| | Alerts & Reports | Stockout, churn and drift alerts, Excel / CSV export |

## Accounts (sign-up and log-in)

The app opens on a **Log in / Sign up** page; no dashboard page is reachable until you are signed in. Accounts are stored in a local SQLite file
(`data/users.db`, git-ignored). Only a salted PBKDF2-SHA256 hash of each password is saved, and an account locks for 5 minutes after 5 wrong passwords.
Sessions last until you log out or refresh the browser tab. There is no password-reset flow: delete the row (or the file) to start again.

## Data: public dataset plus own columns

The app uses the public **UCI / Kaggle "Online Retail"** transactions (8 columns) and extends them with engineered columns in `src/data.py`:
product category (parsed from descriptions), region, channel, payment method, promotion calendar and discounts, holiday and season flags,
unit cost, profit, supplier, lead time, satisfaction rating, customer demographics, acquisition channel and loyalty tier, plus margin %, price band,
basket value, customer type, days since previous order (calculated) and shipping cost, delivery days / status and return reason (simulated).

Engineered columns are generated deterministically (hash-based), so results are reproducible. Columns that do not exist in the public data
(channel, supplier, lead time, satisfaction, demographics, promotions, stock on hand) are **simulated**; they demonstrate the full pipeline
and must not be presented as real business facts.

1. Download the dataset (UCI: *Online Retail*, or Kaggle *Online Retail / Online Retail II*) and place the `.xlsx` or `.csv` in `data/raw/`.
2. Build the enriched dataset: `python -m src.data --build`
3. If no raw file is present, the app falls back to a synthetic dataset with the same schema so it always starts.

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m src.data --build
streamlit run app.py
```

Docker: `docker build -t retailpulse . && docker run -p 8501:8501 retailpulse`

## Deploy (Streamlit Community Cloud)

1. Commit `data/processed/retailpulse_enriched.csv.gz` (built from your raw file) so the app starts in seconds.
2. Push to GitHub, open share.streamlit.io, choose the repo, set the main file to `app.py`.

## Project layout

```
app.py                 login gate, navigation (12 pages) and global filters
views/                 one file per page (login.py is the sign-in page)
src/data.py            load, clean, enrich, quality checks
src/realtime.py        live order stream engine (replay, alerts, stock depletion, spike detection)
src/auth.py            sign-up / log-in (SQLite, hashed passwords, lockout)
src/features.py        RFM, K-Means segmentation, churn model
src/forecast.py        forecasting models and hybrid blend
src/inventory.py       safety stock, ROP, EOQ, ABC-XYZ
src/drift.py           PSI, KS, drift simulator
src/tracking.py        experiment log and model registry (MLflow optional)
airflow/dags/          weekly retraining DAG
.streamlit/config.toml dark theme
tests/                 pipeline tests and a smoke test for every page
```

## Method notes

- **Churn:** features are computed at a cutoff date and the label is "no purchase in the following N days". Recency is measured before the label window, so there is no target leakage.
- **Forecast metrics:** reported on a 30-day holdout. MAPE skips zero-sales days; WAPE is also shown because it is more stable for daily data.
- **Hybrid forecast:** inverse-error weighted blend of the two best non-baseline models.
- **Inventory:** safety stock = z x sigma x sqrt(lead time); reorder point = demand x lead time + safety stock; EOQ from annual demand, order cost and holding cost.
- **Drift:** PSI below 0.10 stable, 0.10-0.25 warning, 0.25+ drift. A drift alert recommends retraining.

## Tests

`pytest -q` runs 21 checks, including a smoke test that loads every dashboard page.
