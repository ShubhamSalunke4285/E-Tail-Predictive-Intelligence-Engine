# ShopSphere: Predictive E-Commerce Platform

An Amazon-style online store with a built-in **predictive intelligence layer**.
Shoppers browse and search a real product catalog and get personalized
recommendations that learn from their behaviour; an admin console forecasts
sales, drives inventory reorder decisions and flags slow-moving / aging stock
for clearance.

Built on the real **[Online Retail II](https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci)**
dataset (1M+ transactions, 4,500+ products, 5,800+ customers).

---

## The two views

### 🛍️ Shopper storefront (`/`)
- Browse 4,500+ products across 10 categories, search, sort, product pages, cart & checkout.
- **"Because you browsed"**: fresh recommendations built in real time from your
  clicks via an item-similarity map (no retrain needed).
- **"Recommended for you"**: item-based **collaborative filtering** on your history.
- Every view / search / add-to-cart / purchase is written to the **ETL
  clickstream** (`interaction_logs`), the data that powers the models.

### 📊 Admin console (`/admin`)
- **SARIMA sales forecasts** (weekly-seasonal, overall + per category) with 95% confidence bands → demand planning.
- **Inventory reorder**: fast sellers projected to run out, with suggested reorder quantities.
- **Slow-mover / clearance engine**: stock sitting unsold past its category
  *shelf life* (perishability) gets a suggested discount to move it.
- KPIs, revenue-by-category, and a **"Retrain models"** button that reruns the
  pipeline on the latest clickstream so recommendations refresh on demand.

---

## How the pieces fit

```
  shopper clicks                ETL                     models                    views
 ┌──────────────┐      ┌────────────────────┐   ┌────────────────────┐   ┌──────────────┐
 │ view/search/ │─────▶│ interaction_logs   │──▶│ collaborative      │──▶│ storefront   │
 │ cart/buy     │      │ (clickstream)      │   │  filtering (CF)    │   │ recommends   │
 └──────────────┘      ├────────────────────┤   ├────────────────────┤   ├──────────────┤
                       │ sales_daily(_cat)  │──▶│ SARIMA forecasting │──▶│ admin: demand│
                       ├────────────────────┤   ├────────────────────┤   │ + reorder    │
                       │ products + stock   │──▶│ inventory / slow-  │──▶│ + clearance  │
                       └────────────────────┘   │  mover engine      │   └──────────────┘
                                                └────────────────────┘
            orchestrated by Apache Airflow (daily) or the admin "Retrain" button
```

---

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate                 # Windows
pip install -r requirements.txt

python run.py pipeline                  # seeds the DB + trains all models (~30s)
python run.py web                       # http://127.0.0.1:5000
```

> **Data:** put `online_retail_II.csv` (or the downloaded `archive.zip`) from
> [Kaggle](https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci) into
> `data/raw/`. The loader auto-detects and unzips it on first run.

Open the store at **http://127.0.0.1:5000/** and the admin at
**http://127.0.0.1:5000/admin**. Sign in as any sample customer to see
personalized recommendations.

### See the live loop
1. Sign in, open a few products in one category (e.g. browse *Seasonal*).
2. The **"Because you browsed"** shelf updates immediately from your clicks.
3. Open **/admin → Retrain models**: collaborative filtering reruns on your new
   clickstream and the **"Recommended for you"** shelf refreshes.

---

## The models

| Model | Library | What it does |
|---|---|---|
| **Item-based collaborative filtering** | scikit-learn | user×item matrix → cosine item similarity → per-user top-N + "also bought" |
| **SARIMA (2,1,1)(1,1,1,7)** | statsmodels | weekly-seasonal daily revenue forecast, overall + per category, 30-day horizon + 95% CI |
| **Inventory / slow-mover engine** | pandas/numpy | sales velocity vs stock → reorder qty; days-unsold vs category shelf life → clearance discount |

---

## Pipeline & orchestration

`python run.py pipeline` runs four stages, and each is an importable function the
Airflow DAG calls as a task:

```
etl_recommendations ─┐
forecast_sales ──────┼──► build_kpis
inventory_engine ────┘
```

The DAG ([airflow/dags/etail_pipeline_dag.py](airflow/dags/etail_pipeline_dag.py))
schedules a daily retrain at 02:00. Airflow isn't required to run the project;
the same stages run via the CLI and the admin "Retrain" button.

---

## Project layout

```
etail-predictive-intelligence/
├── config.py                  # paths, categories, shelf-life, model params
├── run.py                     # CLI: setup | pipeline | web | all
├── data/                      # raw Kaggle file + app.db (generated)
├── airflow/dags/              # Airflow orchestration DAG
└── src/
    ├── data_loading/online_retail.py   # load + clean the Kaggle dataset
    ├── catalog.py             # category mapping + inventory synthesis
    ├── setup_database.py      # one-time seed of the application DB
    ├── db.py                  # SQLite access layer
    ├── pipeline.py            # ETL + model retrain stages
    ├── models/                # collaborative filtering · SARIMA · inventory
    └── app/                   # Flask storefront + admin (templates/)
```

---

## Notes on the data
- **Categories** are derived from product descriptions (the sales export has none).
- **Stock levels & product age** are synthesized deterministically (operational
  state a pure sales export doesn't contain) so the reorder / perishability
  engine has realistic cases to act on.
- The 2009-2011 dataset timeline is **shifted forward to "today"** so live clicks,
  the forecast horizon and stock-recency math line up with the real calendar.

## Tech stack
**Python · Flask · pandas · NumPy · scikit-learn · statsmodels · SQLAlchemy ·
SQLite · Chart.js · Apache Airflow**
