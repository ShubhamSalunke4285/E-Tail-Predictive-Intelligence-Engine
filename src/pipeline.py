"""End-to-end model pipeline for the e-commerce platform.

This is the "retrain" job. It reads the live clickstream (interaction_logs) and
sales tables from the application DB and refreshes three model outputs:

    ETL clickstream ─► collaborative filtering ─► recommendations
    daily sales      ─► ARIMA (overall + per category) ─► forecasts
    stock + sales    ─► inventory / slow-mover engine  ─► inventory

Run it with `python run.py pipeline`, on a schedule via the Airflow DAG, or
on-demand from the admin dashboard's "Retrain models" button. Each stage is an
argument-free function so Airflow can call them as independent tasks.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

import config
from src import db
from src.models import collaborative_filtering as cf
from src.models import forecasting
from src.models import inventory


def _ensure_seeded() -> None:
    if not db.table_exists("products"):
        from src import setup_database
        setup_database.build()


# ----------------------------------------------------------------------------
# Stage 1 - ETL clickstream -> collaborative-filtering recommendations
# ----------------------------------------------------------------------------
def _logs_to_ratings(logs: pd.DataFrame) -> pd.DataFrame:
    """Collapse the raw event stream into an implicit user x item rating."""
    logs = logs.copy()
    logs["w"] = logs["event_type"].map(config.EVENT_WEIGHTS).fillna(1.0)
    # purchases scale with quantity; other events count once
    qty = logs["qty"].where(logs["event_type"] == "purchase", 1).fillna(1)
    logs["signal"] = logs["w"] * np.clip(qty, 1, None)
    ratings = (logs.groupby(["user_id", "product_id"])["signal"]
               .sum().reset_index(name="rating"))
    ratings["rating"] = np.log1p(ratings["rating"])
    return ratings


def stage_etl_recommendations() -> None:
    print("[1/4] ETL clickstream -> collaborative filtering")
    logs = db.read_table("interaction_logs")
    print(f"  {len(logs):,} interaction logs")
    ratings = _logs_to_ratings(logs)
    recs, neighbors = cf.train_and_score(ratings)
    db.write_table(recs, "recommendations")
    db.write_table(neighbors, "item_similar")
    db.execute("CREATE INDEX IF NOT EXISTS ix_recs_user "
               "ON recommendations(user_id)")
    db.execute("CREATE INDEX IF NOT EXISTS ix_sim_product "
               "ON item_similar(product_id)")


# ----------------------------------------------------------------------------
# Stage 2 - ARIMA sales forecasting (overall + per category)
# ----------------------------------------------------------------------------
def stage_forecast_sales() -> None:
    print("[2/4] ARIMA sales forecasting")
    overall = db.read_table("sales_daily")
    overall["date"] = pd.to_datetime(overall["date"])
    frames = [forecasting.forecast_scope(overall, "overall", "All categories")]

    by_cat = db.read_table("sales_daily_category")
    by_cat["date"] = pd.to_datetime(by_cat["date"])
    top_cats = (by_cat.groupby("category")["revenue"].sum()
                .sort_values(ascending=False).head(6).index)
    for cat in top_cats:
        sub = by_cat[by_cat["category"] == cat][["date", "revenue"]]
        frames.append(forecasting.forecast_scope(sub, "category", cat))

    db.write_table(pd.concat(frames, ignore_index=True), "forecasts")
    print(f"  forecast overall + {len(top_cats)} categories")


# ----------------------------------------------------------------------------
# Stage 3 - inventory / slow-mover engine
# ----------------------------------------------------------------------------
def stage_inventory() -> None:
    print("[3/4] Inventory & slow-mover engine")
    products = db.read_table("products")
    purchases = db.query(
        "SELECT product_id, qty, timestamp FROM interaction_logs "
        "WHERE event_type = 'purchase'"
    )
    as_of = pd.to_datetime(purchases["timestamp"], format="mixed").max()
    actions = inventory.build_inventory_actions(products, purchases, as_of)
    db.write_table(actions, "inventory")
    counts = actions["action"].value_counts().to_dict()
    print(f"  {counts}")


# ----------------------------------------------------------------------------
# Stage 4 - KPI summary
# ----------------------------------------------------------------------------
def stage_build_kpis() -> None:
    print("[4/4] KPI summary")
    inv = db.read_table("inventory")
    sales = db.read_table("sales_daily")
    n_logs = db.query("SELECT COUNT(*) n FROM interaction_logs")["n"].iloc[0]
    n_users = db.query("SELECT COUNT(*) n FROM customers")["n"].iloc[0]
    fc = db.read_table("forecasts")
    fc_next = fc[(fc["scope"] == "overall") & (fc["kind"] == "forecast")]["value"].sum()

    kpis = pd.DataFrame([{
        "total_revenue": round(float(sales["revenue"].sum()), 2),
        "total_products": int(len(inv)),
        "total_customers": int(n_users),
        "total_interactions": int(n_logs),
        "forecast_revenue_30d": round(float(fc_next), 2),
        "restock_items": int((inv["action"] == "restock").sum()),
        "clearance_items": int((inv["action"] == "clearance").sum()),
        "inventory_units": int(inv["stock_qty"].sum()),
        "generated_at": pd.Timestamp.utcnow().isoformat(timespec="seconds"),
    }])
    db.write_table(kpis, "kpis")


def run_pipeline() -> None:
    t0 = time.time()
    print("=" * 60)
    print(" E-COMMERCE PREDICTIVE PIPELINE  -  model retrain")
    print("=" * 60)
    _ensure_seeded()
    stage_etl_recommendations()
    stage_forecast_sales()
    stage_inventory()
    stage_build_kpis()
    print("=" * 60)
    print(f" Pipeline complete in {time.time() - t0:.1f}s")
    print(" Launch the store with:  python run.py web")
    print("=" * 60)


if __name__ == "__main__":
    run_pipeline()
