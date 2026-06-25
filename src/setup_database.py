"""One-time setup: seed the live application DB from Online Retail II.

Builds the storefront catalog, the customer base, the historical clickstream
(``interaction_logs``) and the daily sales series that the ARIMA forecaster and
inventory engine consume. Run once via ``python run.py setup`` (the pipeline
calls it automatically if the DB is missing).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
from src import catalog, db
from src.data_loading import online_retail


_FIRST = ["Olivia", "Liam", "Emma", "Noah", "Ava", "Aarav", "Priya", "Wei",
          "Sofia", "Mateo", "Yuki", "Ravi", "Chloe", "Diego", "Aisha", "Sam",
          "Mia", "Lucas", "Nina", "Omar", "Zara", "Leo", "Hana", "Ibrahim"]
_LAST = ["Smith", "Johnson", "Patel", "Chen", "Garcia", "Kim", "Mueller",
         "Rossi", "Silva", "Khan", "Nguyen", "Ali", "Brown", "Lopez", "Singh"]


def _display_name(uid: int) -> str:
    rng = np.random.default_rng(uid)
    return f"{_FIRST[rng.integers(len(_FIRST))]} {_LAST[rng.integers(len(_LAST))]}"


def build() -> None:
    print("Seeding application database from Online Retail II...")
    raw = online_retail.load_raw()
    clean = online_retail.clean(raw)
    clean = clean.dropna(subset=["customer_id"]).copy()
    clean["customer_id"] = clean["customer_id"].astype(int)

    # Anchor the (2009-2011) dataset to "today" so the storefront's live clicks,
    # the ARIMA forecast horizon and stock-recency math are all coherent with
    # the real calendar instead of being stuck 15 years in the past.
    shift = pd.Timestamp.today().normalize() - clean["invoice_date"].max().normalize()
    clean["invoice_date"] = clean["invoice_date"] + shift
    as_of = clean["invoice_date"].max().normalize()
    print(f"  shifted timeline forward by {shift.days} days -> anchored at {as_of.date()}")

    # ---- products / catalog -------------------------------------------------
    g = clean.groupby("product_id")
    products = g.agg(
        stock_code=("stock_code", "first"),
        name=("description", "first"),
        price=("price", "mean"),
        units=("quantity", "sum"),
        orders=("invoice", "nunique"),
        revenue=("revenue", "sum"),
        unique_customers=("customer_id", "nunique"),
        first_seen=("invoice_date", "min"),
        last_seen=("invoice_date", "max"),
    ).reset_index()
    products["name"] = products["name"].fillna("Unknown").str.title().str.slice(0, 60)
    products["price"] = products["price"].round(2)
    products["revenue"] = products["revenue"].round(2)
    products["category"] = products["name"].map(catalog.categorize)
    products["shelf_life_days"] = products["category"].map(catalog.shelf_life_days)
    products = catalog.synthesize_inventory(products, as_of=as_of)

    # keep a sensible storefront: drop ultra-rare junk SKUs
    products = products[products["orders"] >= 2].reset_index(drop=True)
    keep_ids = set(products["product_id"])
    clean = clean[clean["product_id"].isin(keep_ids)]

    # ---- customers ----------------------------------------------------------
    cust_ids = np.sort(clean["customer_id"].unique())
    customers = pd.DataFrame({
        "user_id": cust_ids,
        "name": [_display_name(int(u)) for u in cust_ids],
    })

    # ---- historical clickstream (seeded as purchases) -----------------------
    logs = pd.DataFrame({
        "user_id": clean["customer_id"].to_numpy(),
        "product_id": clean["product_id"].to_numpy(),
        "event_type": "purchase",
        "qty": clean["quantity"].to_numpy(),
        "timestamp": clean["invoice_date"].to_numpy(),
        "source": "seed",
    })

    # ---- daily sales (overall + per category) for ARIMA ---------------------
    cat_map = products.set_index("product_id")["category"]
    clean = clean.assign(category=clean["product_id"].map(cat_map))
    daily = (clean.set_index("invoice_date")["revenue"]
             .resample("D").sum().reset_index())
    daily.columns = ["date", "revenue"]
    daily = daily[daily["revenue"] > 0].reset_index(drop=True)

    daily_cat = (clean.groupby([pd.Grouper(key="invoice_date", freq="D"), "category"])
                 ["revenue"].sum().reset_index())
    daily_cat.columns = ["date", "category", "revenue"]
    daily_cat = daily_cat[daily_cat["revenue"] > 0].reset_index(drop=True)

    # ---- persist ------------------------------------------------------------
    db.write_table(products, "products")
    db.write_table(customers, "customers")
    db.write_table(logs, "interaction_logs")
    db.write_table(daily, "sales_daily")
    db.write_table(daily_cat, "sales_daily_category")
    db.execute("CREATE INDEX IF NOT EXISTS ix_logs_user "
               "ON interaction_logs(user_id)")
    db.execute("CREATE INDEX IF NOT EXISTS ix_logs_product "
               "ON interaction_logs(product_id)")

    print(f"  products       : {len(products):>9,}")
    print(f"  customers      : {len(customers):>9,}")
    print(f"  interaction_logs:{len(logs):>9,}  (seeded purchases)")
    print(f"  sales_daily    : {len(daily):>9,} days")
    print(f"  categories     : {products['category'].nunique()}  "
          f"({', '.join(sorted(products['category'].unique()))})")
    print(f"  as-of date     : {as_of.date()}")
    print("Database seeded ->", config.APP_DB_PATH)


if __name__ == "__main__":
    build()
