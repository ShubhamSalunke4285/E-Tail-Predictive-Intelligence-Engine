"""Ingest and transform the real **Online Retail II** dataset (Kaggle / UCI).

Source: https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci
Columns: Invoice, StockCode, Description, Quantity, InvoiceDate, Price,
         Customer ID, Country  (~1,067,371 rows, Dec 2009 - Dec 2011)

This module locates the downloaded file (CSV or zip), cleans it, and maps it
onto the same canonical warehouse schema the rest of the pipeline expects, so
collaborative filtering, ARIMA forecasting, anomaly detection and the dashboard
all keep working unchanged.

Real-world data wrangling handled here:
  * cancellations / returns  -> Invoice numbers starting with "C" (negative qty)
  * missing Customer ID      -> excluded from per-customer modelling
  * non-positive price/qty    -> dropped
  * postage / fee pseudo-SKUs -> dropped (POST, DOT, M, BANK CHARGES, ...)
  * duplicate rows            -> de-duplicated
"""
from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

import config

# StockCodes that are not real products (shipping, fees, adjustments, samples)
NON_PRODUCT_CODES = {
    "POST", "DOT", "M", "C2", "BANK CHARGES", "BANKCHARGES", "S",
    "AMAZONFEE", "ADJUST", "ADJUST2", "GIFT", "PADS", "B", "CRUK",
}


# ----------------------------------------------------------------------------
# Locating the downloaded file
# ----------------------------------------------------------------------------
def _find_source_file() -> Path:
    """Find the Online Retail CSV (or a zip containing it)."""
    candidates: list[Path] = []
    search_dirs = [config.RAW_DIR, Path.home() / "Downloads"]

    if config.ONLINE_RETAIL_CSV.exists():
        return config.ONLINE_RETAIL_CSV

    for d in search_dirs:
        if not d.exists():
            continue
        candidates += sorted(d.glob("online_retail*II*.csv"))
        candidates += sorted(d.glob("online_retail*.csv"))
        candidates += sorted(d.glob("*online*retail*.zip"))
        candidates += sorted(d.glob("archive*.zip"))

    for c in candidates:
        if c.suffix.lower() == ".csv":
            return c
        if c.suffix.lower() == ".zip":
            return _extract_csv_from_zip(c)

    raise FileNotFoundError(
        "Could not find the Online Retail II dataset.\n"
        "  1. Download it from "
        "https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci\n"
        f"  2. Put 'online_retail_II.csv' (or the downloaded zip) into:\n"
        f"     {config.RAW_DIR}\n"
        "  3. Re-run the pipeline."
    )


def _extract_csv_from_zip(zip_path: Path) -> Path:
    with zipfile.ZipFile(zip_path) as zf:
        inner = next((n for n in zf.namelist() if n.lower().endswith(".csv")), None)
        if inner is None:
            raise FileNotFoundError(f"No CSV found inside {zip_path}")
        target = config.ONLINE_RETAIL_CSV
        with zf.open(inner) as src, open(target, "wb") as dst:
            dst.write(src.read())
        print(f"  extracted {inner} -> {target.name}")
        return target


# ----------------------------------------------------------------------------
# Load + clean
# ----------------------------------------------------------------------------
def load_raw() -> pd.DataFrame:
    path = _find_source_file()
    print(f"  reading {path}")
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="ISO-8859-1")

    # Normalise the two common column spellings to a single schema.
    df = df.rename(columns={
        "Customer ID": "customer_id", "CustomerID": "customer_id",
        "Invoice": "invoice", "InvoiceNo": "invoice",
        "StockCode": "stock_code", "Description": "description",
        "Quantity": "quantity", "InvoiceDate": "invoice_date",
        "Price": "price", "UnitPrice": "price", "Country": "country",
    })
    print(f"  raw rows: {len(df):,}")
    return df


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Return cleaned, purchase-only line items with a revenue column."""
    df = raw.copy()
    n0 = len(df)

    df["invoice"] = df["invoice"].astype(str)
    df["stock_code"] = df["stock_code"].astype(str).str.strip().str.upper()
    df["invoice_date"] = pd.to_datetime(df["invoice_date"], errors="coerce")

    df = df.drop_duplicates()
    df = df.dropna(subset=["invoice_date", "stock_code", "price", "quantity"])

    # Flag cancellations (Invoice starts with C / negative quantity).
    is_cancel = df["invoice"].str.startswith("C") | (df["quantity"] < 0)

    # Keep only valid product purchases for the analytical tables.
    df = df[~is_cancel]
    df = df[(df["quantity"] > 0) & (df["price"] > 0)]
    df = df[~df["stock_code"].isin(NON_PRODUCT_CODES)]
    # genuine product codes are mostly 5-6 chars (e.g. 85123A); drop oddballs
    df = df[df["stock_code"].str.len().between(4, 12)]

    df["revenue"] = (df["quantity"] * df["price"]).round(2)

    # Stable integer product_id from the alphanumeric stock_code.
    codes, _ = pd.factorize(df["stock_code"])
    df["product_id"] = codes + 1

    n_cancel = int(is_cancel.sum())
    print(f"  cleaned: {len(df):,} purchase lines kept "
          f"({n0 - len(df):,} dropped incl. {n_cancel:,} cancellations)")
    return df.reset_index(drop=True)


# ----------------------------------------------------------------------------
# Canonical warehouse tables
# ----------------------------------------------------------------------------
def build_interactions(clean_df: pd.DataFrame) -> pd.DataFrame:
    """One row per purchase line, in the project's fact_interactions schema."""
    out = pd.DataFrame({
        "user_id": clean_df["customer_id"],
        "product_id": clean_df["product_id"],
        "stock_code": clean_df["stock_code"],
        "event_type": "purchase",
        "quantity": clean_df["quantity"],
        "price": clean_df["price"],
        "revenue": clean_df["revenue"],
        "country": clean_df["country"],
        "timestamp": clean_df["invoice_date"],
    })
    return out


def build_user_item_ratings(clean_df: pd.DataFrame) -> pd.DataFrame:
    """Implicit-feedback signal: total units a customer bought of each product."""
    df = clean_df.dropna(subset=["customer_id"]).copy()
    df["customer_id"] = df["customer_id"].astype(int)
    ratings = (
        df.groupby(["customer_id", "product_id"])["quantity"]
        .sum().reset_index(name="rating")
    )
    ratings = ratings.rename(columns={"customer_id": "user_id"})
    ratings["rating"] = np.log1p(ratings["rating"])      # squash heavy buyers
    return ratings


def build_product_features(clean_df: pd.DataFrame) -> pd.DataFrame:
    """Per-product engagement + revenue metrics for the BI layer."""
    g = clean_df.groupby("product_id")
    feats = g.agg(
        name=("description", "first"),
        price=("price", "mean"),
        interactions=("invoice", "size"),       # order lines
        units=("quantity", "sum"),
        orders=("invoice", "nunique"),
        unique_users=("customer_id", "nunique"),
        est_revenue=("revenue", "sum"),
    ).reset_index()

    # "category" = the product's primary market (top country by revenue).
    top_country = (
        clean_df.groupby(["product_id", "country"])["revenue"].sum()
        .reset_index()
        .sort_values("revenue", ascending=False)
        .drop_duplicates("product_id")[["product_id", "country"]]
        .rename(columns={"country": "category"})
    )
    feats = feats.merge(top_country, on="product_id", how="left")
    feats["price"] = feats["price"].round(2)
    feats["est_revenue"] = feats["est_revenue"].round(2)
    feats["name"] = feats["name"].fillna("Unknown").str.title().str.slice(0, 40)
    return feats


def build_daily_sales(clean_df: pd.DataFrame) -> pd.DataFrame:
    """Daily revenue time series (trading days only) for ARIMA + anomalies.

    This retailer does not trade every calendar day (notably no Saturdays, plus
    UK holiday closures). Keeping only trading days gives the forecaster and the
    anomaly detector a clean signal instead of flagging every closure as a dip;
    the forecaster re-grids to a continuous daily index and interpolates the
    handful of remaining gaps.
    """
    daily = (
        clean_df.set_index("invoice_date")["revenue"]
        .resample("D").sum().reset_index()
    )
    daily.columns = ["date", "revenue"]
    daily = daily[daily["revenue"] > 0].reset_index(drop=True)   # trading days
    return daily


def build_country_revenue(clean_df: pd.DataFrame) -> pd.DataFrame:
    """Revenue by country for the dashboard's market-mix chart."""
    out = (
        clean_df.groupby("country")
        .agg(revenue=("revenue", "sum"),
             orders=("invoice", "nunique"))
        .reset_index()
        .sort_values("revenue", ascending=False)
    )
    out["revenue"] = out["revenue"].round(2)
    return out
