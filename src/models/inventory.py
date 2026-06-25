"""Inventory & slow-mover engine.

For every product it combines current stock, recent sales velocity and how long
it has sat unsold against its category shelf life to produce two admin actions:

  * REORDER   - fast sellers projected to run out -> suggested reorder quantity
  * CLEARANCE - aging / perishable stock not selling -> suggested discount

This is the operational decision layer the brief calls for: "predict sales
trends ... to help inventory purchase decisions, and promote/offer products
that are not being sold based on their perishability."
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config


def build_inventory_actions(
    products: pd.DataFrame, purchases: pd.DataFrame, as_of: pd.Timestamp
) -> pd.DataFrame:
    """Return one row per product with stock health + recommended action."""
    window = config.VELOCITY_WINDOW_DAYS
    recent_cutoff = as_of - pd.Timedelta(days=window)

    p = purchases.copy()
    p["timestamp"] = pd.to_datetime(p["timestamp"], format="mixed")

    # last sale + recent velocity per product
    last_sold = p.groupby("product_id")["timestamp"].max()
    recent = p[p["timestamp"] >= recent_cutoff]
    recent_units = recent.groupby("product_id")["qty"].sum()

    df = products.copy()
    df["last_sold"] = df["product_id"].map(last_sold)
    df["recent_units"] = df["product_id"].map(recent_units).fillna(0)
    df["days_since_sold"] = (as_of - pd.to_datetime(df["last_sold"])).dt.days
    df["days_since_sold"] = df["days_since_sold"].fillna(9999).astype(int)

    df["velocity"] = (df["recent_units"] / window).round(3)        # units/day
    df["est_demand_30d"] = (df["velocity"] * 30).round(1)

    # days of cover left at the current rate (inf when nothing is selling)
    with np.errstate(divide="ignore", invalid="ignore"):
        cover = np.where(df["velocity"] > 0,
                         df["stock_qty"] / df["velocity"], np.inf)
    df["days_of_stock"] = np.where(np.isfinite(cover),
                                   np.round(cover, 0), 9999).astype(int)

    # reorder: projected to run out within the lead window
    target = df["velocity"] * config.REORDER_COVER_DAYS
    reorder = np.where(
        (df["velocity"] > 0) & (df["days_of_stock"] < config.REORDER_LEAD_DAYS),
        np.ceil((target - df["stock_qty"]).clip(lower=0)), 0,
    )
    df["reorder_qty"] = reorder.astype(int)

    # clearance: unsold longer than its category shelf life, still in stock
    overdue = df["days_since_sold"] - df["shelf_life_days"]
    overdue_ratio = (overdue / df["shelf_life_days"]).clip(lower=0)
    df["suggested_discount"] = np.where(
        (overdue > 0) & (df["stock_qty"] > 0),
        np.minimum(config.MAX_CLEARANCE_DISCOUNT,
                   (0.10 + 0.40 * overdue_ratio)).round(2),
        0.0,
    )

    df["action"] = "healthy"
    df.loc[df["days_of_stock"] > 240, "action"] = "overstock"
    df.loc[df["suggested_discount"] > 0, "action"] = "clearance"
    df.loc[df["reorder_qty"] > 0, "action"] = "restock"

    df["last_sold"] = pd.to_datetime(df["last_sold"]).dt.strftime("%Y-%m-%d")
    cols = ["product_id", "stock_code", "name", "category", "price",
            "stock_qty", "recent_units", "velocity", "days_of_stock",
            "est_demand_30d", "days_since_sold", "shelf_life_days",
            "reorder_qty", "suggested_discount", "action", "last_sold"]
    return df[cols]
