"""Catalog enrichment: derive categories + synthesize inventory state.

A pure sales export has no merchandising metadata (category) or operational
state (current stock on hand, when an item was first stocked). This module
adds both deterministically so the storefront has browsable categories and the
admin inventory / perishability engine has stock + age to reason about.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config


def categorize(description: str) -> str:
    """Map a raw product description to a shopper-facing category."""
    text = (description or "").upper()
    for category, keywords in config.CATEGORY_KEYWORDS:
        if any(k in text for k in keywords):
            return category
    return config.DEFAULT_CATEGORY


def shelf_life_days(category: str) -> int:
    return config.CATEGORY_SHELF_LIFE.get(category, config.DEFAULT_SHELF_LIFE)


def synthesize_inventory(products: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """Add stock_qty and date_added columns, seeded for reproducibility.

    Stock is loosely tied to historical demand (popular items are restocked
    more), with deliberate over/under-stocking so the reorder + clearance
    engine has interesting cases to surface.
    """
    rng = np.random.default_rng(config.RANDOM_SEED)
    n = len(products)
    out = products.copy().reset_index(drop=True)

    units = out.get("units", pd.Series(np.ones(n))).fillna(0).to_numpy()
    # baseline stock ~ a few weeks of historical demand, with noise
    base = np.clip(units / max(units.max(), 1) * 400, 5, None)
    noise = rng.uniform(0.3, 2.5, size=n)
    out["stock_qty"] = np.round(base * noise).astype(int) + rng.integers(0, 25, n)

    # product age: first time we ever saw the item, jittered earlier a bit
    if "first_seen" in out.columns:
        first = pd.to_datetime(out["first_seen"])
    else:
        first = pd.Series([as_of - pd.Timedelta(days=400)] * n)
    jitter = pd.to_timedelta(rng.integers(0, 120, size=n), unit="D")
    out["date_added"] = (first - jitter).dt.normalize()
    return out
