"""Anomaly detection on the daily revenue series.

Uses a robust median + MAD (median absolute deviation) z-score against a rolling
baseline. This is far less sensitive to the very spikes/dips we want to catch
than a mean/standard-deviation z-score would be, so promo spikes and outage-style
dips get flagged without the baseline itself being dragged around by them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config


def detect_sales_anomalies(sales: pd.DataFrame) -> pd.DataFrame:
    """Return the daily series annotated with robust_z / is_anomaly / type."""
    df = sales.sort_values("date").reset_index(drop=True).copy()

    window = 14
    med = df["revenue"].rolling(window, center=True, min_periods=3).median()
    resid = df["revenue"] - med

    mad = np.median(np.abs(resid - np.median(resid)))
    scale = 1.4826 * mad if mad > 0 else 1.0           # MAD -> std-equivalent
    df["robust_z"] = (resid / scale).round(2)
    df["is_anomaly"] = df["robust_z"].abs() > config.ANOMALY_Z_THRESHOLD
    df["anomaly_type"] = np.where(
        ~df["is_anomaly"], "normal",
        np.where(df["robust_z"] > 0, "spike", "dip"),
    )
    return df
