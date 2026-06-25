"""SARIMA (seasonal ARIMA) time-series forecasting of daily revenue.

A plain ARIMA forecast flattens to the series mean after a few days because it
has no notion of weekly structure. SARIMA adds a 7-day seasonal term so the
forecast keeps projecting the weekly sales rhythm forward.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

import config

warnings.filterwarnings("ignore")  # silence statsmodels convergence chatter


def forecast_scope(sales: pd.DataFrame, scope: str, label: str) -> pd.DataFrame:
    """Forecast one revenue series with SARIMA, in a unified long format.

    Columns: scope, label, date, kind ('history'|'forecast'), value, lower,
    upper, mape. Used to stack the overall + per-category forecasts into one
    table. Falls back to a simple non-seasonal ARIMA for short series that
    can't support the weekly seasonal fit.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.arima.model import ARIMA

    df = sales.sort_values("date").reset_index(drop=True)
    series = df.set_index("date")["revenue"].asfreq("D").interpolate()

    # Need at least ~2 full seasonal cycles to fit the weekly term sensibly.
    seasonal = len(series) >= 2 * config.SARIMA_SEASONAL_ORDER[3] + 14

    try:
        if seasonal:
            fitted = SARIMAX(
                series,
                order=config.SARIMA_ORDER,
                seasonal_order=config.SARIMA_SEASONAL_ORDER,
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False)
            tag = f"SARIMA{config.SARIMA_ORDER}x{config.SARIMA_SEASONAL_ORDER}"
        else:
            fitted = ARIMA(series, order=config.ARIMA_FALLBACK_ORDER).fit()
            tag = f"ARIMA{config.ARIMA_FALLBACK_ORDER} (short series)"

        fc = fitted.get_forecast(steps=config.FORECAST_HORIZON)
        mean, ci = fc.predicted_mean, fc.conf_int(alpha=0.05)
        tail = series.iloc[-30:]
        pred = fitted.fittedvalues.reindex(tail.index)
        m = tail > 0
        mape = float(np.mean(np.abs((tail[m] - pred[m]) / tail[m])) * 100)
        print(f"  {label}: {tag}, MAPE {mape:.1f}%")
    except Exception as exc:                       # pragma: no cover - safety net
        print(f"    [warn] {label}: fit failed ({exc}); flat forecast")
        idx = pd.date_range(series.index[-1] + pd.Timedelta(days=1),
                            periods=config.FORECAST_HORIZON, freq="D")
        mean = pd.Series(series.tail(14).mean(), index=idx)
        ci = pd.DataFrame({0: mean * 0.7, 1: mean * 1.3}, index=idx)
        mape = float("nan")

    # Revenue can't go negative — clip the forecast and its lower band at 0.
    mean = mean.clip(lower=0)
    lower = ci.iloc[:, 0].clip(lower=0)
    upper = ci.iloc[:, 1].clip(lower=0)

    hist = pd.DataFrame({
        "scope": scope, "label": label, "date": series.index,
        "kind": "history", "value": series.values.round(2),
        "lower": None, "upper": None,
    })
    fut = pd.DataFrame({
        "scope": scope, "label": label, "date": mean.index,
        "kind": "forecast", "value": mean.values.round(2),
        "lower": lower.values.round(2), "upper": upper.values.round(2),
    })
    out = pd.concat([hist, fut], ignore_index=True)
    out["mape"] = round(mape, 2)
    return out
