"""ARIMA time-series forecasting of daily revenue."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

import config

warnings.filterwarnings("ignore")  # silence statsmodels convergence chatter


def forecast_revenue(sales: pd.DataFrame) -> pd.DataFrame:
    """Fit ARIMA on the cleaned daily-revenue series and forecast forward.

    Returns one tidy frame containing the in-sample fit, the forward forecast
    and 95% confidence bounds, ready to load into the warehouse and chart.
    """
    from statsmodels.tsa.arima.model import ARIMA

    df = sales.sort_values("date").reset_index(drop=True)
    series = df.set_index("date")["revenue"].asfreq("D").interpolate()

    model = ARIMA(series, order=config.ARIMA_ORDER)
    fitted = model.fit()
    print(f"  ARIMA{config.ARIMA_ORDER} fitted  (AIC={fitted.aic:,.0f})")

    horizon = config.FORECAST_HORIZON
    fc = fitted.get_forecast(steps=horizon)
    mean = fc.predicted_mean
    ci = fc.conf_int(alpha=0.05)

    history = pd.DataFrame({
        "date": series.index,
        "actual": series.values,
        "kind": "history",
    })
    future = pd.DataFrame({
        "date": mean.index,
        "forecast": mean.values.round(2),
        "lower": ci.iloc[:, 0].values.round(2),
        "upper": ci.iloc[:, 1].values.round(2),
        "kind": "forecast",
    })
    out = pd.concat([history, future], ignore_index=True)

    # Simple backtest: in-sample MAPE on the trailing 30 days. Guard against
    # any zero-revenue days so a single closure can't blow the metric up to inf.
    fitted_vals = fitted.fittedvalues
    tail = series.iloc[-30:]
    pred = fitted_vals.reindex(tail.index)
    mask = tail > 0
    mape = float(np.mean(np.abs((tail[mask] - pred[mask]) / tail[mask])) * 100)
    print(f"  in-sample MAPE (last 30d): {mape:.2f}%")
    out.attrs["mape"] = round(mape, 2)
    return out


def forecast_scope(sales: pd.DataFrame, scope: str, label: str) -> pd.DataFrame:
    """Forecast one revenue series and return it in a unified long format.

    Columns: scope, label, date, kind ('history'|'forecast'), value, lower,
    upper, mape. Used to stack overall + per-category forecasts into one table.
    """
    from statsmodels.tsa.arima.model import ARIMA

    df = sales.sort_values("date").reset_index(drop=True)
    series = df.set_index("date")["revenue"].asfreq("D").interpolate()
    # short/flat series can't support the full order -> fall back gracefully
    order = config.ARIMA_ORDER if len(series) > 60 else (1, 1, 1)

    try:
        fitted = ARIMA(series, order=order).fit()
        fc = fitted.get_forecast(steps=config.FORECAST_HORIZON)
        mean, ci = fc.predicted_mean, fc.conf_int(alpha=0.05)
        tail = series.iloc[-30:]
        pred = fitted.fittedvalues.reindex(tail.index)
        m = tail > 0
        mape = float(np.mean(np.abs((tail[m] - pred[m]) / tail[m])) * 100)
    except Exception as exc:                       # pragma: no cover - safety net
        print(f"    [warn] {label}: ARIMA failed ({exc}); flat forecast")
        idx = pd.date_range(series.index[-1] + pd.Timedelta(days=1),
                            periods=config.FORECAST_HORIZON, freq="D")
        mean = pd.Series(series.tail(14).mean(), index=idx)
        ci = pd.DataFrame({0: mean * 0.7, 1: mean * 1.3}, index=idx)
        mape = float("nan")

    hist = pd.DataFrame({
        "scope": scope, "label": label, "date": series.index,
        "kind": "history", "value": series.values.round(2),
        "lower": None, "upper": None,
    })
    fut = pd.DataFrame({
        "scope": scope, "label": label, "date": mean.index,
        "kind": "forecast", "value": mean.values.round(2),
        "lower": ci.iloc[:, 0].values.round(2),
        "upper": ci.iloc[:, 1].values.round(2),
    })
    out = pd.concat([hist, fut], ignore_index=True)
    out["mape"] = round(mape, 2)
    return out
