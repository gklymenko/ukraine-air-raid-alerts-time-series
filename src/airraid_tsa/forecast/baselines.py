"""Baseline Forecaster implementations.

NaiveForecaster
    Point = last observed value; interval from lag-1 in-sample residuals.

SeasonalNaiveForecaster
    Point = value from the same weekday last week (lag-7).
    Primary bar to beat.

MovingAverageForecaster
    Point = rolling mean of the last `window` observations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from airraid_tsa.forecast.base import Forecast, Forecaster


def _future_index(last_date: pd.Timestamp, horizon: int) -> pd.DatetimeIndex:
    """Daily DatetimeIndex starting the day after last_date."""
    return pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=horizon,
        freq="D",
        tz=last_date.tzinfo,
    )


def _interval_shifts(residuals: np.ndarray, level: float) -> tuple[float, float]:
    """Additive lower/upper shifts at the nominal prediction-interval level."""
    if len(residuals) == 0:
        return 0.0, 0.0
    q_low = (1.0 - level) / 2.0
    q_high = (1.0 + level) / 2.0
    return float(np.quantile(residuals, q_low)), float(np.quantile(residuals, q_high))


class NaiveForecaster(Forecaster):
    """Point = last observed value for all forecast steps."""

    def fit(self, y: pd.Series) -> "NaiveForecaster":
        self._last_value = float(y.iloc[-1])
        self._last_date = y.index[-1]
        # Residuals: actual[t] - last_value_at_t = y[t] - y[t-1]
        self._residuals = y.values[1:] - y.values[:-1]
        return self

    def predict(self, horizon: int, level: float = 0.80) -> Forecast:
        r_low, r_high = _interval_shifts(self._residuals, level)
        idx = _future_index(self._last_date, horizon)
        pts = np.full(horizon, self._last_value)
        return Forecast(
            point=pd.Series(pts, index=idx),
            lower=pd.Series(pts + r_low, index=idx),
            upper=pd.Series(pts + r_high, index=idx),
            level=level,
        )


class SeasonalNaiveForecaster(Forecaster):
    """Point = value from the same weekday last week (lag-7)."""

    def __init__(self, period: int = 7) -> None:
        self._period = period

    def fit(self, y: pd.Series) -> "SeasonalNaiveForecaster":
        self._y = y.copy()
        self._last_date = y.index[-1]
        p = self._period
        if len(y) > p:
            # Residuals: y[t] - y[t - period]
            self._residuals = y.values[p:] - y.values[:-p]
        else:
            self._residuals = np.array([], dtype=float)
        return self

    def predict(self, horizon: int, level: float = 0.80) -> Forecast:
        r_low, r_high = _interval_shifts(self._residuals, level)
        idx = _future_index(self._last_date, horizon)
        p = self._period
        # For step h (0-indexed): repeat the last `period` values cyclically.
        # h=0 → y[-p], h=1 → y[-p+1], …, h=p-1 → y[-1], h=p → y[-p] again.
        pts = np.array([float(self._y.iloc[-(p - h % p)]) for h in range(horizon)])
        return Forecast(
            point=pd.Series(pts, index=idx),
            lower=pd.Series(pts + r_low, index=idx),
            upper=pd.Series(pts + r_high, index=idx),
            level=level,
        )


class MovingAverageForecaster(Forecaster):
    """Point = mean of the last `window` observations (flat for all steps)."""

    def __init__(self, window: int = 7) -> None:
        self._window = window

    def fit(self, y: pd.Series) -> "MovingAverageForecaster":
        w = self._window
        self._last_window = y.values[-w:] if len(y) >= w else y.values
        self._last_date = y.index[-1]
        if len(y) > w:
            # Residuals: y[t] - mean(y[t-w : t])
            rolling_means = np.array([y.values[t - w:t].mean() for t in range(w, len(y))])
            self._residuals = y.values[w:] - rolling_means
        else:
            self._residuals = np.array([], dtype=float)
        return self

    def predict(self, horizon: int, level: float = 0.80) -> Forecast:
        r_low, r_high = _interval_shifts(self._residuals, level)
        idx = _future_index(self._last_date, horizon)
        point_val = float(self._last_window.mean()) if len(self._last_window) > 0 else 0.0
        pts = np.full(horizon, point_val)
        return Forecast(
            point=pd.Series(pts, index=idx),
            lower=pd.Series(pts + r_low, index=idx),
            upper=pd.Series(pts + r_high, index=idx),
            level=level,
        )
