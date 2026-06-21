"""Baseline Forecaster implementations.

TODO (Phase 3): Implement the three baselines below.

NaiveForecaster
    Point forecast = last observed value.
    Interval from in-sample residual quantiles.

SeasonalNaiveForecaster
    Point forecast = value from same weekday last week (lag-7).
    Interval from in-sample residual quantiles.
    This is the primary bar to beat.

MovingAverageForecaster
    Point forecast = rolling mean of the last `window` observations.
    Interval from rolling residual quantiles.
"""

from airraid_tsa.forecast.base import Forecast, Forecaster  # noqa: F401

# TODO: implement NaiveForecaster, SeasonalNaiveForecaster, MovingAverageForecaster.