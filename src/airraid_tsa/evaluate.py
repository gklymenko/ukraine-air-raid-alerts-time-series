"""Time-based evaluation: walk-forward, point + probabilistic metrics.

Public functions
----------------
time_split(series, cutoff_date)         -> (train, test)
mae(actual, predicted)                  -> float
rmse(actual, predicted)                 -> float
mase(actual, predicted, train, period)  -> float
interval_coverage(actual, lower, upper) -> float
mean_interval_width(lower, upper)       -> float
pinball_loss(actual, predicted, q)      -> float
print_metrics_table(metrics)            -> None   (printed + saved to outputs/)

Walk-forward evaluation (TODO Phase 3):
walk_forward_evaluate(forecaster, series, min_train, horizon) -> pd.DataFrame
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from airraid_tsa.config import OUTPUTS_DIR


# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------

def time_split(
    series: pd.Series,
    cutoff_date: pd.Timestamp,
) -> tuple[pd.Series, pd.Series]:
    """Split a time series at a cutoff date.

    Parameters
    ----------
    series : pd.Series
        Daily time series with a DatetimeIndex.
    cutoff_date : pd.Timestamp
        All dates strictly before cutoff go to train; cutoff and after to test.

    Returns
    -------
    (train, test) : tuple of pd.Series
    """
    train = series[series.index < cutoff_date]
    test = series[series.index >= cutoff_date]
    return train, test


# ---------------------------------------------------------------------------
# Point metrics
# ---------------------------------------------------------------------------

def mae(actual: pd.Series, predicted: pd.Series) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(actual.values - predicted.values)))


def rmse(actual: pd.Series, predicted: pd.Series) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((actual.values - predicted.values) ** 2)))


def mase(
    actual: pd.Series,
    predicted: pd.Series,
    train: pd.Series,
    seasonal_period: int = 7,
) -> float:
    """Mean Absolute Scaled Error.

    Scales MAE by the in-sample (train) MAE of the seasonal naive forecaster
    (y[t] = y[t - seasonal_period]).  A value < 1 means the model beats that
    naive baseline.

    Parameters
    ----------
    actual : pd.Series       Test-set actuals.
    predicted : pd.Series    Test-set predictions aligned with actual.
    train : pd.Series        Training series used to compute the scale.
    seasonal_period : int    Season length in periods (default 7 for weekly).

    Returns
    -------
    float
        MASE value, or NaN if the scale denominator is zero.
    """
    model_errors = np.abs(actual.values - predicted.values)

    # In-sample seasonal naive errors.
    naive_errors = np.abs(
        train.values[seasonal_period:] - train.values[:-seasonal_period]
    )
    scale = float(np.mean(naive_errors)) if len(naive_errors) > 0 else 0.0

    if scale == 0.0:
        return float("nan")

    return float(np.mean(model_errors) / scale)


# ---------------------------------------------------------------------------
# Probabilistic metrics
# ---------------------------------------------------------------------------

def interval_coverage(
    actual: pd.Series,
    lower: pd.Series,
    upper: pd.Series,
) -> float:
    """Fraction of actual values that fall within [lower, upper].

    Well-calibrated ≈ nominal level (e.g. 0.80 for an 80 % PI).
    """
    inside = (actual.values >= lower.values) & (actual.values <= upper.values)
    return float(inside.mean())


def mean_interval_width(lower: pd.Series, upper: pd.Series) -> float:
    """Average width of the prediction interval (sharpness — lower is sharper)."""
    return float((upper.values - lower.values).mean())


def pinball_loss(actual: pd.Series, predicted: pd.Series, quantile: float) -> float:
    """Pinball (quantile) loss — a proper scoring rule.

    Lower is better.  At quantile=0.5 this equals MAE / 2.

    Parameters
    ----------
    actual : pd.Series
    predicted : pd.Series    The quantile forecast (lower or upper bound).
    quantile : float         The target quantile (e.g. 0.10 for the lower bound
                             of an 80 % PI, 0.90 for the upper bound).
    """
    errors = actual.values - predicted.values
    return float(np.mean(np.where(errors >= 0, quantile * errors, (quantile - 1) * errors)))


# ---------------------------------------------------------------------------
# Metrics table output
# ---------------------------------------------------------------------------

def print_metrics_table(metrics: dict[str, float], output_dir: Path = OUTPUTS_DIR) -> None:
    """Print a metrics dict as a formatted table and save it to outputs/.

    Parameters
    ----------
    metrics : dict[str, float]
        Keys are metric names, values are floats.
    output_dir : Path
        Directory where metrics.txt is saved.
    """
    lines = ["=" * 40, "FORECAST METRICS", "=" * 40]
    for name, value in metrics.items():
        lines.append(f"  {name:<22} {value:.4f}")
    lines.append("=" * 40)

    table = "\n".join(lines)
    print(table)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.txt").write_text(table + "\n")