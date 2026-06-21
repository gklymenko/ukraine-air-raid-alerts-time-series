"""Time-based evaluation: walk-forward, point + probabilistic metrics.

Public functions
----------------
time_split(series, cutoff_date)                           -> (train, test)
mae(actual, predicted)                                    -> float
rmse(actual, predicted)                                   -> float
mase(actual, predicted, train, period)                    -> float
interval_coverage(actual, lower, upper)                   -> float
mean_interval_width(lower, upper)                         -> float
pinball_loss(actual, predicted, q)                        -> float
print_metrics_table(metrics)                              -> None
walk_forward(forecaster, y, initial_train_size, ...)      -> pd.DataFrame
evaluate_all(y, level)                                    -> pd.DataFrame
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from airraid_tsa.config import OUTPUTS_DIR

if TYPE_CHECKING:
    from airraid_tsa.forecast.base import Forecaster


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


# ---------------------------------------------------------------------------
# Walk-forward backtest
# ---------------------------------------------------------------------------

def walk_forward(
    forecaster: "Forecaster",
    y: pd.Series,
    initial_train_size: int,
    horizon: int = 1,
    step: int = 1,
    level: float = 0.80,
) -> pd.DataFrame:
    """Expanding-window walk-forward backtest.

    At each fold, the forecaster is re-fit on all data up to that point
    (the training window expands by `step` each iteration), then asked to
    predict `horizon` steps ahead.  The test observations are collected and
    returned as a DataFrame.

    Parameters
    ----------
    forecaster        : Forecaster   Any fitted or unfitted Forecaster instance.
    y                 : pd.Series    Full daily series (DatetimeIndex).
    initial_train_size: int          Number of observations in the first training window.
    horizon           : int          Steps ahead to forecast per fold (default 1).
    step              : int          How many observations to advance per fold (default 1).
    level             : float        Nominal prediction-interval coverage.

    Returns
    -------
    pd.DataFrame
        Indexed by date with columns: actual, point, lower, upper.
        Rows cover all test observations across all folds.
    """
    n = len(y)
    records: list[dict] = []

    for train_end in range(initial_train_size, n - horizon + 1, step):
        train = y.iloc[:train_end]
        test_slice = y.iloc[train_end:train_end + horizon]

        forecaster.fit(train)
        fc = forecaster.predict(horizon=horizon, level=level)

        for i, (date, actual) in enumerate(test_slice.items()):
            records.append(
                {
                    "date": date,
                    "actual": float(actual),
                    "point": float(fc.point.iloc[i]),
                    "lower": float(fc.lower.iloc[i]),
                    "upper": float(fc.upper.iloc[i]),
                }
            )

    if not records:
        return pd.DataFrame(columns=["actual", "point", "lower", "upper"])

    return pd.DataFrame(records).set_index("date")


# ---------------------------------------------------------------------------
# Multi-forecaster evaluation
# ---------------------------------------------------------------------------

def evaluate_all(
    y: pd.Series,
    level: float = 0.80,
    output_dir: Path = OUTPUTS_DIR,
) -> pd.DataFrame:
    """Run all three baselines through walk-forward and return a metrics table.

    Uses an expanding-window backtest with initial_train_size = 2/3 of the
    series length (minimum 14 observations so seasonal residuals are meaningful).

    Parameters
    ----------
    y          : pd.Series   Full daily series for one oblast.
    level      : float       Nominal PI coverage (default 0.80).
    output_dir : Path        Where to write metrics.csv.

    Returns
    -------
    pd.DataFrame
        Rows = forecasters, columns = MAE, RMSE, MASE, Coverage,
        Interval_Width, Pinball_q10, Pinball_q90, Pinball_avg.
        Also printed to stdout and saved to output_dir/metrics.csv.
    """
    # Deferred import avoids a circular dependency at module load time.
    from airraid_tsa.forecast.baselines import (
        MovingAverageForecaster,
        NaiveForecaster,
        SeasonalNaiveForecaster,
    )

    initial_train_size = max(14, len(y) * 2 // 3)

    forecasters: list[tuple[str, "Forecaster"]] = [
        ("NaiveForecaster", NaiveForecaster()),
        ("SeasonalNaive", SeasonalNaiveForecaster()),
        ("MovingAverage7", MovingAverageForecaster(window=7)),
    ]

    rows = []
    all_results: dict[str, pd.DataFrame] = {}

    for name, fc in forecasters:
        results = walk_forward(fc, y, initial_train_size, horizon=1, level=level)
        all_results[name] = results

        if results.empty:
            continue

        actual = results["actual"]
        point = results["point"]
        lower = results["lower"]
        upper = results["upper"]
        train = y.iloc[:initial_train_size]

        q_low = (1.0 - level) / 2.0
        q_high = (1.0 + level) / 2.0
        pl_low = pinball_loss(actual, lower, q_low)
        pl_high = pinball_loss(actual, upper, q_high)

        rows.append(
            {
                "forecaster": name,
                "MAE": mae(actual, point),
                "RMSE": rmse(actual, point),
                "MASE": mase(actual, point, train, seasonal_period=7),
                "Coverage": interval_coverage(actual, lower, upper),
                "Interval_Width": mean_interval_width(lower, upper),
                "Pinball_q10": pl_low,
                "Pinball_q90": pl_high,
                "Pinball_avg": (pl_low + pl_high) / 2.0,
            }
        )

    metrics_df = pd.DataFrame(rows).set_index("forecaster")

    print("\n" + "=" * 78)
    print(f"FORECAST EVALUATION  (walk-forward, horizon=1, level={level:.0%})")
    print("=" * 78)
    print(metrics_df.to_string(float_format=lambda x: f"{x:.4f}"))
    print("=" * 78)

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "metrics.csv"
    metrics_df.to_csv(csv_path)
    print(f"  [evaluate] Saved metrics: {csv_path}")

    # Plot forecast vs actual for the best baseline (lowest MAE).
    if not metrics_df.empty:
        best_name = str(metrics_df["MAE"].idxmin())
        best_results = all_results[best_name]
        _plot_forecast(best_results, best_name, level, output_dir)

    return metrics_df


def _plot_forecast(
    results: pd.DataFrame,
    forecaster_name: str,
    level: float,
    output_dir: Path,
) -> None:
    """Forecast-vs-actual plot with shaded prediction interval."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 5))

    ax.fill_between(
        results.index,
        results["lower"],
        results["upper"],
        alpha=0.25,
        color="steelblue",
        label=f"{level:.0%} prediction interval",
    )
    ax.plot(results.index, results["actual"], color="black", linewidth=1.2, label="Actual")
    ax.plot(
        results.index, results["point"],
        color="steelblue", linewidth=1.0, linestyle="--", label=f"Point ({forecaster_name})",
    )

    ax.set_title(f"Kyiv City — alert_minutes forecast vs actual  [{forecaster_name}]")
    ax.set_xlabel("Date")
    ax.set_ylabel("alert_minutes")
    ax.legend()
    ax.grid(True, alpha=0.25)
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "forecast_kyiv_city.png"
    fig.savefig(path, bbox_inches="tight", dpi=100)
    plt.close(fig)
    print(f"  [evaluate] Saved forecast plot: {path}")