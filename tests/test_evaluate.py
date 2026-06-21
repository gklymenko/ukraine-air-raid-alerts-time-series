"""Unit tests for evaluate.py — metric functions and walk-forward backtest."""

import math

import pandas as pd
import pytest

from airraid_tsa.evaluate import (
    forward_forecast,
    interval_coverage,
    mae,
    mase,
    mean_interval_width,
    pinball_loss,
    rmse,
    time_split,
    walk_forward,
)
from airraid_tsa.forecast.baselines import (
    MovingAverageForecaster,
    NaiveForecaster,
    SeasonalNaiveForecaster,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def daily_series() -> pd.Series:
    """A simple 14-day series: values 0..13."""
    idx = pd.date_range("2024-01-01", periods=14, freq="D", tz="UTC")
    return pd.Series(range(14), index=idx, dtype=float)


# ---------------------------------------------------------------------------
# time_split
# ---------------------------------------------------------------------------

class TestTimeSplit:
    def test_split_sizes(self, daily_series):
        cutoff = pd.Timestamp("2024-01-08", tz="UTC")
        train, test = time_split(daily_series, cutoff)
        assert len(train) == 7   # Jan 1–7
        assert len(test) == 7    # Jan 8–14

    def test_train_before_cutoff(self, daily_series):
        cutoff = pd.Timestamp("2024-01-08", tz="UTC")
        train, _ = time_split(daily_series, cutoff)
        assert (train.index < cutoff).all()

    def test_test_at_and_after_cutoff(self, daily_series):
        cutoff = pd.Timestamp("2024-01-08", tz="UTC")
        _, test = time_split(daily_series, cutoff)
        assert (test.index >= cutoff).all()


# ---------------------------------------------------------------------------
# Point metrics
# ---------------------------------------------------------------------------

class TestMAE:
    def test_perfect_forecast_is_zero(self):
        actual = pd.Series([1.0, 2.0, 3.0])
        assert mae(actual, actual) == pytest.approx(0.0)

    def test_constant_error(self):
        actual = pd.Series([1.0, 2.0, 3.0])
        predicted = pd.Series([2.0, 3.0, 4.0])  # +1 everywhere
        assert mae(actual, predicted) == pytest.approx(1.0)

    def test_mixed_errors(self):
        actual = pd.Series([0.0, 0.0, 0.0])
        predicted = pd.Series([1.0, -2.0, 3.0])
        assert mae(actual, predicted) == pytest.approx(2.0)


class TestRMSE:
    def test_perfect_forecast_is_zero(self):
        actual = pd.Series([1.0, 2.0, 3.0])
        assert rmse(actual, actual) == pytest.approx(0.0)

    def test_constant_error(self):
        actual = pd.Series([1.0, 2.0, 3.0])
        predicted = pd.Series([2.0, 3.0, 4.0])
        assert rmse(actual, predicted) == pytest.approx(1.0)

    def test_rmse_penalises_large_errors(self):
        actual = pd.Series([0.0, 0.0])
        predicted_mild   = pd.Series([1.0, 1.0])    # MAE = 1, RMSE = 1
        predicted_spiked = pd.Series([0.0, 2.0])    # MAE = 1, RMSE > 1
        assert rmse(actual, predicted_spiked) > rmse(actual, predicted_mild)


class TestMASE:
    def test_perfect_model_is_zero(self):
        train = pd.Series(range(20), dtype=float)
        actual = pd.Series([20.0, 21.0, 22.0])
        predicted = actual.copy()
        result = mase(actual, predicted, train=train, seasonal_period=7)
        assert result == pytest.approx(0.0)

    def test_returns_nan_when_scale_is_zero(self):
        # Constant training series → seasonal naive error = 0 → scale = 0
        train = pd.Series([5.0] * 20)
        actual = pd.Series([6.0, 7.0])
        predicted = pd.Series([5.0, 5.0])
        result = mase(actual, predicted, train=train, seasonal_period=7)
        assert math.isnan(result)

    def test_mase_finite_for_noisy_series(self):
        # A noisy weekly-seasonal series: verify MASE is finite and positive.
        # (A perfectly periodic series has zero in-sample naive error → NaN.)
        import numpy as np
        rng = np.random.default_rng(42)
        base = np.tile([10.0, 8.0, 12.0, 9.0, 11.0, 7.0, 6.0], 8)  # 56 days
        vals = pd.Series(base + rng.normal(0, 1.0, 56))

        train = vals.iloc[:49]
        actual = vals.iloc[49:].reset_index(drop=True)
        predicted = vals.iloc[42:49].reset_index(drop=True)  # lag-7 naive

        result = mase(actual, predicted, train=train, seasonal_period=7)
        assert pd.notna(result)
        assert result > 0


# ---------------------------------------------------------------------------
# Probabilistic metrics
# ---------------------------------------------------------------------------

class TestIntervalCoverage:
    def test_all_inside(self):
        actual = pd.Series([1.0, 2.0, 3.0])
        lower  = pd.Series([0.0, 0.0, 0.0])
        upper  = pd.Series([9.0, 9.0, 9.0])
        assert interval_coverage(actual, lower, upper) == pytest.approx(1.0)

    def test_all_outside(self):
        actual = pd.Series([1.0, 2.0, 3.0])
        lower  = pd.Series([10.0, 10.0, 10.0])
        upper  = pd.Series([20.0, 20.0, 20.0])
        assert interval_coverage(actual, lower, upper) == pytest.approx(0.0)

    def test_half_inside(self):
        actual = pd.Series([1.0, 5.0])
        lower  = pd.Series([0.0, 0.0])
        upper  = pd.Series([2.0, 2.0])  # 1.0 inside, 5.0 outside
        assert interval_coverage(actual, lower, upper) == pytest.approx(0.5)

    def test_boundary_values_are_inside(self):
        actual = pd.Series([0.0, 4.0])
        lower  = pd.Series([0.0, 0.0])
        upper  = pd.Series([4.0, 4.0])
        assert interval_coverage(actual, lower, upper) == pytest.approx(1.0)


class TestMeanIntervalWidth:
    def test_constant_width(self):
        lower = pd.Series([0.0, 0.0, 0.0])
        upper = pd.Series([4.0, 4.0, 4.0])
        assert mean_interval_width(lower, upper) == pytest.approx(4.0)

    def test_varying_width(self):
        lower = pd.Series([0.0, 0.0])
        upper = pd.Series([2.0, 4.0])  # widths 2 and 4, mean = 3
        assert mean_interval_width(lower, upper) == pytest.approx(3.0)


class TestPinballLoss:
    def test_perfect_forecast_is_zero(self):
        actual = pd.Series([1.0, 2.0, 3.0])
        assert pinball_loss(actual, actual, quantile=0.5) == pytest.approx(0.0)

    def test_at_median_equals_half_mae(self):
        actual    = pd.Series([1.0, 2.0, 3.0])
        predicted = pd.Series([2.0, 3.0, 4.0])  # constant over-prediction of 1
        # pinball at q=0.5 for over-prediction: (q-1)*(actual-pred) = -0.5*(-1)=0.5
        assert pinball_loss(actual, predicted, quantile=0.5) == pytest.approx(0.5)

    def test_asymmetry_under_vs_over(self):
        # At q=0.9, under-predicting is penalised more than over-predicting.
        actual       = pd.Series([5.0])
        under_pred   = pd.Series([4.0])  # error = +1 (actual > predicted)
        over_pred    = pd.Series([6.0])  # error = -1 (actual < predicted)
        assert pinball_loss(actual, under_pred, quantile=0.9) > \
               pinball_loss(actual, over_pred,  quantile=0.9)


# ---------------------------------------------------------------------------
# Walk-forward
# ---------------------------------------------------------------------------

@pytest.fixture()
def utc_series() -> pd.Series:
    """20-day constant-value series (value=10.0) with a UTC DatetimeIndex."""
    idx = pd.date_range("2024-01-01", periods=20, freq="D", tz="UTC")
    return pd.Series(10.0, index=idx)


class TestWalkForward:
    def test_result_has_correct_columns(self, utc_series):
        results = walk_forward(NaiveForecaster(), utc_series, initial_train_size=14)
        assert set(results.columns) == {"actual", "point", "lower", "upper"}

    def test_number_of_folds(self, utc_series):
        # initial_train_size=14, n=20, horizon=1, step=1  → 6 folds (days 15–20)
        results = walk_forward(NaiveForecaster(), utc_series, initial_train_size=14)
        assert len(results) == 6

    def test_step_reduces_folds(self, utc_series):
        # step=2 halves the number of folds (ceiling of 6/2 = 3)
        results = walk_forward(NaiveForecaster(), utc_series, initial_train_size=14, step=2)
        assert len(results) == 3

    def test_naive_point_forecast_on_constant_series(self, utc_series):
        # NaiveForecaster on a constant series → point == actual everywhere → MAE == 0
        results = walk_forward(NaiveForecaster(), utc_series, initial_train_size=14)
        assert mae(results["actual"], results["point"]) == pytest.approx(0.0)

    def test_interval_covers_constant_series(self, utc_series):
        # Constant series has zero residuals → lower == upper == point → coverage = 1.0
        # (actual == point, and lower <= actual <= upper holds at boundary)
        results = walk_forward(NaiveForecaster(), utc_series, initial_train_size=14)
        cov = interval_coverage(results["actual"], results["lower"], results["upper"])
        assert cov == pytest.approx(1.0)

    def test_seasonal_naive_on_weekly_constant(self):
        # Perfect weekly pattern: SeasonalNaive should forecast exactly.
        idx = pd.date_range("2024-01-01", periods=21, freq="D", tz="UTC")
        pattern = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0] * 3
        y = pd.Series(pattern, index=idx)
        results = walk_forward(SeasonalNaiveForecaster(), y, initial_train_size=14)
        assert mae(results["actual"], results["point"]) == pytest.approx(0.0)

    def test_walk_forward_returns_empty_when_no_folds(self, utc_series):
        # initial_train_size >= len(y) → no folds possible
        results = walk_forward(NaiveForecaster(), utc_series, initial_train_size=20)
        assert results.empty

    def test_lower_le_upper_everywhere(self, utc_series):
        results = walk_forward(MovingAverageForecaster(window=7), utc_series, initial_train_size=14)
        assert (results["lower"] <= results["upper"]).all()


# ---------------------------------------------------------------------------
# Forward forecast
# ---------------------------------------------------------------------------

class TestForwardForecast:
    @pytest.fixture()
    def metrics_stub(self) -> pd.DataFrame:
        """Minimal metrics DataFrame with MAE column — SeasonalNaive wins."""
        return pd.DataFrame(
            {"MAE": [15.0, 8.0, 12.0]},
            index=pd.Index(
                ["NaiveForecaster", "SeasonalNaive", "MovingAverage7"],
                name="forecaster",
            ),
        )

    @pytest.fixture()
    def long_series(self) -> pd.Series:
        """28-day stationary series with a weekly pattern.

        Lag-7 residuals are all zero → lower = point = upper, so lower <= point <= upper
        holds unconditionally regardless of which forecaster is selected.
        """
        pattern = [100.0, 120.0, 90.0, 110.0, 80.0, 130.0, 95.0] * 4  # 28 days
        idx = pd.date_range("2024-01-01", periods=28, freq="D", tz="UTC")
        return pd.Series(pattern, index=idx)

    def test_csv_has_exactly_7_rows(self, metrics_stub, long_series, tmp_path):
        forward_forecast(metrics_stub, long_series, horizon=7, output_dir=tmp_path)
        df = pd.read_csv(tmp_path / "forecast_next_7_days.csv")
        assert len(df) == 7

    def test_all_dates_strictly_after_series_end(self, metrics_stub, long_series, tmp_path):
        forward_forecast(metrics_stub, long_series, horizon=7, output_dir=tmp_path)
        df = pd.read_csv(tmp_path / "forecast_next_7_days.csv", parse_dates=["date"])
        series_end = long_series.index[-1].tz_localize(None)
        assert (df["date"] > series_end).all()

    def test_lower_le_point_le_upper(self, metrics_stub, long_series, tmp_path):
        forward_forecast(metrics_stub, long_series, horizon=7, output_dir=tmp_path)
        df = pd.read_csv(tmp_path / "forecast_next_7_days.csv")
        assert (df["lower"] <= df["point"]).all()
        assert (df["point"] <= df["upper"]).all()

    def test_select_by_is_respected(self, long_series, tmp_path):
        # Build a metrics stub where RMSE column picks a different winner than MAE.
        metrics = pd.DataFrame(
            {"MAE": [5.0, 10.0, 8.0], "RMSE": [9.0, 4.0, 7.0]},
            index=pd.Index(
                ["NaiveForecaster", "SeasonalNaive", "MovingAverage7"],
                name="forecaster",
            ),
        )
        forward_forecast(metrics, long_series, horizon=7, select_by="RMSE", output_dir=tmp_path)
        df = pd.read_csv(tmp_path / "forecast_next_7_days.csv")
        assert df["model"].iloc[0] == "SeasonalNaive"

    def test_model_column_matches_winner(self, metrics_stub, long_series, tmp_path):
        forward_forecast(metrics_stub, long_series, horizon=7, output_dir=tmp_path)
        df = pd.read_csv(tmp_path / "forecast_next_7_days.csv")
        assert (df["model"] == "SeasonalNaive").all()  # lowest MAE in metrics_stub