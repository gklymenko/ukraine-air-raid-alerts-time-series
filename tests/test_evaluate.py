"""Unit tests for evaluate.py — metric functions only (walk-forward is Phase 3)."""

import math

import pandas as pd
import pytest

from airraid_tsa.evaluate import (
    interval_coverage,
    mae,
    mase,
    mean_interval_width,
    pinball_loss,
    rmse,
    time_split,
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