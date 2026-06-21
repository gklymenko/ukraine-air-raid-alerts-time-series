"""Unit tests for resample.py — focuses on the tricky edge cases."""

import pandas as pd
import pytest

from airraid_tsa.resample import resample_daily


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz="UTC")


def make_events(*rows: tuple) -> pd.DataFrame:
    """Build a minimal canonical events DataFrame from (region, start, end, naive) tuples."""
    records = [
        {"region": r, "started_at": s, "finished_at": e, "naive": n, "source": "test"}
        for r, s, e, n in rows
    ]
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Basic single-day alerts
# ---------------------------------------------------------------------------

class TestSingleDayAlert:
    def test_count_and_minutes(self):
        events = make_events(
            ("Kyiv City", ts("2024-01-01 10:00"), ts("2024-01-01 11:30"), False),
        )
        daily = resample_daily(events)
        row = daily.xs("Kyiv City", level="region").loc[ts("2024-01-01")]

        assert row["alerts_count"] == 1
        assert row["alert_minutes"] == pytest.approx(90.0)

    def test_two_alerts_same_day_aggregated(self):
        events = make_events(
            ("Kyiv City", ts("2024-01-01 10:00"), ts("2024-01-01 10:30"), False),
            ("Kyiv City", ts("2024-01-01 14:00"), ts("2024-01-01 14:45"), False),
        )
        daily = resample_daily(events)
        row = daily.xs("Kyiv City", level="region").loc[ts("2024-01-01")]

        assert row["alerts_count"] == 2
        assert row["alert_minutes"] == pytest.approx(75.0)

    def test_alerts_count_is_integer_dtype(self):
        events = make_events(
            ("Kyiv City", ts("2024-01-01 10:00"), ts("2024-01-01 11:00"), False),
        )
        daily = resample_daily(events)
        assert pd.api.types.is_integer_dtype(daily["alerts_count"])


# ---------------------------------------------------------------------------
# Midnight-crossing alerts (the key correctness case)
# ---------------------------------------------------------------------------

class TestMidnightCrossing:
    def test_alert_split_across_two_days(self):
        # 23:30 → 00:45 next day: 30 min on day 1, 45 min on day 2
        events = make_events(
            ("Kyiv City", ts("2024-01-01 23:30"), ts("2024-01-02 00:45"), False),
        )
        daily = resample_daily(events)
        region = daily.xs("Kyiv City", level="region")

        assert region.loc[ts("2024-01-01"), "alerts_count"] == 1
        assert region.loc[ts("2024-01-01"), "alert_minutes"] == pytest.approx(30.0)

        assert region.loc[ts("2024-01-02"), "alerts_count"] == 1
        assert region.loc[ts("2024-01-02"), "alert_minutes"] == pytest.approx(45.0)

    def test_minutes_sum_to_total_duration(self):
        # Total duration: 23:30 → 01:30 = 120 min over 2 days
        events = make_events(
            ("Kyiv City", ts("2024-01-01 23:30"), ts("2024-01-02 01:30"), False),
        )
        daily = resample_daily(events)
        region = daily.xs("Kyiv City", level="region")

        total_minutes = region["alert_minutes"].sum()
        assert total_minutes == pytest.approx(120.0)

    def test_exactly_at_midnight(self):
        # Alert ending exactly at midnight: all minutes go to day 1
        events = make_events(
            ("Kyiv City", ts("2024-01-01 23:00"), ts("2024-01-02 00:00"), False),
        )
        daily = resample_daily(events)
        region = daily.xs("Kyiv City", level="region")

        assert region.loc[ts("2024-01-01"), "alert_minutes"] == pytest.approx(60.0)
        # Day 2 entry may exist with 0 minutes or not at all — either is fine
        if ts("2024-01-02") in region.index:
            assert region.loc[ts("2024-01-02"), "alert_minutes"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Zero-fill for missing days
# ---------------------------------------------------------------------------

class TestZeroFill:
    def test_gap_days_are_zero(self):
        events = make_events(
            ("Kyiv City", ts("2024-01-01 10:00"), ts("2024-01-01 10:30"), False),
            ("Kyiv City", ts("2024-01-05 10:00"), ts("2024-01-05 10:30"), False),
        )
        daily = resample_daily(events)
        region = daily.xs("Kyiv City", level="region")

        for gap_day in ["2024-01-02", "2024-01-03", "2024-01-04"]:
            row = region.loc[ts(gap_day)]
            assert row["alerts_count"] == 0
            assert row["alert_minutes"] == pytest.approx(0.0)

    def test_continuous_index_no_holes(self):
        events = make_events(
            ("Kyiv City", ts("2024-01-01 10:00"), ts("2024-01-01 10:30"), False),
            ("Kyiv City", ts("2024-01-10 10:00"), ts("2024-01-10 10:30"), False),
        )
        daily = resample_daily(events)
        region = daily.xs("Kyiv City", level="region")

        # Expect exactly 10 dates (Jan 1 through Jan 10)
        assert len(region) == 10


# ---------------------------------------------------------------------------
# Multi-region
# ---------------------------------------------------------------------------

class TestMultiRegion:
    def test_regions_independent(self):
        events = make_events(
            ("Kyiv City", ts("2024-01-01 10:00"), ts("2024-01-01 11:00"), False),
            ("Lviv",      ts("2024-01-01 10:00"), ts("2024-01-01 10:30"), False),
        )
        daily = resample_daily(events)

        kyiv = daily.xs("Kyiv City", level="region").loc[ts("2024-01-01")]
        lviv = daily.xs("Lviv",      level="region").loc[ts("2024-01-01")]

        assert kyiv["alert_minutes"] == pytest.approx(60.0)
        assert lviv["alert_minutes"] == pytest.approx(30.0)

    def test_both_regions_present_in_output(self):
        events = make_events(
            ("Kyiv City", ts("2024-01-01 10:00"), ts("2024-01-01 11:00"), False),
            ("Lviv",      ts("2024-01-02 10:00"), ts("2024-01-02 10:30"), False),
        )
        daily = resample_daily(events)
        regions = set(daily.index.get_level_values("region"))
        assert regions == {"Kyiv City", "Lviv"}


# ---------------------------------------------------------------------------
# Empty input
# ---------------------------------------------------------------------------

class TestEmpty:
    def test_empty_events_returns_empty_dataframe(self):
        events = pd.DataFrame(
            columns=["region", "started_at", "finished_at", "naive", "source"]
        )
        daily = resample_daily(events)
        assert daily.empty