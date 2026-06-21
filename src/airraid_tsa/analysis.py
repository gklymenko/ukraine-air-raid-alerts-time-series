"""EDA: decomposition, weekday/hour patterns, duration distributions, structural breaks.

Public API
----------
daily_series_filled(events, region)  -> pd.DataFrame  (DatetimeIndex, zero-filled)
decompose(series, period, region)    -> DecomposeResult | None
weekday_profile(series)              -> pd.Series  (index 0–6)
hour_of_day_profile(events, region)  -> pd.Series  (index 0–23)
duration_distribution(events, region)-> tuple[pd.Series, pd.Series]
structural_break_view(series)        -> pd.Series  (30-day rolling mean)
"""

from __future__ import annotations

import pandas as pd
from statsmodels.tsa.seasonal import DecomposeResult, seasonal_decompose

from airraid_tsa.config import ALWAYS_ON_REGIONS, BREAK_NOTE_2025
from airraid_tsa.resample import resample_daily


# ---------------------------------------------------------------------------
# Daily series
# ---------------------------------------------------------------------------

def daily_series_filled(events: pd.DataFrame, region: str) -> pd.DataFrame:
    """Zero-filled daily DataFrame for one region, built from raw events.

    Delegates to resample_daily (which already zero-fills date gaps), then
    extracts the requested region.  A day with no alert events gets
    alerts_count=0 and alert_minutes=0.0 — NOT NaN — which is the
    precondition for seasonal decomposition.

    Parameters
    ----------
    events : pd.DataFrame   Canonical events (output of an ingest adapter).
    region : str            Oblast name to extract.

    Returns
    -------
    pd.DataFrame
        DatetimeIndex (UTC daily), columns: alerts_count (int), alert_minutes (float).

    Raises
    ------
    ValueError   If region is absent from events.
    """
    daily = resample_daily(events)
    available = daily.index.get_level_values("region").unique().tolist()
    if region not in available:
        raise ValueError(
            f"Region '{region}' not found in events. Available: {available}"
        )
    return daily.xs(region, level="region")


# ---------------------------------------------------------------------------
# Decomposition
# ---------------------------------------------------------------------------

def decompose(
    series: pd.Series,
    period: int = 7,
    region: str = "",
) -> DecomposeResult | None:
    """Additive seasonal decomposition of a daily time series.

    Skips decomposition — returning None and printing a note — when:
    - region is in the always-on set (near-constant, makes decomp meaningless)
    - series is effectively constant (std ≈ 0)
    - series is too short (need ≥ 2 × period observations)

    Parameters
    ----------
    series : pd.Series   Zero-filled daily series (no NaN).
    period : int         Seasonal period in days (7 = weekly).
    region : str         Oblast name used for always-on lookup (optional).

    Returns
    -------
    DecomposeResult or None
    """
    if region in ALWAYS_ON_REGIONS:
        print(f"  [decompose] Skipping '{region}' — in always-on region set.")
        return None

    if series.std() < 1e-6:
        print(f"  [decompose] Skipping '{region}' — series is constant (std ≈ 0).")
        return None

    min_obs = 2 * period
    if len(series) < min_obs:
        print(
            f"  [decompose] Skipping: need ≥ {min_obs} observations, "
            f"got {len(series)}."
        )
        return None

    # extrapolate_trend fills NaN at the edges of the trend component so
    # the downstream plots have a complete line without gaps.
    return seasonal_decompose(
        series,
        model="additive",
        period=period,
        extrapolate_trend="freq",
    )


# ---------------------------------------------------------------------------
# Weekday profile
# ---------------------------------------------------------------------------

def weekday_profile(series: pd.Series) -> pd.Series:
    """Mean value of a daily series grouped by day of week.

    Parameters
    ----------
    series : pd.Series   Daily series with a DatetimeIndex.

    Returns
    -------
    pd.Series
        Index: 0–6 (Monday = 0, Sunday = 6).
        Values: mean of the input series on each day of week.
    """
    return series.groupby(series.index.dayofweek).mean()


# ---------------------------------------------------------------------------
# Hour-of-day profile
# ---------------------------------------------------------------------------

def hour_of_day_profile(events: pd.DataFrame, region: str) -> pd.Series:
    """Alert count per UTC start-hour for a region.

    Derived from started_at timestamps, not the daily series.
    All 24 hours are represented (zero-filled).

    Parameters
    ----------
    events : pd.DataFrame   Canonical events.
    region : str            Oblast to filter on.

    Returns
    -------
    pd.Series
        Index: 0–23 (UTC hour), values: number of alert starts.
    """
    region_starts = events.loc[events["region"] == region, "started_at"]
    hour_counts = region_starts.dt.hour.value_counts()
    return hour_counts.reindex(range(24), fill_value=0).sort_index()


# ---------------------------------------------------------------------------
# Duration distribution
# ---------------------------------------------------------------------------

def duration_distribution(
    events: pd.DataFrame,
    region: str,
) -> tuple[pd.Series, pd.Series]:
    """Alert durations in minutes for a region.

    Returns two series so callers can compare the true measured durations
    (non-naive rows) against the full set that includes the 30-min estimates.

    Parameters
    ----------
    events : pd.DataFrame   Canonical events. Naive rows already have
                            finished_at = started_at + 30 min (set by ingest).
    region : str            Oblast to filter on.

    Returns
    -------
    (all_durations, non_naive_durations) : tuple[pd.Series, pd.Series]
        Both series are in minutes with reset integer index.
    """
    region_df = events[events["region"] == region].copy()
    region_df["duration_min"] = (
        (region_df["finished_at"] - region_df["started_at"]).dt.total_seconds() / 60.0
    )
    all_dur = region_df["duration_min"].reset_index(drop=True)
    non_naive = region_df.loc[~region_df["naive"], "duration_min"].reset_index(drop=True)
    return all_dur, non_naive


# ---------------------------------------------------------------------------
# Structural break view
# ---------------------------------------------------------------------------

def structural_break_view(series: pd.Series) -> pd.Series:
    """Compute a 30-day rolling mean and print the 2025 structural-break note.

    The rolling mean is intended to be overlaid on the raw series in a plot to
    make any level shifts (including the 2025 methodology change) visually clear.
    No statistical changepoint algorithm is applied.

    Parameters
    ----------
    series : pd.Series   Full-range daily series with a DatetimeIndex.

    Returns
    -------
    pd.Series
        30-day rolling mean.  min_periods=7 so the first week still has a value.
    """
    print("\n[STRUCTURAL BREAK NOTE]")
    print(f"  {BREAK_NOTE_2025}")
    return series.rolling(window=30, min_periods=7).mean()