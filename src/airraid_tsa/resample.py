"""Resample canonical events into a daily per-oblast time series.

Output columns (indexed by date)
---------------------------------
alerts_count  : int   — number of alerts that were active on this day
alert_minutes : float — total minutes of alert coverage on this day

Key correctness points
----------------------
- An alert that crosses midnight is split: each calendar day (UTC) receives
  only the minutes that fall within that day.
- Naive alerts already have finished_at = started_at + 30 min by the time
  they reach this module (set by ingest._fix_naive_alerts).
- The date index is continuous from the first to the last event date
  (missing days filled with zeros).
"""

from __future__ import annotations

import pandas as pd


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def resample_daily(events: pd.DataFrame) -> pd.DataFrame:
    """Convert a canonical events DataFrame into a daily per-oblast series.

    Parameters
    ----------
    events : pd.DataFrame
        Canonical events (output of an ingest adapter).
        Must have: region, started_at (UTC), finished_at (UTC).

    Returns
    -------
    pd.DataFrame
        MultiIndex (region, date) with columns:
        alerts_count (int), alert_minutes (float).
        The date index is a continuous DatetimeIndex (daily, UTC midnight)
        with zero-filled gaps.
    """
    if events.empty:
        return pd.DataFrame(
            columns=["region", "date", "alerts_count", "alert_minutes"]
        ).set_index(["region", "date"])

    # Expand each event into per-day contribution rows.
    rows = [_split_alert_by_day(row) for row in events.itertuples(index=False)]
    daily_pieces = pd.concat(rows, ignore_index=True)

    # Aggregate per (region, date).
    grouped = (
        daily_pieces.groupby(["region", "date"])
        .agg(alerts_count=("alerts_count", "sum"), alert_minutes=("alert_minutes", "sum"))
    )

    # Fill in missing calendar days with zeros so every region has a
    # continuous daily series (makes downstream resampling and lag features easy).
    grouped = _fill_date_gaps(grouped)
    grouped["alerts_count"] = grouped["alerts_count"].astype(int)
    return grouped


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _split_alert_by_day(row: object) -> pd.DataFrame:
    """Return a DataFrame with one row per calendar day this alert spans.

    Each row contains:
    - region       : str
    - date         : pd.Timestamp (UTC midnight of the day)
    - alerts_count : 1  (this alert touched this day)
    - alert_minutes: float (minutes of coverage within that day only)
    """
    start: pd.Timestamp = row.started_at  # type: ignore[attr-defined]
    end: pd.Timestamp = row.finished_at  # type: ignore[attr-defined]
    region: str = row.region  # type: ignore[attr-defined]

    # Floor to UTC calendar days.
    day_start = start.normalize()   # midnight at or before start
    day_end = end.normalize()       # midnight at or before end

    results: list[dict] = []

    current_day = day_start
    while current_day <= day_end:
        next_day = current_day + pd.Timedelta(days=1)

        # Clamp the alert interval to [current_day, next_day).
        interval_start = max(start, current_day)
        interval_end = min(end, next_day)

        minutes = (interval_end - interval_start).total_seconds() / 60.0

        if minutes > 0:
            results.append(
                {
                    "region": region,
                    "date": current_day,
                    "alerts_count": 1,
                    "alert_minutes": minutes,
                }
            )

        current_day = next_day

    return pd.DataFrame(results)


def _fill_date_gaps(grouped: pd.DataFrame) -> pd.DataFrame:
    """Reindex each region to a continuous daily date range, filling zeros."""
    all_dates = grouped.index.get_level_values("date")
    date_range = pd.date_range(
        start=all_dates.min(),
        end=all_dates.max(),
        freq="D",
        tz="UTC",
    )

    filled_pieces: list[pd.DataFrame] = []
    for region in grouped.index.get_level_values("region").unique():
        region_df = grouped.xs(region, level="region").reindex(date_range, fill_value=0)
        region_df.index.name = "date"
        region_df["region"] = region
        filled_pieces.append(region_df.reset_index().set_index(["region", "date"]))

    return pd.concat(filled_pieces).sort_index()