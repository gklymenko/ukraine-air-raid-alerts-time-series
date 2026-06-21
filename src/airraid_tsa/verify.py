"""Data-quality and freshness verification.

Produces a printed report covering:
- Freshness: newest event vs today.
- Schema: required columns present and parseable; no inverted durations.
- Coverage: date range, row count, % naive per region.
- Geo-level assertion: every canonical row has geo_level == "oblast".
- Always-on regions: flagged by a duration heuristic, not hardcoded names.
- Cross-source check (optional): monthly comparison of official vs volunteer
  for the focal oblast, plus a post-Dec-2025 continuity check.
- 2025 structural-break note.

Usage
-----
    from airraid_tsa.verify import run_quality_report
    run_quality_report(events)                          # official only
    run_quality_report(events, volunteer_events=vol)    # with cross-check
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from airraid_tsa.config import (
    ALWAYS_ON_HOURS_THRESHOLD,
    BREAK_NOTE_2025,
    FOCAL_OBLAST,
    FRESHNESS_WARN_DAYS,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_quality_report(
    events: pd.DataFrame,
    volunteer_events: pd.DataFrame | None = None,
    ingest_stats: dict | None = None,
) -> None:
    """Print a data-quality report for the canonical events DataFrame.

    Parameters
    ----------
    events            : pd.DataFrame   Official-oblast canonical events.
    volunteer_events  : pd.DataFrame   Volunteer canonical events (optional).
                                       When supplied, a monthly cross-source
                                       comparison is printed.
    ingest_stats      : dict           Dedup stats from load_official_csv (optional).
                                       Keys: rows_before_dedup, rows_after_dedup,
                                       duplicates_removed.

    Raises
    ------
    ValueError   If the DataFrame has critical schema problems (wrong types, empty).
    """
    print("=" * 60)
    print("DATA QUALITY REPORT")
    print("=" * 60)

    _check_not_empty(events)
    _check_schema(events)
    _check_dedup(events, ingest_stats)
    _check_freshness(events)
    _check_coverage(events)
    _check_geo_level(events)
    _check_always_on(events)

    if volunteer_events is not None and not volunteer_events.empty:
        _check_cross_source(events, volunteer_events, FOCAL_OBLAST)

    _print_break_note()

    print("=" * 60)
    print("Report complete.")
    print()


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def _check_not_empty(df: pd.DataFrame) -> None:
    if df.empty:
        raise ValueError("Events DataFrame is empty — nothing to analyse.")
    print(f"\n[OK] Row count: {len(df):,}")


def _check_dedup(df: pd.DataFrame, ingest_stats: dict | None) -> None:
    """Report dedup stats and assert no duplicate keys remain after ingestion.

    Prints: "official adapter: kept N oblast events, removed M exact duplicates (~P%)".
    Raises ValueError if any (region, started_at, finished_at) duplicates are found.

    Parameters
    ----------
    df           : pd.DataFrame   Canonical events (already deduped by adapter).
    ingest_stats : dict | None    Stats from load_official_csv (.attrs), or None.
    """
    print(f"\n[DEDUP]")

    if ingest_stats:
        kept = ingest_stats.get("rows_after_dedup", len(df))
        removed = ingest_stats.get("duplicates_removed", 0)
        before = ingest_stats.get("rows_before_dedup", kept + removed)
        pct = removed / before * 100 if before > 0 else 0.0
        print(
            f"  Official adapter: kept {kept:,} oblast events, "
            f"removed {removed:,} exact duplicates ({pct:.1f}%)"
        )

    # Post-ingestion uniqueness assertion — fail loudly if adapter dedup broke.
    dup_count = df.duplicated(subset=["region", "started_at", "finished_at"]).sum()
    if dup_count > 0:
        raise ValueError(
            f"{dup_count} duplicate (region, started_at, finished_at) keys remain "
            "after ingestion — the adapter dedup step may have failed."
        )
    print("  [OK] No duplicate (region, started_at, finished_at) keys after ingestion.")


def _check_schema(df: pd.DataFrame) -> None:
    """Verify column dtypes are sensible after ingestion."""
    issues: list[str] = []

    if not pd.api.types.is_datetime64_any_dtype(df["started_at"]):
        issues.append("started_at is not datetime")
    if not pd.api.types.is_datetime64_any_dtype(df["finished_at"]):
        issues.append("finished_at is not datetime")

    bad_duration = (df["finished_at"] < df["started_at"]).sum()
    if bad_duration > 0:
        issues.append(f"{bad_duration} rows have finished_at < started_at")

    if issues:
        raise ValueError("Schema problems: " + "; ".join(issues))

    print("[OK] Schema looks valid (timestamps parse, no inverted durations).")


def _check_freshness(df: pd.DataFrame) -> None:
    """Warn if the newest event is older than FRESHNESS_WARN_DAYS."""
    newest: pd.Timestamp = df["started_at"].max()
    now = pd.Timestamp(datetime.now(timezone.utc))
    age_days = (now - newest).days

    print(f"\n[FRESHNESS]")
    print(f"  Newest event : {newest.strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"  Today        : {now.strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"  Data age     : {age_days} day(s)")

    if age_days > FRESHNESS_WARN_DAYS:
        print(
            f"  WARNING: Data is {age_days} day(s) old "
            f"(threshold: {FRESHNESS_WARN_DAYS}). "
            "Refresh official_data_en.csv before drawing conclusions."
        )
    else:
        print("  Data is fresh.")


def _check_coverage(df: pd.DataFrame) -> None:
    """Print date range, region count, and % naive per region."""
    start = df["started_at"].min()
    end = df["started_at"].max()
    regions = df["region"].nunique()

    print(f"\n[COVERAGE]")
    print(f"  Date range   : {start.date()} — {end.date()}")
    print(f"  Total events : {len(df):,}")
    print(f"  Regions      : {regions}")

    naive_by_region = (
        df.groupby("region")["naive"]
        .agg(total="count", naive_count="sum")
        .assign(pct_naive=lambda x: 100 * x["naive_count"] / x["total"])
        .sort_values("pct_naive", ascending=False)
    )

    print("\n  % naive alerts by region (top 10):")
    print(
        naive_by_region.head(10)
        .to_string(
            columns=["total", "naive_count", "pct_naive"],
            header=["  total", "  naive", "  % naive"],
            float_format="%.1f",
        )
    )


def _check_geo_level(df: pd.DataFrame) -> None:
    """Assert every canonical row has geo_level == 'oblast' (level filter check)."""
    print(f"\n[GEO LEVEL]")

    if "geo_level" not in df.columns:
        print("  WARNING: geo_level column missing from events DataFrame.")
        return

    non_oblast = df.loc[df["geo_level"] != "oblast"]
    if not non_oblast.empty:
        breakdown = non_oblast["geo_level"].value_counts().to_dict()
        raise ValueError(
            f"{len(non_oblast)} canonical rows have geo_level != 'oblast': {breakdown}. "
            "The level filter in OfficialOblastCsvAdapter may have failed."
        )

    known_regions = set(df["region"].dropna().unique())
    print(f"  All {len(df):,} rows have geo_level='oblast'.")
    print(f"  Unique regions in data: {len(known_regions)}")


def _check_always_on(df: pd.DataFrame) -> None:
    """Flag regions with very long individual alert durations (heuristic, no hardcoded names)."""
    df2 = df.copy()
    df2["duration_hours"] = (
        (df2["finished_at"] - df2["started_at"]).dt.total_seconds() / 3600
    )

    long_alerts = df2[df2["duration_hours"] > ALWAYS_ON_HOURS_THRESHOLD]
    affected = long_alerts["region"].value_counts()

    print(f"\n[ALWAYS-ON CHECK]  (threshold: >{ALWAYS_ON_HOURS_THRESHOLD}h per alert)")
    if affected.empty:
        print("  No suspiciously long individual alerts found.")
    else:
        print(
            f"  WARNING: {len(long_alerts)} alert(s) exceed {ALWAYS_ON_HOURS_THRESHOLD}h "
            f"across {len(affected)} region(s):"
        )
        for region, count in affected.head(10).items():
            print(f"    {region}: {count} long alert(s)")
        print(
            "  These regions may be degenerate for volume forecasting — "
            "consider excluding or treating separately."
        )


# ---------------------------------------------------------------------------
# Cross-source comparison
# ---------------------------------------------------------------------------

def monthly_event_counts(events: pd.DataFrame, region: str) -> pd.Series:
    """Monthly alert-start counts for a single region.

    Parameters
    ----------
    events : pd.DataFrame   Canonical events.
    region : str            Oblast name.

    Returns
    -------
    pd.Series   Index = period (YYYY-MM), values = event count.
    """
    mask = events["region"] == region
    starts = events.loc[mask, "started_at"]
    if starts.empty:
        return pd.Series(dtype=int)
    # to_period drops tz info (periods have no tz); normalize first to avoid warning.
    return starts.dt.tz_localize(None).dt.to_period("M").value_counts().sort_index().rename(region)


def _check_cross_source(
    official: pd.DataFrame,
    volunteer: pd.DataFrame,
    region: str,
    divergence_pct: float = 50.0,
) -> None:
    """Monthly cross-source comparison for one region.

    Compares monthly event counts between official-oblast and volunteer
    sources. Reports months where they diverge by more than divergence_pct
    percent.  Also checks that official-oblast data continues after Dec 2025
    (confirming oblast rows did not thin out after the methodology change).

    Parameters
    ----------
    official       : pd.DataFrame   Official-oblast canonical events.
    volunteer      : pd.DataFrame   Volunteer canonical events.
    region         : str            Oblast to compare (e.g. "Kyiv City").
    divergence_pct : float          Alert threshold (default 50 %).
    """
    print(f"\n[CROSS-SOURCE CHECK]  focal oblast: {region}")

    off_counts = monthly_event_counts(official, region)
    vol_counts = monthly_event_counts(volunteer, region)

    if off_counts.empty:
        print(f"  WARNING: No official data found for '{region}'.")
        return
    if vol_counts.empty:
        print(f"  WARNING: No volunteer data found for '{region}'.")
        return

    # Align on the months both sources cover.
    combined = pd.DataFrame({"official": off_counts, "volunteer": vol_counts}).dropna()

    if combined.empty:
        print("  No overlapping months between official and volunteer sources.")
        return

    combined["pct_diff"] = (
        (combined["official"] - combined["volunteer"]).abs()
        / combined[["official", "volunteer"]].max(axis=1)
        * 100
    )

    large = combined[combined["pct_diff"] > divergence_pct]
    n_overlap = len(combined)

    print(f"  Overlapping months: {n_overlap}")
    print(f"  Divergence threshold: >{divergence_pct:.0f}%")

    if large.empty:
        print(f"  [OK] No months with large divergence — sources agree well.")
    else:
        print(f"  WARNING: {len(large)} month(s) with >{divergence_pct:.0f}% divergence:")
        print(
            large[["official", "volunteer", "pct_diff"]]
            .rename(columns={"pct_diff": "% diff"})
            .to_string(float_format="%.1f")
        )

    # Post-Dec-2025 continuity check for official source.
    cutoff = pd.Period("2025-12", freq="M")
    recent_official = off_counts[off_counts.index > cutoff]
    if off_counts.index.max() <= cutoff:
        print(
            f"\n  INFO: Official-oblast data ends at {off_counts.index.max()} — "
            "no post-Dec-2025 months to check."
        )
    elif recent_official.empty or recent_official.sum() == 0:
        print(
            f"\n  WARNING: Official-oblast events for '{region}' thin out after Dec-2025 "
            "— check whether the level filter is excluding post-transition rows."
        )
    else:
        n_recent = len(recent_official)
        print(
            f"\n  [OK] Official-oblast rows continue after Dec-2025 "
            f"({n_recent} month(s) present for '{region}')."
        )


def _print_break_note() -> None:
    """Print the 2025 structural-break advisory."""
    print(f"\n[2025 STRUCTURAL BREAK NOTE]")
    print(f"  {BREAK_NOTE_2025}")