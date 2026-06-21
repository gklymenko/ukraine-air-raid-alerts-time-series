"""Data-quality and freshness verification.

Produces a printed report covering:
- Freshness: newest event vs today.
- Schema: required columns present and parseable.
- Coverage: date range, row count, % naive per region.
- Sanity: region names within known oblast set; always-on flags.
- 2025 structural-break note.

Usage
-----
    from airraid_tsa.verify import run_quality_report
    run_quality_report(df)
"""

from datetime import datetime, timezone

import pandas as pd

from airraid_tsa.config import (
    ALWAYS_ON_HOURS_THRESHOLD,
    ALWAYS_ON_REGIONS,
    BREAK_NOTE_2025,
    FRESHNESS_WARN_DAYS,
    OBLAST_SET,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_quality_report(df: pd.DataFrame) -> None:
    """Print a data-quality report for the canonical events DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Canonical events (output of an ingest adapter).

    Raises
    ------
    ValueError
        If the DataFrame has critical schema problems (wrong column types, empty).
    """
    print("=" * 60)
    print("DATA QUALITY REPORT")
    print("=" * 60)

    _check_not_empty(df)
    _check_schema(df)
    _check_freshness(df)
    _check_coverage(df)
    _check_oblast_names(df)
    _check_always_on(df)
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


def _check_schema(df: pd.DataFrame) -> None:
    """Verify column dtypes are sensible after ingestion."""
    issues: list[str] = []

    if not pd.api.types.is_datetime64_any_dtype(df["started_at"]):
        issues.append("started_at is not datetime")
    if not pd.api.types.is_datetime64_any_dtype(df["finished_at"]):
        issues.append("finished_at is not datetime")

    # finished_at must be >= started_at for all rows.
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
    total_rows = len(df)
    regions = df["region"].nunique()

    print(f"\n[COVERAGE]")
    print(f"  Date range   : {start.date()} — {end.date()}")
    print(f"  Total events : {total_rows:,}")
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


def _check_oblast_names(df: pd.DataFrame) -> None:
    """Flag region names that are not in the known oblast set."""
    found_regions = set(df["region"].unique())
    unknown = found_regions - OBLAST_SET

    print(f"\n[OBLAST NAMES]")
    if unknown:
        print(
            f"  WARNING: {len(unknown)} region name(s) not in known oblast list: "
            f"{sorted(unknown)}"
        )
        print(
            "  This may indicate a naming variant or a new district-level entry. "
            "Check config.OBLAST_LIST."
        )
    else:
        print(f"  All {len(found_regions)} region name(s) match the known oblast list.")


def _check_always_on(df: pd.DataFrame) -> None:
    """Flag regions with very long individual alert durations."""
    df2 = df.copy()
    df2["duration_hours"] = (
        (df2["finished_at"] - df2["started_at"]).dt.total_seconds() / 3600
    )

    long_alerts = df2[df2["duration_hours"] > ALWAYS_ON_HOURS_THRESHOLD]
    affected_regions = long_alerts["region"].unique()

    print(f"\n[ALWAYS-ON CHECK]  (threshold: >{ALWAYS_ON_HOURS_THRESHOLD}h per alert)")
    if len(affected_regions) == 0:
        print("  No suspiciously long individual alerts found.")
    else:
        print(
            f"  WARNING: {len(long_alerts)} alert(s) exceed {ALWAYS_ON_HOURS_THRESHOLD}h "
            f"in regions: {sorted(affected_regions)}"
        )
        print(
            "  Known always-on candidates from config: "
            + (", ".join(sorted(ALWAYS_ON_REGIONS)) or "none")
        )
        print(
            "  These regions may be degenerate for volume forecasting — "
            "consider excluding or treating separately."
        )


def _print_break_note() -> None:
    """Print the 2025 structural-break advisory."""
    print(f"\n[2025 STRUCTURAL BREAK NOTE]")
    print(f"  {BREAK_NOTE_2025}")