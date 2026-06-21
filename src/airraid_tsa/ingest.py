"""Ingest adapters: raw CSV files -> canonical events DataFrame.

Canonical schema (both adapters must produce exactly these columns)
-------------------------------------------------------------------
region       : str               — oblast name
started_at   : datetime64[UTC]
finished_at  : datetime64[UTC]
naive        : bool              — True when finished_at was estimated
source       : str               — "official" | "volunteer"
geo_level    : str               — always "oblast" in the MVP

Public API
----------
load_official_csv(path)   -> pd.DataFrame   OfficialOblastCsvAdapter
load_volunteer_csv(path)  -> pd.DataFrame   VolunteerCsvAdapter
"""

from pathlib import Path

import pandas as pd

from airraid_tsa.config import NAIVE_DURATION_MINUTES

# Canonical column order produced by every adapter.
_CANONICAL: list[str] = [
    "region", "started_at", "finished_at", "naive", "source", "geo_level"
]

# Columns required in the official source file.
_OFFICIAL_REQUIRED: list[str] = [
    "oblast", "level", "started_at", "finished_at"
]

# Columns required in the volunteer source file.
_VOLUNTEER_REQUIRED: list[str] = [
    "region", "started_at", "finished_at", "naive"
]


# ---------------------------------------------------------------------------
# OfficialOblastCsvAdapter  (primary source)
# ---------------------------------------------------------------------------

def load_official_csv(path: Path) -> pd.DataFrame:
    """Load official_data_en.csv, keep only oblast-level rows, return canonical events.

    Performs:
    - Level filter: drops raion / hromada rows.
    - Rename: oblast -> region.
    - Constant assignments: naive=False (no such signal), source="official",
      geo_level="oblast".
    - Dedup: removes duplicate (region, started_at, finished_at) triples that
      appear in the official file since the Dec-2025 schema change.

    Parameters
    ----------
    path : Path   Location of official_data_en.csv.

    Returns
    -------
    pd.DataFrame  Canonical events.

    Raises
    ------
    FileNotFoundError   If the file does not exist.
    ValueError          If required columns are absent or timestamps cannot be parsed.
    """
    if not path.exists():
        raise FileNotFoundError(f"Official CSV not found: {path}")

    df = pd.read_csv(path)
    _assert_columns(df, _OFFICIAL_REQUIRED, path)

    # Filter to oblast level and assign canonical columns — dedup comes next.
    oblast_raw = (
        df.loc[df["level"].eq("oblast")]
        .rename(columns={"oblast": "region"})
        .assign(naive=False, source="official", geo_level="oblast")
        [["region", "started_at", "finished_at", "naive", "source", "geo_level"]]
        .copy()
    )

    rows_before = len(oblast_raw)
    deduped = oblast_raw.drop_duplicates(subset=["region", "started_at", "finished_at"])
    rows_after = len(deduped)
    duplicates_removed = rows_before - rows_after

    deduped = _parse_timestamps(deduped.copy())
    deduped["region"] = deduped["region"].astype(str).str.strip()
    result = deduped.reset_index(drop=True)

    # Store dedup stats as DataFrame metadata so callers can surface them.
    pct = duplicates_removed / rows_before * 100 if rows_before > 0 else 0.0
    result.attrs["rows_before_dedup"] = rows_before
    result.attrs["rows_after_dedup"] = rows_after
    result.attrs["duplicates_removed"] = duplicates_removed
    print(
        f"[ingest] Official-oblast: {rows_before:,} raw rows → "
        f"{rows_after:,} after dedup (removed {duplicates_removed:,}, {pct:.1f}%)"
    )
    return result


# ---------------------------------------------------------------------------
# VolunteerCsvAdapter  (secondary, cross-check only)
# ---------------------------------------------------------------------------

def load_volunteer_csv(path: Path) -> pd.DataFrame:
    """Load volunteer_data_en.csv into the canonical events DataFrame.

    Used only for quality cross-checking (§6).  Never merged into the
    analysis series — call load_official_csv for that.

    Parameters
    ----------
    path : Path   Location of volunteer_data_en.csv.

    Returns
    -------
    pd.DataFrame  Canonical events with source="volunteer", geo_level="oblast".

    Raises
    ------
    FileNotFoundError   If the file does not exist.
    ValueError          If required columns are absent or timestamps cannot be parsed.
    """
    if not path.exists():
        raise FileNotFoundError(f"Volunteer CSV not found: {path}")

    df = pd.read_csv(path)
    _assert_columns(df, _VOLUNTEER_REQUIRED, path)

    volunteer = df[["region", "started_at", "finished_at", "naive"]].copy()
    volunteer["source"] = "volunteer"
    volunteer["geo_level"] = "oblast"

    volunteer = _parse_timestamps(volunteer)
    volunteer["region"] = volunteer["region"].astype(str).str.strip()
    volunteer = _fix_naive_alerts(volunteer)
    return volunteer[_CANONICAL].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _assert_columns(df: pd.DataFrame, required: list[str], path: Path) -> None:
    """Raise ValueError if any required column is absent."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns {missing} in {path}. Found: {list(df.columns)}"
        )


def _parse_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Parse started_at and finished_at as timezone-aware UTC datetimes in-place."""
    for col in ("started_at", "finished_at"):
        try:
            df[col] = pd.to_datetime(df[col], utc=True)
        except Exception as exc:
            raise ValueError(
                f"Cannot parse column '{col}' as datetime: {exc}"
            ) from exc
    return df


def _fix_naive_alerts(df: pd.DataFrame) -> pd.DataFrame:
    """For naive rows, set finished_at = started_at + NAIVE_DURATION_MINUTES."""
    naive_mask = df["naive"].astype(bool)
    df.loc[naive_mask, "finished_at"] = (
        df.loc[naive_mask, "started_at"]
        + pd.Timedelta(minutes=NAIVE_DURATION_MINUTES)
    )
    df["naive"] = naive_mask
    return df