"""Ingest adapters: raw CSV files -> canonical events DataFrame.

Canonical schema
----------------
region       : str         — oblast name
started_at   : datetime64[UTC]
finished_at  : datetime64[UTC]
naive        : bool        — True when finished_at was estimated (no end signal)
source       : str         — identifier of the data source
"""

from pathlib import Path

import pandas as pd

from airraid_tsa.config import NAIVE_DURATION_MINUTES, REQUIRED_COLUMNS


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_vadimkin_csv(path: Path) -> pd.DataFrame:
    """Load Vadimkin official_data_en.csv into the canonical events DataFrame.

    Parameters
    ----------
    path : Path
        Location of the CSV file.

    Returns
    -------
    pd.DataFrame
        Canonical events with columns:
        region, started_at (UTC), finished_at (UTC), naive, source.

    Raises
    ------
    FileNotFoundError
        If the CSV does not exist at *path*.
    ValueError
        If required columns are missing or timestamps cannot be parsed.
    """
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_csv(path)
    _validate_columns(df, path)

    df = _parse_timestamps(df)
    df = _fix_naive_alerts(df)
    df["source"] = "vadimkin_official"

    # Keep only canonical columns, in order.
    return df[["region", "started_at", "finished_at", "naive", "source"]].copy()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_columns(df: pd.DataFrame, path: Path) -> None:
    """Raise ValueError if any required column is absent."""
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns {missing} in {path}. "
            f"Found: {list(df.columns)}"
        )


def _parse_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Parse started_at and finished_at as timezone-aware UTC datetimes."""
    for col in ("started_at", "finished_at"):
        try:
            parsed = pd.to_datetime(df[col], utc=True)
        except Exception as exc:
            raise ValueError(f"Cannot parse column '{col}' as datetime: {exc}") from exc
        df[col] = parsed

    # Ensure region is a plain string (strip whitespace).
    df["region"] = df["region"].astype(str).str.strip()

    return df


def _fix_naive_alerts(df: pd.DataFrame) -> pd.DataFrame:
    """For naive alerts, set finished_at = started_at + NAIVE_DURATION_MINUTES.

    The source CSV marks rows with naive=True when no end signal was received.
    We do not overwrite a finished_at that is already later than started_at for
    non-naive rows — the raw value is trusted as-is.
    """
    naive_mask = df["naive"].astype(bool)

    # Replace finished_at only for naive rows.
    df.loc[naive_mask, "finished_at"] = (
        df.loc[naive_mask, "started_at"]
        + pd.Timedelta(minutes=NAIVE_DURATION_MINUTES)
    )

    df["naive"] = naive_mask  # ensure bool dtype
    return df