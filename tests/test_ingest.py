"""Tests for ingest.py adapters and verify.py cross-source check."""

import io
from pathlib import Path

import pandas as pd
import pytest

from airraid_tsa.ingest import load_official_csv, load_volunteer_csv
from airraid_tsa.verify import monthly_event_counts


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz="UTC")


def write_official_csv(tmp_path: Path, rows: list[dict]) -> Path:
    """Write rows in official_data_en.csv format to a temp file."""
    df = pd.DataFrame(rows)
    path = tmp_path / "official_data_en.csv"
    df.to_csv(path, index=False)
    return path


def write_volunteer_csv(tmp_path: Path, rows: list[dict]) -> Path:
    """Write rows in volunteer_data_en.csv format to a temp file."""
    df = pd.DataFrame(rows)
    path = tmp_path / "volunteer_data_en.csv"
    df.to_csv(path, index=False)
    return path


def official_row(
    oblast: str,
    started: str,
    finished: str,
    level: str = "oblast",
    raion: str = "",
    hromada: str = "",
) -> dict:
    return {
        "oblast": oblast,
        "raion": raion,
        "hromada": hromada,
        "level": level,
        "started_at": started,
        "finished_at": finished,
        "source": "official",
    }


def volunteer_row(
    region: str,
    started: str,
    finished: str,
    naive: bool = False,
) -> dict:
    return {
        "region": region,
        "started_at": started,
        "finished_at": finished,
        "naive": naive,
    }


# ---------------------------------------------------------------------------
# OfficialOblastCsvAdapter
# ---------------------------------------------------------------------------

class TestOfficialAdapter:
    def test_level_filter_keeps_only_oblast(self, tmp_path):
        path = write_official_csv(tmp_path, [
            official_row("Kyiv City", "2024-01-01 10:00+00:00", "2024-01-01 11:00+00:00", "oblast"),
            official_row("Kyiv City", "2024-01-01 12:00+00:00", "2024-01-01 12:30+00:00", "raion", "Desnyanskyi"),
            official_row("Lviv",      "2024-01-01 09:00+00:00", "2024-01-01 09:30+00:00", "hromada", "Lviv r.", "Lviv h."),
        ])
        result = load_official_csv(path)
        assert len(result) == 1
        assert result.iloc[0]["region"] == "Kyiv City"

    def test_dedup_removes_duplicate_intervals(self, tmp_path):
        path = write_official_csv(tmp_path, [
            official_row("Kyiv City", "2024-01-01 10:00+00:00", "2024-01-01 11:00+00:00"),
            official_row("Kyiv City", "2024-01-01 10:00+00:00", "2024-01-01 11:00+00:00"),  # exact dup
            official_row("Kyiv City", "2024-01-02 10:00+00:00", "2024-01-02 11:00+00:00"),
        ])
        result = load_official_csv(path)
        assert len(result) == 2

    def test_naive_always_false(self, tmp_path):
        path = write_official_csv(tmp_path, [
            official_row("Kyiv City", "2024-01-01 10:00+00:00", "2024-01-01 11:00+00:00"),
        ])
        result = load_official_csv(path)
        assert result["naive"].all() == False
        assert result["naive"].dtype == bool

    def test_geo_level_is_oblast(self, tmp_path):
        path = write_official_csv(tmp_path, [
            official_row("Kyiv City", "2024-01-01 10:00+00:00", "2024-01-01 11:00+00:00"),
            official_row("Lviv",      "2024-01-01 12:00+00:00", "2024-01-01 13:00+00:00"),
        ])
        result = load_official_csv(path)
        assert (result["geo_level"] == "oblast").all()

    def test_source_is_official(self, tmp_path):
        path = write_official_csv(tmp_path, [
            official_row("Kyiv City", "2024-01-01 10:00+00:00", "2024-01-01 11:00+00:00"),
        ])
        result = load_official_csv(path)
        assert (result["source"] == "official").all()

    def test_canonical_columns_present(self, tmp_path):
        path = write_official_csv(tmp_path, [
            official_row("Kyiv City", "2024-01-01 10:00+00:00", "2024-01-01 11:00+00:00"),
        ])
        result = load_official_csv(path)
        expected = {"region", "started_at", "finished_at", "naive", "source", "geo_level"}
        assert expected.issubset(set(result.columns))

    def test_timestamps_are_utc(self, tmp_path):
        path = write_official_csv(tmp_path, [
            official_row("Kyiv City", "2024-01-01 10:00+00:00", "2024-01-01 11:00+00:00"),
        ])
        result = load_official_csv(path)
        assert str(result["started_at"].dt.tz) == "UTC"
        assert str(result["finished_at"].dt.tz) == "UTC"

    def test_oblast_renamed_from_raw(self, tmp_path):
        # The raw file has "oblast" column; adapter must rename it to "region".
        path = write_official_csv(tmp_path, [
            official_row("Kharkiv", "2024-01-01 08:00+00:00", "2024-01-01 09:00+00:00"),
        ])
        result = load_official_csv(path)
        assert "region" in result.columns
        assert "oblast" not in result.columns
        assert result.iloc[0]["region"] == "Kharkiv"

    def test_missing_required_column_raises(self, tmp_path):
        # File without "level" column should raise ValueError.
        df = pd.DataFrame([{"oblast": "Kyiv City",
                            "started_at": "2024-01-01 10:00+00:00",
                            "finished_at": "2024-01-01 11:00+00:00"}])
        path = tmp_path / "bad.csv"
        df.to_csv(path, index=False)
        with pytest.raises(ValueError, match="Missing columns"):
            load_official_csv(path)

    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_official_csv(tmp_path / "nonexistent.csv")

    def test_output_has_unique_keys(self, tmp_path):
        # Even with duplicates in the source, output must have no repeated keys.
        path = write_official_csv(tmp_path, [
            official_row("Kyiv City", "2024-01-01 10:00+00:00", "2024-01-01 11:00+00:00"),
            official_row("Kyiv City", "2024-01-01 10:00+00:00", "2024-01-01 11:00+00:00"),  # dup
            official_row("Kyiv City", "2024-01-02 10:00+00:00", "2024-01-02 11:00+00:00"),
        ])
        result = load_official_csv(path)
        dups = result.duplicated(subset=["region", "started_at", "finished_at"]).sum()
        assert dups == 0, f"Found {dups} duplicate (region, started_at, finished_at) keys"

    def test_duplicates_removed_gt_zero_when_source_has_duplicates(self, tmp_path):
        # One genuine duplicate in the source → duplicates_removed must be reported.
        path = write_official_csv(tmp_path, [
            official_row("Kyiv City", "2024-01-01 10:00+00:00", "2024-01-01 11:00+00:00"),
            official_row("Kyiv City", "2024-01-01 10:00+00:00", "2024-01-01 11:00+00:00"),  # dup
        ])
        result = load_official_csv(path)
        assert result.attrs["duplicates_removed"] > 0
        assert result.attrs["rows_before_dedup"] == 2
        assert result.attrs["rows_after_dedup"] == 1


# ---------------------------------------------------------------------------
# VolunteerCsvAdapter
# ---------------------------------------------------------------------------

class TestVolunteerAdapter:
    def test_source_is_volunteer(self, tmp_path):
        path = write_volunteer_csv(tmp_path, [
            volunteer_row("Kyiv City", "2024-01-01 10:00+00:00", "2024-01-01 11:00+00:00"),
        ])
        result = load_volunteer_csv(path)
        assert (result["source"] == "volunteer").all()

    def test_geo_level_is_oblast(self, tmp_path):
        path = write_volunteer_csv(tmp_path, [
            volunteer_row("Kyiv City", "2024-01-01 10:00+00:00", "2024-01-01 11:00+00:00"),
        ])
        result = load_volunteer_csv(path)
        assert (result["geo_level"] == "oblast").all()

    def test_naive_false_preserved(self, tmp_path):
        path = write_volunteer_csv(tmp_path, [
            volunteer_row("Kyiv City", "2024-01-01 10:00+00:00", "2024-01-01 11:00+00:00", naive=False),
        ])
        result = load_volunteer_csv(path)
        assert result.iloc[0]["naive"] == False

    def test_naive_true_preserved_and_finished_fixed(self, tmp_path):
        # naive=True rows must have finished_at overridden to start + 30 min.
        path = write_volunteer_csv(tmp_path, [
            volunteer_row("Kyiv City", "2024-01-01 10:00+00:00", "2024-01-01 10:00+00:00", naive=True),
        ])
        result = load_volunteer_csv(path)
        assert result.iloc[0]["naive"] == True
        expected_end = ts("2024-01-01 10:30+00:00")
        assert result.iloc[0]["finished_at"] == expected_end

    def test_canonical_columns_present(self, tmp_path):
        path = write_volunteer_csv(tmp_path, [
            volunteer_row("Kyiv City", "2024-01-01 10:00+00:00", "2024-01-01 11:00+00:00"),
        ])
        result = load_volunteer_csv(path)
        expected = {"region", "started_at", "finished_at", "naive", "source", "geo_level"}
        assert expected.issubset(set(result.columns))

    def test_region_column_kept_as_is(self, tmp_path):
        path = write_volunteer_csv(tmp_path, [
            volunteer_row("Lviv", "2024-01-01 10:00+00:00", "2024-01-01 11:00+00:00"),
        ])
        result = load_volunteer_csv(path)
        assert result.iloc[0]["region"] == "Lviv"

    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_volunteer_csv(tmp_path / "nonexistent.csv")


# ---------------------------------------------------------------------------
# Cross-source monthly counts (unit-test the pure helper)
# ---------------------------------------------------------------------------

class TestMonthlyEventCounts:
    def _make_events(self, starts: list[str], region: str) -> pd.DataFrame:
        rows = [
            {
                "region": region,
                "started_at": ts(s),
                "finished_at": ts(s) + pd.Timedelta(hours=1),
                "naive": False,
                "source": "official",
                "geo_level": "oblast",
            }
            for s in starts
        ]
        return pd.DataFrame(rows)

    def test_counts_by_month(self):
        events = self._make_events(
            ["2024-01-05 10:00+00:00",
             "2024-01-15 10:00+00:00",
             "2024-02-03 10:00+00:00"],
            "Kyiv City",
        )
        counts = monthly_event_counts(events, "Kyiv City")
        assert counts[pd.Period("2024-01")] == 2
        assert counts[pd.Period("2024-02")] == 1

    def test_unknown_region_returns_empty(self):
        events = self._make_events(["2024-01-05 10:00+00:00"], "Kyiv City")
        counts = monthly_event_counts(events, "Dnipro")
        assert counts.empty

    def test_single_month_single_event(self):
        events = self._make_events(["2024-03-10 10:00+00:00"], "Lviv")
        counts = monthly_event_counts(events, "Lviv")
        assert len(counts) == 1
        assert counts.iloc[0] == 1