"""Generate a tiny synthetic fixture for offline pipeline testing.

Writes to data/synthetic/synthetic_data.csv in the OFFICIAL multi-level
format: oblast, raion, hromada, level, started_at, finished_at, source.

Fixture design
--------------
- Three oblasts: "Kyiv City", "Lviv", "Kharkiv".
- ~60 days of data (2024-01-01 to 2024-02-29).
- Mix of levels: most rows are "oblast"; a few are "raion" and "hromada"
  (to confirm the level-filter in OfficialOblastCsvAdapter works).
- One duplicate oblast interval (same region+start+end twice) to exercise
  the drop_duplicates step.
- One oblast alert crossing midnight.
- naive=False throughout (official source carries no naive signal).
"""

import random
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from airraid_tsa.config import DATA_SYNTHETIC_DIR  # noqa: E402


def main() -> None:
    random.seed(42)
    rows: list[dict] = []

    def _row(
        oblast: str,
        started: pd.Timestamp,
        finished: pd.Timestamp,
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

    base = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")

    # --- Kyiv City: regular oblast-level alerts. ---
    for day_offset in range(60):
        day = base + pd.Timedelta(days=day_offset)
        n_alerts = random.randint(1, 4)
        for _ in range(n_alerts):
            hour = random.randint(0, 22)
            duration_min = random.randint(20, 90)
            started = day + pd.Timedelta(hours=hour, minutes=random.randint(0, 59))
            finished = started + pd.Timedelta(minutes=duration_min)
            rows.append(_row("Kyiv City", started, finished))

    # --- Midnight-crossing oblast alert (key test case for resample.py). ---
    rows.append(_row(
        "Kyiv City",
        pd.Timestamp("2024-01-15 23:30:00", tz="UTC"),
        pd.Timestamp("2024-01-16 00:45:00", tz="UTC"),
    ))

    # --- Duplicate: exact copy of one Kyiv City interval (exercises dedup). ---
    dup_start = pd.Timestamp("2024-01-20 14:00:00", tz="UTC")
    dup_end   = pd.Timestamp("2024-01-20 14:45:00", tz="UTC")
    rows.append(_row("Kyiv City", dup_start, dup_end))
    rows.append(_row("Kyiv City", dup_start, dup_end))   # <-- exact duplicate

    # --- Raion and hromada rows (must be filtered out by the adapter). ---
    rows.append(_row(
        "Kyiv City",
        pd.Timestamp("2024-01-10 10:00:00", tz="UTC"),
        pd.Timestamp("2024-01-10 10:30:00", tz="UTC"),
        level="raion",
        raion="Desnyanskyi raion",
    ))
    rows.append(_row(
        "Lviv",
        pd.Timestamp("2024-01-10 10:00:00", tz="UTC"),
        pd.Timestamp("2024-01-10 10:30:00", tz="UTC"),
        level="hromada",
        raion="Lviv raion",
        hromada="Lviv hromada",
    ))

    # --- Lviv: quieter, ~1 oblast alert every 2 days. ---
    for day_offset in range(60):
        if random.random() < 0.5:
            day = base + pd.Timedelta(days=day_offset)
            started = day + pd.Timedelta(hours=random.randint(8, 20))
            finished = started + pd.Timedelta(minutes=random.randint(30, 60))
            rows.append(_row("Lviv", started, finished))

    # --- Kharkiv: always-on style — a few very long oblast alerts. ---
    long_alerts = [
        ("2024-01-01 06:00:00", "2024-01-01 22:00:00"),  # 16 h
        ("2024-01-10 00:00:00", "2024-01-10 20:00:00"),  # 20 h
        ("2024-02-01 04:00:00", "2024-02-01 23:59:00"),  # ~20 h
    ]
    for start_str, end_str in long_alerts:
        rows.append(_row(
            "Kharkiv",
            pd.Timestamp(start_str, tz="UTC"),
            pd.Timestamp(end_str, tz="UTC"),
        ))
    for day_offset in range(0, 60, 3):
        day = base + pd.Timedelta(days=day_offset)
        started = day + pd.Timedelta(hours=random.randint(0, 23))
        finished = started + pd.Timedelta(minutes=45)
        rows.append(_row("Kharkiv", started, finished))

    df = pd.DataFrame(rows).sort_values("started_at").reset_index(drop=True)

    DATA_SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_SYNTHETIC_DIR / "synthetic_data.csv"
    df.to_csv(out_path, index=False)

    level_counts = df["level"].value_counts()
    print(f"Wrote {len(df)} rows to {out_path}")
    print(f"Level breakdown: {level_counts.to_dict()}")
    print(df.groupby(["level", "oblast"]).size().rename("events").to_string())


if __name__ == "__main__":
    main()