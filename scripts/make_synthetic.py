"""Generate a tiny synthetic fixture for offline pipeline testing.

Writes to data/synthetic/synthetic_data.csv.

Fixture design
--------------
- Three oblasts: "Kyiv City", "Lviv", "Kharkiv".
- ~60 days of data (2024-01-01 to 2024-02-29).
- Varied alert counts per day (Kyiv City: busier; Lviv: quieter; Kharkiv: always-on).
- One alert that crosses midnight (tests the day-split logic in resample.py).
- One naive alert (finished_at should be ignored; ingest sets it to start + 30 min).
- A few suspiciously long Kharkiv alerts (tests always-on detection in verify.py).
"""

import random
import sys
from pathlib import Path

import pandas as pd

# Allow running as `python scripts/make_synthetic.py` from the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from airraid_tsa.config import DATA_SYNTHETIC_DIR  # noqa: E402


def main() -> None:
    random.seed(42)

    rows: list[dict] = []

    # --- Kyiv City: regular alerts, varied times, one crosses midnight. ---
    base = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
    for day_offset in range(60):
        day = base + pd.Timedelta(days=day_offset)
        n_alerts = random.randint(1, 4)
        for _ in range(n_alerts):
            hour = random.randint(0, 22)
            duration_min = random.randint(20, 90)
            started = day + pd.Timedelta(hours=hour, minutes=random.randint(0, 59))
            finished = started + pd.Timedelta(minutes=duration_min)
            rows.append(
                {
                    "region": "Kyiv City",
                    "started_at": started,
                    "finished_at": finished,
                    "naive": False,
                }
            )

    # --- Midnight-crossing alert (key test case for resample.py) ---
    rows.append(
        {
            "region": "Kyiv City",
            "started_at": pd.Timestamp("2024-01-15 23:30:00", tz="UTC"),
            "finished_at": pd.Timestamp("2024-01-16 00:45:00", tz="UTC"),
            "naive": False,
        }
    )

    # --- Naive alert: finished_at is meaningless; ingest will override it. ---
    rows.append(
        {
            "region": "Kyiv City",
            "started_at": pd.Timestamp("2024-01-20 14:00:00", tz="UTC"),
            # This will be replaced by start + 30 min by ingest.
            "finished_at": pd.Timestamp("2024-01-20 14:00:00", tz="UTC"),
            "naive": True,
        }
    )

    # --- Lviv: quieter, ~1 alert every 2 days. ---
    for day_offset in range(60):
        if random.random() < 0.5:
            day = base + pd.Timedelta(days=day_offset)
            started = day + pd.Timedelta(hours=random.randint(8, 20))
            finished = started + pd.Timedelta(minutes=random.randint(30, 60))
            rows.append(
                {
                    "region": "Lviv",
                    "started_at": started,
                    "finished_at": finished,
                    "naive": False,
                }
            )

    # --- Kharkiv: always-on style — a few very long alerts. ---
    long_alerts = [
        ("2024-01-01 06:00:00", "2024-01-01 22:00:00"),  # 16 h
        ("2024-01-10 00:00:00", "2024-01-10 20:00:00"),  # 20 h
        ("2024-02-01 04:00:00", "2024-02-01 23:59:00"),  # nearly 20 h
    ]
    for start_str, end_str in long_alerts:
        rows.append(
            {
                "region": "Kharkiv",
                "started_at": pd.Timestamp(start_str, tz="UTC"),
                "finished_at": pd.Timestamp(end_str, tz="UTC"),
                "naive": False,
            }
        )
    # Plus some short ones.
    for day_offset in range(0, 60, 3):
        day = base + pd.Timedelta(days=day_offset)
        started = day + pd.Timedelta(hours=random.randint(0, 23))
        finished = started + pd.Timedelta(minutes=45)
        rows.append(
            {
                "region": "Kharkiv",
                "started_at": started,
                "finished_at": finished,
                "naive": False,
            }
        )

    df = pd.DataFrame(rows)
    df = df.sort_values("started_at").reset_index(drop=True)

    DATA_SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_SYNTHETIC_DIR / "synthetic_data.csv"
    df.to_csv(out_path, index=False)

    print(f"Wrote {len(df)} rows to {out_path}")
    print(df.groupby("region").size().rename("events").to_string())


if __name__ == "__main__":
    main()