"""End-to-end pipeline runner.

Execution order:
  1. Ingest  — load CSV into canonical events DataFrame.
  2. Verify  — print data-quality report.
  3. Resample — events -> daily per-oblast series.
  4. Analysis — TODO (Phase 2).
  5. Forecast — TODO (Phase 3).
  6. Evaluate — TODO (Phase 3).

Picks up the real CSV if it exists at data/raw/official_data_en.csv,
otherwise falls back to the synthetic fixture at data/synthetic/synthetic_data.csv.
"""

import sys
from pathlib import Path

# Make the src package importable when running as a script.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from airraid_tsa.config import FOCAL_OBLAST, REAL_CSV, SYNTHETIC_CSV  # noqa: E402
from airraid_tsa.ingest import load_vadimkin_csv  # noqa: E402
from airraid_tsa.resample import resample_daily  # noqa: E402
from airraid_tsa.verify import run_quality_report  # noqa: E402


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Ingest
    # ------------------------------------------------------------------
    if REAL_CSV.exists():
        csv_path = REAL_CSV
        print(f"[ingest] Using real data: {csv_path}")
    elif SYNTHETIC_CSV.exists():
        csv_path = SYNTHETIC_CSV
        print(f"[ingest] Real data not found. Using synthetic fixture: {csv_path}")
    else:
        print(
            "[ingest] ERROR: No data file found.\n"
            f"  Expected real CSV at  : {REAL_CSV}\n"
            f"  Expected synthetic at : {SYNTHETIC_CSV}\n"
            "Run 'python scripts/make_synthetic.py' first."
        )
        sys.exit(1)

    events = load_vadimkin_csv(csv_path)
    print(f"[ingest] Loaded {len(events):,} events.\n")

    # ------------------------------------------------------------------
    # 2. Verify
    # ------------------------------------------------------------------
    run_quality_report(events)

    # ------------------------------------------------------------------
    # 3. Resample
    # ------------------------------------------------------------------
    print("[resample] Building daily per-oblast series…")
    daily = resample_daily(events)
    print(f"[resample] Done. Shape: {daily.shape}  (region × date rows)\n")

    # Show the daily series for the focal oblast.
    if FOCAL_OBLAST in daily.index.get_level_values("region"):
        focal_daily = daily.xs(FOCAL_OBLAST, level="region")
        print(f"Daily series for '{FOCAL_OBLAST}' (first 10 rows):")
        print(focal_daily.head(10).to_string())
    else:
        print(f"Focal oblast '{FOCAL_OBLAST}' not found in data. Available regions:")
        print(" ", list(daily.index.get_level_values("region").unique()))

    print()
    print("Summary statistics across all regions:")
    print(daily.describe().to_string())

    # ------------------------------------------------------------------
    # 4–6. TODO
    # ------------------------------------------------------------------
    print("\n[analysis]  TODO — Phase 2")
    print("[forecast]  TODO — Phase 3")
    print("[evaluate]  TODO — Phase 3")


if __name__ == "__main__":
    main()