"""End-to-end pipeline runner.

Execution order
---------------
1. Ingest   — load CSV into canonical events DataFrame.
2. Verify   — print data-quality report.
3. Resample — events -> daily per-oblast series (MultiIndex).
4. Analysis — EDA: decomposition, weekday/hour profiles, duration dist, break view.
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

from airraid_tsa.analysis import (  # noqa: E402
    daily_series_filled,
    decompose,
    duration_distribution,
    hour_of_day_profile,
    structural_break_view,
    weekday_profile,
)
from airraid_tsa.config import FOCAL_OBLAST, REAL_CSV, SYNTHETIC_CSV  # noqa: E402
from airraid_tsa.ingest import load_vadimkin_csv  # noqa: E402
from airraid_tsa.plots import (  # noqa: E402
    plot_decomposition,
    plot_duration_distribution,
    plot_hour_profile,
    plot_structural_break,
    plot_weekday_profile,
)
from airraid_tsa.resample import resample_daily  # noqa: E402
from airraid_tsa.verify import run_quality_report  # noqa: E402

_DAY_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


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

    # ------------------------------------------------------------------
    # 4. Analysis
    # ------------------------------------------------------------------
    print("=" * 60)
    print(f"ANALYSIS — focal oblast: {FOCAL_OBLAST}")
    print("=" * 60)

    # 4a. Zero-filled daily series for the focal oblast.
    focal_df = daily_series_filled(events, FOCAL_OBLAST)
    focal_minutes = focal_df["alert_minutes"]

    print(f"\nFocal-oblast daily series (first 10 rows):")
    print(focal_df.head(10).to_string())

    # 4b. Seasonal decomposition — focal oblast.
    print(f"\n[analysis] Decomposing '{FOCAL_OBLAST}' alert_minutes (period=7)…")
    decomp_result = decompose(focal_minutes, period=7, region=FOCAL_OBLAST)
    if decomp_result is not None:
        plot_decomposition(
            decomp_result,
            title=f"{FOCAL_OBLAST} — alert_minutes  (additive, period=7)",
            filename=f"{_slug(FOCAL_OBLAST)}_decomposition.png",
        )

    # 4c. Weekday profile — focal oblast.
    wp = weekday_profile(focal_minutes)
    print(f"\nWeekday profile — mean alert_minutes ({FOCAL_OBLAST}):")
    for dow, val in wp.items():
        print(f"  {_DAY_LABELS[dow]}: {val:7.1f} min")
    plot_weekday_profile(wp, region=FOCAL_OBLAST, metric="alert_minutes")

    # 4d. Hour-of-day profile — focal oblast.
    hp = hour_of_day_profile(events, FOCAL_OBLAST)
    print(f"\nHour-of-day profile — top 6 hours ({FOCAL_OBLAST}):")
    top_hours = hp.sort_values(ascending=False).head(6)
    for hour, count in top_hours.items():
        print(f"  {hour:02d}:00 UTC  →  {count} alert(s)")
    plot_hour_profile(hp, region=FOCAL_OBLAST)

    # 4e. Duration distribution — focal oblast.
    all_dur, non_naive_dur = duration_distribution(events, FOCAL_OBLAST)
    print(f"\nAlert duration distribution ({FOCAL_OBLAST}):")
    print(f"  All alerts       :  n={len(all_dur):4d}  "
          f"mean={all_dur.mean():.1f} min  "
          f"median={all_dur.median():.1f} min  "
          f"p90={all_dur.quantile(0.9):.1f} min")
    print(f"  Non-naive alerts :  n={len(non_naive_dur):4d}  "
          f"mean={non_naive_dur.mean():.1f} min  "
          f"median={non_naive_dur.median():.1f} min  "
          f"p90={non_naive_dur.quantile(0.9):.1f} min")
    plot_duration_distribution(all_dur, non_naive_dur, region=FOCAL_OBLAST)

    # 4f. Structural break view — focal oblast.
    rolling = structural_break_view(focal_minutes)
    plot_structural_break(focal_minutes, rolling, region=FOCAL_OBLAST)

    # 4g. Cross-oblast decompositions — all non-always-on regions.
    available_regions = daily.index.get_level_values("region").unique().tolist()
    other_regions = sorted(r for r in available_regions if r != FOCAL_OBLAST)

    if other_regions:
        print(f"\n[analysis] Cross-oblast decompositions ({len(other_regions)} region(s))…")
        for region in other_regions:
            region_minutes = daily.xs(region, level="region")["alert_minutes"]
            result = decompose(region_minutes, period=7, region=region)
            if result is not None:
                plot_decomposition(
                    result,
                    title=f"{region} — alert_minutes  (additive, period=7)",
                    filename=f"{_slug(region)}_decomposition.png",
                )

    # ------------------------------------------------------------------
    # Summary of generated outputs
    # ------------------------------------------------------------------
    from airraid_tsa.config import OUTPUTS_DIR
    output_files = sorted(OUTPUTS_DIR.glob("*.png"))
    print(f"\n[outputs] {len(output_files)} file(s) in {OUTPUTS_DIR}:")
    for f in output_files:
        print(f"  {f.name}")

    # ------------------------------------------------------------------
    # 5–6. TODO
    # ------------------------------------------------------------------
    print("\n[forecast]  TODO — Phase 3")
    print("[evaluate]  TODO — Phase 3")


def _slug(name: str) -> str:
    return name.lower().replace(" ", "_").replace("/", "_")


if __name__ == "__main__":
    main()