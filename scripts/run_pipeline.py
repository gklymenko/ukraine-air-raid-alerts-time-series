"""End-to-end pipeline runner.

Execution order
---------------
1. Ingest   — load CSV(s) into canonical events DataFrames.
2. Verify   — print data-quality report + cross-source check when volunteer data present.
3. Resample — events -> daily per-oblast series (MultiIndex).
4. Analysis — EDA: decomposition, weekday/hour profiles, duration dist, break view.
5. Forecast — probabilistic baselines for the focal oblast.
6. Evaluate — walk-forward backtest; metrics table + forecast plot.

Data priority
-------------
Official source:
  1. data/raw/official_data_en.csv  (real data)
  2. data/synthetic/synthetic_data.csv  (offline fallback)

Volunteer source (cross-check only, never merged):
  data/raw/volunteer_data_en.csv  (used if present; skipped silently otherwise)
"""

import sys
from pathlib import Path

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
from airraid_tsa.config import (  # noqa: E402
    FOCAL_OBLAST,
    OFFICIAL_CSV,
    SYNTHETIC_CSV,
    VOLUNTEER_CSV,
)
from airraid_tsa.evaluate import evaluate_all  # noqa: E402
from airraid_tsa.ingest import load_official_csv, load_volunteer_csv  # noqa: E402
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
    if OFFICIAL_CSV.exists():
        csv_path = OFFICIAL_CSV
        print(f"[ingest] Using official data: {csv_path}")
    elif SYNTHETIC_CSV.exists():
        csv_path = SYNTHETIC_CSV
        print(f"[ingest] Official data not found. Using synthetic fixture: {csv_path}")
    else:
        print(
            "[ingest] ERROR: No data file found.\n"
            f"  Expected official CSV at  : {OFFICIAL_CSV}\n"
            f"  Expected synthetic at     : {SYNTHETIC_CSV}\n"
            "Run 'python scripts/make_synthetic.py' first."
        )
        sys.exit(1)

    events = load_official_csv(csv_path)
    print(f"[ingest] Loaded {len(events):,} official-oblast events.\n")

    # Volunteer data for cross-check (silently skip if absent).
    volunteer_events = None
    if VOLUNTEER_CSV.exists():
        try:
            volunteer_events = load_volunteer_csv(VOLUNTEER_CSV)
            print(f"[ingest] Loaded {len(volunteer_events):,} volunteer events for cross-check.\n")
        except Exception as exc:
            print(f"[ingest] WARNING: Could not load volunteer CSV: {exc}\n")

    # ------------------------------------------------------------------
    # 2. Verify
    # ------------------------------------------------------------------
    run_quality_report(
        events,
        volunteer_events=volunteer_events,
        ingest_stats=events.attrs,
    )

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

    focal_df = daily_series_filled(events, FOCAL_OBLAST)
    focal_minutes = focal_df["alert_minutes"]

    print(f"\nFocal-oblast daily series (first 10 rows):")
    print(focal_df.head(10).to_string())

    print(f"\n[analysis] Decomposing '{FOCAL_OBLAST}' alert_minutes (period=7)…")
    decomp_result = decompose(focal_minutes, period=7, region=FOCAL_OBLAST)
    if decomp_result is not None:
        plot_decomposition(
            decomp_result,
            title=f"{FOCAL_OBLAST} — alert_minutes  (additive, period=7)",
            filename=f"{_slug(FOCAL_OBLAST)}_decomposition.png",
        )

    wp = weekday_profile(focal_minutes)
    print(f"\nWeekday profile — mean alert_minutes ({FOCAL_OBLAST}):")
    for dow, val in wp.items():
        print(f"  {_DAY_LABELS[dow]}: {val:7.1f} min")
    plot_weekday_profile(wp, region=FOCAL_OBLAST, metric="alert_minutes")

    hp = hour_of_day_profile(events, FOCAL_OBLAST)
    print(f"\nHour-of-day profile — top 6 hours ({FOCAL_OBLAST}):")
    top_hours = hp.sort_values(ascending=False).head(6)
    for hour, count in top_hours.items():
        print(f"  {hour:02d}:00 UTC  →  {count} alert(s)")
    plot_hour_profile(hp, region=FOCAL_OBLAST)

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

    rolling = structural_break_view(focal_minutes)
    plot_structural_break(focal_minutes, rolling, region=FOCAL_OBLAST)

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
    # 5–6. Forecast + Evaluate
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"FORECAST + EVALUATION — focal oblast: {FOCAL_OBLAST}")
    print("=" * 60)
    print(f"\n[forecast] Target: daily alert_minutes for '{FOCAL_OBLAST}'")
    print(f"[forecast] Series length: {len(focal_minutes)} days")

    evaluate_all(focal_minutes, level=0.80)

    # ------------------------------------------------------------------
    # Summary of generated outputs
    # ------------------------------------------------------------------
    from airraid_tsa.config import OUTPUTS_DIR
    output_files = sorted(OUTPUTS_DIR.glob("*"))
    print(f"\n[outputs] {len(output_files)} file(s) in {OUTPUTS_DIR}:")
    for f in output_files:
        print(f"  {f.name}")

    print("\n[pipeline] Done.")


def _slug(name: str) -> str:
    return name.lower().replace(" ", "_").replace("/", "_")


if __name__ == "__main__":
    main()