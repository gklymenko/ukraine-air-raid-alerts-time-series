# Air Raid Alerts — Time Series Analysis (Ukraine)

A small, extensible Python pipeline for **time-series analysis of air-raid alerts in
Ukraine**. It turns the public alert log into clean daily per-oblast series, explores their
temporal structure, cross-checks two independent sources, and produces an
honestly-evaluated probabilistic short-horizon forecast.

The focus is **analysis and honest measurement**, not a polished prediction product. This
project models alert *activity* (minutes per day) for a civilian-preparedness framing; it
does **not** predict strikes, targets, or casualties and must not be read that way.

## What it does

- **Ingest** the official alert dataset and normalise it to a clean canonical event log.
- **Verify** data quality — freshness, schema, duplicates, always-on regions, and an
  official-vs-volunteer cross-check.
- **Resample** events into daily per-oblast series (alert count and total alert minutes).
- **Analyse** trend, weekly/daily seasonality, alert-duration distributions, and the
  December-2025 source-aggregation change.
- **Forecast** (beyond the original brief) a short horizon with prediction intervals,
  selected and evaluated by walk-forward backtest.

The code is structured so each piece — data source, model, forecast horizon, region — can
be swapped or extended without rewrites.

## Dataset

Source: **[Vadimkin/ukrainian-air-raid-sirens-dataset](https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset)**,
which compiles Ukrainian air-raid sirens from two streams — an **official** feed and an
**unofficial** one collected by volunteers via the [eTryvoga](https://app.etryvoga.com/)
channel. Both update daily; all timestamps are UTC.

- **Official** (`official_data_en.csv`): from 15 March 2022. Since December 2025 most
  alerts are issued at **raion (district)** level rather than for the whole oblast; before
  that they were oblast-wide. Each row is tagged by a `level` column
  (`oblast` / `raion` / `hromada`).
- **Volunteer** (`volunteer_data_en.csv`): more data, from 25 February 2022, **oblast
  level only**. Alerts with no end signal are marked `naive=True` with
  `finished_at = started_at + 30 min`.

**What this project uses:** the **official** file, filtered to `level == "oblast"` and
deduplicated on `(region, started_at, finished_at)` — the oblast subset has systematic ~2×
exact duplicates. This yields ~65,000 clean oblast-level events across 25 oblasts, from
15 March 2022 to the latest update. The volunteer file is used **only as an independent
cross-check, never merged**. Kyiv City and Kyivska oblast are treated as separate units.

The committed `data/raw/` and `outputs/` are a **snapshot** for inspection without running
the pipeline; both are regenerated on each run and reflect their generation date. Data
credit: Vadym Klymenko / eTryvoga — see the source repository for its license before
redistributing the CSVs.

## How it works

Pipeline stages (`scripts/run_pipeline.py`):

1. **Ingest** → canonical event log
   (`region, started_at, finished_at, naive, source, geo_level`).
2. **Verify** → printed data-quality report + official-vs-volunteer cross-check.
3. **Resample** → daily per-oblast `alert_count` and `alert_minutes`.
4. **Analyse** → decomposition, weekday/hour-of-day profiles, duration distribution,
   December-2025 marker.
5. **Forecast & evaluate** → baselines, walk-forward metrics, and a saved 7-day forward
   forecast.

All figures, the metrics table, and the forecast CSV are written to `outputs/`.

## Forecasting (added beyond the original scope)

Forecasting was **not** part of the original task (time-series *analysis*). It was added to
round the project out and to demonstrate **honest forecast evaluation**. Principles used:

- **Baselines first** — Naive (last value), SeasonalNaive (same weekday last week),
  MovingAverage(7) — so any future model has a meaningful bar to beat.
- **Probabilistic, not point** — every forecast carries an 80% prediction interval from
  empirical residual quantiles.
- **Honest backtesting** — expanding-window walk-forward (never a random split),
  horizon = 1 day.
- **Metrics that fit the problem** — **MAE** / **RMSE** for point error; **MASE** (error
  scaled by the seasonal-naive benchmark; `< 1` beats naive); **interval coverage** (does
  an 80% interval contain ~80% of actuals?); **mean interval width** (sharpness); **pinball
  loss** (a proper score for the interval quantiles).
- **Select → refit → forecast** — the walk-forward winner (by MAE) is refit on the full
  series and emits `outputs/forecast_next_7_days.csv` with intervals, explicitly flagged as
  a pipeline demonstration rather than a reliable prediction (see *Limitations*).

## Results

Focal target: daily total alert minutes for **Kyiv City**. Expanding-window walk-forward,
horizon = 1 day, 80% prediction interval. ~65,000 oblast-level events, 25 oblasts,
2022-03-15 → 2026-06-21 (1,560 days). The oblast series survives the December-2025
transition intact and agrees with the volunteer source across 52 overlapping months.

| Forecaster       | MAE   | RMSE  | MASE  | Coverage (80% PI) | Pinball |
|------------------|-------|-------|-------|-------------------|---------|
| Naive            | 110.2 | 160.5 | 1.179 | 73.1%             | 30.5    |
| SeasonalNaive    | 116.6 | 164.2 | 1.248 | 72.5%             | 31.0    |
| MovingAverage(7) | 96.3  | 129.1 | 1.031 | 71.9%             | 23.6    |

- **The series is hard to predict.** Every baseline has MASE ≈ 1 — none meaningfully beats
  the seasonal-naive benchmark. Daily alert volume is dominated by sporadic, high-intensity
  episodes (mass-attack nights) that memoryless baselines cannot anticipate. This is an
  honest result, not a tuning failure.
- **No strong weekly cycle.** SeasonalNaive (same weekday last week) is the *worst* model.
- **Smoothing helps.** MovingAverage(7) wins on every point metric — consistent with a
  noisy, bursty signal where averaging beats persistence.
- **Intervals are too narrow.** An 80% PI achieves only ~72% empirical coverage; the
  residual distribution has a heavy right tail the empirical quantiles underestimate.

Generated figures (decomposition, weekday/hour profiles, duration distribution,
forecast-vs-actual) are in `outputs/`.

## Limitations

- **Scope.** Models alert *activity* (minutes/day) for civilian preparedness; not a
  strike/target/casualty predictor.
- **Models.** Memoryless baselines only, no covariates. Neighbouring-region activity and
  richer calendar/regime features are the most likely source of improvement.
- **Uncertainty.** Empirical-residual intervals undercover (~72% vs nominal 80%) because of
  heavy tails.
- **Non-stationarity.** Alert patterns drift with the course of the war (escalation
  periods), so any single global model is fragile across regimes.
- **Always-on regions.** Frontline oblasts (e.g. Sumska, Kharkivska, Donetska) are
  near-continuously under alert and degenerate for volume forecasting; they are flagged and
  handled separately.
- **Hour-of-day is in UTC.** The hour-of-day profile is not yet converted to local
  (Europe/Kyiv) time, so peaks read ~2–3 hours earlier than local wall-clock. *Local-time
  conversion is planned for a future version.*
- **Interval lower bound can be negative.** Prediction intervals are symmetric and may dip
  below zero for alert minutes (a non-negative quantity); e.g. the saved 7-day forecast
  shows an 80% lower bound below 0. *Clipping the lower bound at 0 is planned for a future
  version.*
- **Source caveats.** Official data has no end-of-alert `naive` flag (durations treated as
  exact); the volunteer cross-check is monthly-aggregate, not row-level.

## Running it

Requires Python 3.12; a virtual environment (`.venv`) is expected.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python scripts/run_pipeline.py      # ingest → verify → resample → analyse → forecast
pytest                              # 62 tests
```

To try it fully offline first, generate the synthetic fixture and run on that:
`python scripts/make_synthetic.py`.

## Project structure

```
src/airraid_tsa/
  config.py       # paths, focal oblast, constants (regions derived from data)
  ingest.py       # official + volunteer adapters -> canonical event log
  verify.py       # data-quality report + cross-source check
  resample.py     # events -> daily per-oblast series
  analysis.py     # decomposition, profiles, distributions, Dec-2025 marker
  plots.py        # figure helpers (PNG -> outputs/)
  forecast/       # Forecaster interface + baselines
  evaluate.py     # walk-forward, metrics, forward forecast
scripts/          # make_synthetic.py, run_pipeline.py
tests/            # unit tests
data/             # raw (committed snapshot) + synthetic fixture
outputs/          # generated figures, metrics table, forecast CSV
```

See **`CLAUDE.md`** for the full project spec, data contract, and extension roadmap.

## Future work

- **Self-updating data** — fetch the latest CSV directly from the source (raw GitHub URL,
  or the alerts.in.ua / alerts.com.ua APIs) instead of relying on a committed snapshot.
- **Local-time (Europe/Kyiv) conversion** for the hour-of-day profile.
- **Clip the prediction-interval lower bound at 0** (alert minutes are non-negative).
- Richer models (neighbouring-region & calendar features, gradient boosting), heavy-tail-
  aware prediction intervals, and multi-oblast / multi-horizon forecasting — see
  `CLAUDE.md` §10.

## Data handling (summary)

The official source contains alerts at oblast, raion, and hromada levels. This project uses
only records where `level == "oblast"` to preserve a consistent geographic unit. Kyiv City
and Kyivska oblast are treated as separate units. The oblast-level subset contains
systematic exact-duplicate records; before aggregation, events are deduplicated using the
key `(oblast, started_at, finished_at)`. The resulting clean dataset contains ~65,000
oblast-level alert events from 15 March 2022 to the latest update.