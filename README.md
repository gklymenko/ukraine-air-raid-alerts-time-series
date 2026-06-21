# Ukrainian Air Raid Sirens — Time Series Analysis

A self-contained Python analysis pipeline for Ukrainian air raid alert data.
The primary goal is understanding the temporal structure of alerts (trend, seasonality,
patterns, structural breaks) with a small probabilistic forecast on top.

## Setup

The `.venv` (Python 3.12) already exists. Activate it and install dependencies:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## Data

Download both files from
[Vadimkin / ukrainian-air-raid-sirens-dataset](https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset)
and place them at:

```
data/raw/official_data_en.csv    ← primary source (official alerts)
data/raw/volunteer_data_en.csv   ← secondary source (cross-check only)
```

**Data handling.** The official source contains alerts at oblast, raion, and hromada
levels. This project uses only records where `level == "oblast"` to preserve a
consistent geographic unit. Kyiv City and Kyivska oblast are treated as separate units.

The oblast-level subset contains systematic exact-duplicate records. Before aggregation,
events are deduplicated using the key `(oblast, started_at, finished_at)`.

The resulting clean dataset contains ~65,000 oblast-level alert events from 15 March 2022
to the latest update (the pipeline prints exact, current counts on each run).

Without real data the pipeline runs on a tiny synthetic fixture in the same format:

```bash
python scripts/make_synthetic.py   # generates data/synthetic/synthetic_data.csv
```

## Running

```bash
# Full pipeline (uses synthetic data by default when real data is absent)
python scripts/run_pipeline.py

# Run tests
pytest tests/
```

## Project structure

```
src/airraid_tsa/   — core package
  config.py        — paths, focal oblast, constants (no hardcoded region lists)
  ingest.py        — OfficialOblastCsvAdapter + VolunteerCsvAdapter -> canonical events
  verify.py        — data-quality + freshness + cross-source report
  resample.py      — events -> daily per-oblast time series
  analysis.py      — EDA: decomposition, patterns, structural breaks
  plots.py         — plotting helpers (save PNGs to outputs/)
  forecast/        — Forecaster ABC + NaiveForecaster, SeasonalNaive, MovingAverage
  evaluate.py      — walk-forward backtest, point + probabilistic metrics
scripts/
  make_synthetic.py  — generate synthetic fixture (official multi-level format)
  run_pipeline.py    — end-to-end pipeline runner
tests/
  test_resample.py
  test_evaluate.py
  test_ingest.py
data/
  raw/             — real CSVs (git-ignored)
  synthetic/       — committed fixture
outputs/           — generated plots and reports (git-ignored)
```

## Canonical events model

Every adapter produces a DataFrame with these columns:

| Column | Type | Description |
|---|---|---|
| `region` | str | Oblast name |
| `started_at` | datetime[UTC] | Alert start |
| `finished_at` | datetime[UTC] | Alert end |
| `naive` | bool | `True` when end was estimated (volunteer source only) |
| `source` | str | `"official"` or `"volunteer"` |
| `geo_level` | str | Always `"oblast"` in the MVP |

## Honesty note

This project analyses alert *activity* (counts / total minutes) for civilian-preparedness
framing. It does **not** predict strikes, targets, or casualties, and never claims to.
Forecasts are always reported against a naive baseline and with explicit uncertainty.