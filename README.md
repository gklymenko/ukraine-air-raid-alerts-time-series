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

Place the Vadimkin dataset at `data/raw/official_data_en.csv` (git-ignored).
Download from: https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset

Without real data the pipeline runs on a tiny synthetic fixture:

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
  config.py        — paths, oblast list, constants
  ingest.py        — CSV -> canonical events DataFrame
  verify.py        — data-quality + freshness report
  resample.py      — events -> daily per-oblast time series
  analysis.py      — EDA: decomposition, patterns (TODO)
  plots.py         — plotting helpers (TODO)
  forecast/        — Forecaster ABC + baseline models (TODO)
  evaluate.py      — metrics and walk-forward evaluation (TODO)
scripts/
  make_synthetic.py  — generate synthetic fixture
  run_pipeline.py    — end-to-end pipeline runner
tests/
  test_resample.py
  test_evaluate.py
data/
  raw/             — real CSV (git-ignored)
  synthetic/       — committed fixture
outputs/           — generated plots and reports (git-ignored)
```

## Honesty note

This project analyses alert *activity* (counts / total minutes) for civilian-preparedness
framing. It does **not** predict strikes, targets, or casualties, and never claims to.
Forecasts are always reported against a naive baseline and with explicit uncertainty.