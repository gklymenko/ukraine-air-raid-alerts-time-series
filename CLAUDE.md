# CLAUDE.md — Air Raid Alerts: Time Series Analysis (Ukraine)

> Context file for Claude Code. Read this fully before generating or editing any code.
> This is the single source of truth for scope, stack, and conventions.

## 1. What this project is

A **mini, self-contained Python project** that performs **Time Series Analysis of air
raid alerts in Ukraine**. The deliverable is an analysis pipeline, not a product.

The name is *Analysis* first: the heart of the project is understanding the temporal
structure of alerts (trend, seasonality, patterns, structural breaks). A small,
**probabilistic** baseline forecast sits on top of that analysis and is honestly
evaluated. Anything heavier (ML models, multi-region, live data) is an explicitly
deferred extension — see §9.

### Honesty constraints (do not violate)
- This project analyzes and forecasts **alert activity** (counts / total minutes) for
  civilian-preparedness framing. It does **not** predict strikes, targets, or casualties,
  and must never claim to.
- No overclaiming of predictive power. Always report a forecast against a naive baseline
  and against its own uncertainty.

## 2. Hard constraints

- **Time budget: ~5 hours total.** Favor a working end-to-end pipeline over completeness.
- **Mini but extensible.** Clean module boundaries and a `Forecaster` interface so
  extensions slot in without rewrites. Do not pre-build the extensions.
- **Author is new to Python, strong in Java.** Prefer readable, explicit code over clever
  one-liners. Use type hints and short docstrings. Structure the project like a small,
  well-organized package (packages = modules, interfaces = ABCs).
- **Runnable before real data exists.** Ship a tiny synthetic data fixture so the whole
  pipeline runs offline; the real CSV is dropped in later.

## 3. Tech stack

- Python 3.11+
- `pandas`, `numpy` — data
- `matplotlib` — plots
- `statsmodels` — `seasonal_decompose` (and SARIMA later, extension only)
- `pytest` — a few unit tests on the tricky logic (resampling, metrics)
- Standard `venv`. All deps pinned in `requirements.txt`.

Do **not** add: web frameworks, CLIs frameworks, databases, Prophet, LightGBM, or any
API client in the MVP. Those belong to extensions.

## 4. Data source

Primary: **Vadimkin / `ukrainian-air-raid-sirens-dataset`** (GitHub), file
`datasets/official_data_en.csv`.
- Event-log format: one row per alert with `region`, `started_at`, `finished_at` (UTC).
- `naive=True` rows have no end signal; `finished_at = started_at + 30 min` (estimate).
- Long continuous history from 2022, oblast level, no auth needed.

The repo claims daily auto-updates but freshness must be **verified at runtime**, not
assumed (see §6). Fallback sources (extension): `alerts.com.ua` `/api/history`,
`alerts.in.ua` API.

Place the downloaded CSV at `data/raw/official_data_en.csv` (git-ignored).

## 5. Architecture & module responsibilities

> Create this structure **at the existing project root** (the current repo, e.g.
> `ukrainian-air-raid-sirens-dataset`). The top-level name below is illustrative — do
> NOT create a new nested folder. The venv already exists as `.venv` (Python 3.12).

```
<project root>/
  README.md
  CLAUDE.md
  requirements.txt
  .gitignore
  data/
    raw/                      # real CSV here (git-ignored)
    synthetic/                # tiny generated fixture, committed
  src/airraid_tsa/
    __init__.py
    config.py                 # paths, oblast list, "always-on" regions, constants
    ingest.py                 # source adapters -> canonical events DataFrame
    verify.py                 # data-quality + freshness report
    resample.py               # events -> daily per-oblast series
    analysis.py               # EDA: decomposition, dow/hour patterns, distributions, breaks
    plots.py                  # plotting helpers (save PNGs to outputs/)
    forecast/
      __init__.py
      base.py                 # Forecaster ABC: fit() / predict() -> Forecast(point, lower, upper)
      baselines.py            # NaiveForecaster, SeasonalNaiveForecaster, MovingAverageForecaster
    evaluate.py               # time split, walk-forward, point + probabilistic metrics
  scripts/
    make_synthetic.py         # writes data/synthetic fixture
    run_pipeline.py           # ingest -> verify -> resample -> analysis -> forecast -> evaluate
  tests/
    test_resample.py
    test_evaluate.py
  outputs/                    # generated plots + reports (git-ignored)
```

### Canonical events model (the contract every source must produce)
A `pandas.DataFrame` with columns:
`region: str`, `started_at: datetime[UTC]`, `finished_at: datetime[UTC]`,
`naive: bool`, `source: str`.

### Forecaster interface (extension seam — keep stable)
```python
@dataclass
class Forecast:
    point: pd.Series      # indexed by date
    lower: pd.Series      # lower bound of prediction interval
    upper: pd.Series      # upper bound of prediction interval
    level: float          # nominal coverage, e.g. 0.80

class Forecaster(ABC):
    @abstractmethod
    def fit(self, y: pd.Series) -> "Forecaster": ...
    @abstractmethod
    def predict(self, horizon: int, level: float = 0.80) -> Forecast: ...
```
A new model later (Prophet, SARIMA, LightGBM) is just a new `Forecaster` subclass.

## 6. Data verification (required, not optional)

`verify.py` produces a printed/markdown **data-quality report** covering:
- **Freshness**: `max(started_at)` vs today; warn if older than 2 days.
- **Schema**: required columns present and parseable; `finished_at >= started_at` or `naive`.
- **Coverage**: date range, row count, and `% naive` per region.
- **Sanity**: region names within the known oblast set; flag "always-on" regions
  (continuous alert for very long spans — degenerate for volume forecasting).
- **Structural-break note**: explicitly check the series around 2025, when some regions
  moved to district-level alerts and aggregators changed methodology — this can create a
  break that is *not* a real change in activity.

A quick manual check the author can also run:
```python
import pandas as pd
df = pd.read_csv("data/raw/official_data_en.csv", parse_dates=["started_at"])
print(df["started_at"].max())   # newest event in the data
```

## 7. Analysis (the core deliverable)

In `analysis.py`, for a focal oblast (default **Kyiv City**) and over all oblasts:
- daily series of `alerts_count` and `alert_minutes`;
- `seasonal_decompose` (trend / weekly seasonality / residual);
- day-of-week and hour-of-day patterns;
- alert-duration distribution;
- highlight the 2025 structural-break region.
  All figures saved to `outputs/`.

## 8. Forecast + evaluation (probabilistic, honest)

MVP forecast target: **daily `alert_minutes` for the focal oblast.**

Forecast must be **probabilistic**, not just a point:
- Point: `SeasonalNaiveForecaster` (same weekday last week) as the bar to beat;
  `MovingAverageForecaster` as a second baseline.
- Interval: empirical prediction interval from in-sample residual quantiles
  (e.g. 10th/90th percentile → 80% PI).

Evaluation in `evaluate.py`, using a **time-based split** (train before cutoff date,
test after) plus a simple **expanding-window walk-forward**:
- Point metrics: **MAE**, **RMSE**, **MASE** (scaled vs seasonal-naive; `<1` = useful).
- Probabilistic metrics:
    - **Interval coverage**: share of test points inside [lower, upper] vs nominal level
      (well-calibrated ≈ 0.80 for an 80% PI).
    - **Mean interval width** (sharpness).
    - **Pinball (quantile) loss** at the interval quantiles — proper score, lower is better.
- Output: a small metrics table (printed + saved) and a forecast-vs-actual plot with the
  prediction interval shaded.

### ⚠️ Known pitfall — label leakage (applies ONLY to the classification extension)
The MVP forecasts daily volume from an event log and is NOT affected by this. But when
the classification framing (§10, "alert in next N minutes") is built later, beware:
- If the target is "an alert occurs within the next N minutes", then every minute of an
  **already-ongoing** alert is trivially labelled 1. Lumping "alert about to start"
  together with "alert already running" makes the positive class easy and inflates
  metrics (high AUC/accuracy that does NOT reflect real onset-prediction skill).
- Mitigation when we get there: evaluate specifically on **onset transitions (0→1)**, or
  exclude minutes where an alert is already active from the positive class, and never
  feed an "alert currently active" feature when predicting onset.
- This is a target-definition / evaluation problem, not a data-quality problem — the raw
  event log itself is fine.

## 9. Definition of done (MVP)

`python scripts/run_pipeline.py` runs end-to-end on the synthetic fixture **and** on the
real CSV, and produces:
1. a printed data-quality report,
2. the analysis figures in `outputs/`,
3. a forecast-vs-actual plot with intervals,
4. a metrics table (MAE, RMSE, MASE, coverage, interval width, pinball loss).

Plus passing `pytest` on `test_resample.py` and `test_evaluate.py`.

## 10. Extensions (designed-for, NOT built in MVP)

Keep interfaces clean so these drop in later:
- Real models as `Forecaster` subclasses: SARIMA / Prophet (native intervals), LightGBM.
- Multi-region & geospatial neighbor features; lead-lag ("early warning") analysis.
- Live data adapters (`alerts.com.ua`, `alerts.in.ua`) behind the same canonical model.
- Classification framing (P(alert in next N minutes)) as a separate target —
  **mind the label-leakage pitfall in §8 before trusting any metric here.**
- District/hromada-level granularity.

## 11. Coding conventions

- Type hints on public functions; short docstrings stating purpose, inputs, outputs.
- Pure functions where possible; no global state beyond `config.py`.
- No silent failures — `verify.py` raises or clearly warns on bad data.
- Small, named helper functions over long scripts; the author reviews this code as a
  Java dev learning Python, so optimize for readability.
- Comments explain *why*, not *what*.