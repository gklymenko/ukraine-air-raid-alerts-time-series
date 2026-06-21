"""Project-wide paths, constants, and reference data."""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_SYNTHETIC_DIR = PROJECT_ROOT / "data" / "synthetic"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

OFFICIAL_CSV = DATA_RAW_DIR / "official_data_en.csv"
VOLUNTEER_CSV = DATA_RAW_DIR / "volunteer_data_en.csv"
SYNTHETIC_CSV = DATA_SYNTHETIC_DIR / "synthetic_data.csv"

# Back-compat alias used in verify/run_pipeline
REAL_CSV = OFFICIAL_CSV

# ---------------------------------------------------------------------------
# Source identifiers
# ---------------------------------------------------------------------------
PRIMARY_SOURCE: str = "official"

# ---------------------------------------------------------------------------
# Dataset constants
# ---------------------------------------------------------------------------

# Duration assigned to naive alerts (volunteer source only).
NAIVE_DURATION_MINUTES: int = 30

# Freshness threshold — warn if newest event is older than this many days.
FRESHNESS_WARN_DAYS: int = 2

# Threshold: an alert spanning more than this many hours is "always-on" suspect.
ALWAYS_ON_HOURS_THRESHOLD: float = 12.0

# ---------------------------------------------------------------------------
# Analysis defaults
# ---------------------------------------------------------------------------

# Default region for focused single-region analysis.
FOCAL_OBLAST: str = "Kyiv City"

# ---------------------------------------------------------------------------
# Source-aggregation change marker (December 2025)
# Single source of truth — import this everywhere a date is needed for the marker.
# ---------------------------------------------------------------------------
import pandas as pd  # noqa: E402 (imported here so config stays self-contained)

SOURCE_AGGREGATION_CHANGE: pd.Timestamp = pd.Timestamp("2025-12-01", tz="UTC")

BREAK_NOTE_2025: str = (
    "Around December 2025 the official source's aggregation changed: district "
    "(raion/hromada) alerts became the dominant recording unit nationwide (phased in "
    "from spring 2025). This project analyses the level=='oblast' subset, which stays "
    "continuous across the change — the marker is shown for context, not because a "
    "discontinuity was observed in the oblast series."
)