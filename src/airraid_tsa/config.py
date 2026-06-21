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
# 2025 structural-break note
# ---------------------------------------------------------------------------
BREAK_NOTE_2025: str = (
    "NOTE: Around early 2025 some regions transitioned to district-level alert "
    "aggregation and aggregators changed their methodology. This creates an apparent "
    "structural break in the oblast-level series that does NOT reflect a real change "
    "in underlying threat activity. Treat any sharp 2025 level-shift with caution."
)