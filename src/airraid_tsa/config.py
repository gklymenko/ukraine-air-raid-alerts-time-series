"""Project-wide paths, constants, and reference data."""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # …/ukrainian-air-raid-sirens-dataset

DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_SYNTHETIC_DIR = PROJECT_ROOT / "data" / "synthetic"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

REAL_CSV = DATA_RAW_DIR / "official_data_en.csv"
SYNTHETIC_CSV = DATA_SYNTHETIC_DIR / "synthetic_data.csv"

# ---------------------------------------------------------------------------
# Dataset schema
# ---------------------------------------------------------------------------
REQUIRED_COLUMNS: list[str] = ["region", "started_at", "finished_at", "naive"]

# Duration assigned to naive alerts (no end signal recorded).
NAIVE_DURATION_MINUTES: int = 30

# Freshness threshold — warn if newest event is older than this many days.
FRESHNESS_WARN_DAYS: int = 2

# ---------------------------------------------------------------------------
# Oblast reference list (25 oblasts + Kyiv City + AR Crimea)
# ---------------------------------------------------------------------------
OBLAST_LIST: list[str] = [
    "Cherkasy",
    "Chernihiv",
    "Chernivtsi",
    "Crimea",          # AR Crimea / Autonomous Republic of Crimea
    "Dnipropetrovsk",
    "Donetsk",
    "Ivano-Frankivsk",
    "Kharkiv",
    "Kherson",
    "Khmelnytskyi",
    "Kirovohrad",
    "Kyiv",            # Kyiv Oblast (region surrounding the city)
    "Kyiv City",       # The city itself — default focal oblast
    "Luhansk",
    "Lviv",
    "Mykolaiv",
    "Odesa",
    "Poltava",
    "Rivne",
    "Sumy",
    "Ternopil",
    "Vinnytsia",
    "Volyn",
    "Zakarpattia",
    "Zaporizhzhia",
    "Zhytomyr",
]

OBLAST_SET: set[str] = set(OBLAST_LIST)

# Default region for focused single-region analysis.
FOCAL_OBLAST: str = "Kyiv City"

# ---------------------------------------------------------------------------
# "Always-on" regions
# Regions known to have had very long continuous alert spans — these are
# degenerate for volume forecasting and should be flagged.
# ---------------------------------------------------------------------------
ALWAYS_ON_REGIONS: set[str] = {"Donetsk", "Luhansk", "Zaporizhzhia", "Kherson"}

# Threshold: an alert spanning more than this many hours is "always-on" suspect.
ALWAYS_ON_HOURS_THRESHOLD: float = 12.0

# ---------------------------------------------------------------------------
# 2025 structural-break note
# ---------------------------------------------------------------------------
BREAK_NOTE_2025: str = (
    "NOTE: Around early 2025 some regions transitioned to district-level alert "
    "aggregation and aggregators changed their methodology. This creates an apparent "
    "structural break in the oblast-level series that does NOT reflect a real change "
    "in underlying threat activity. Treat any sharp 2025 level-shift with caution."
)