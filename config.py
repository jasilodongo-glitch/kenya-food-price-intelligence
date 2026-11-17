"""Central configuration for Kenya Food Price Intelligence."""
from pathlib import Path

APP_NAME = "Kenya Food Price Intelligence"
TAGLINE = "Transforming food price data into actionable intelligence."

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "outputs"
DB_PATH = BASE_DIR / "data" / "metadata.sqlite3"

ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls"}
MAX_UPLOAD_MB = 100
MAX_ROWS = 2_000_000
PREVIEW_ROWS = 200
PAGE_SIZE_OPTIONS = [25, 50, 100, 250]

# Minimum observations required before forecasting / decomposition is attempted.
MIN_FORECAST_POINTS = 24
MIN_DECOMPOSE_PERIODS = 2  # full seasonal cycles

# Colour system (used by both CSS and Plotly).
COLORS = {
    "canvas": "#F6F8F5",
    "ink": "#17211C",
    "muted": "#66736B",
    "line": "#DCE3DB",
    "green": "#1F5F46",
    "green_soft": "#E3EEE7",
    "maize": "#D9A21B",
    "clay": "#B5533C",
    "slate": "#3E5C76",
}
SERIES_COLORS = ["#1F5F46", "#D9A21B", "#3E5C76", "#B5533C", "#7A9E7E", "#8C6A9F", "#5C6B73", "#C98B4B"]
