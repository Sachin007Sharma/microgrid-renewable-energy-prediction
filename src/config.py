"""
config.py — Global constants, paths, and split boundaries.
All other modules import from here to keep settings in one place.
"""
import os

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH   = os.path.join(BASE_DIR, "daily_data.csv")
FIG_DIR     = os.path.join(BASE_DIR, "outputs", "figures")
RESULTS_DIR = os.path.join(BASE_DIR, "outputs", "results")

os.makedirs(FIG_DIR,     exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Chronological train / validation / test split dates ───────────────────────
TRAIN_START = "2015-01-01"
TRAIN_END   = "2017-06-30"   # ~912 days  (62.5 %)
VAL_START   = "2017-07-01"
VAL_END     = "2017-12-31"   # ~184 days  (12.6 %)
TEST_START  = "2018-01-01"
TEST_END    = "2018-12-30"   # ~364 days  (24.9 %)

# ── Target columns ────────────────────────────────────────────────────────────
TARGETS     = ["solar_MWh", "wind_MWh"]

# ── Feature engineering settings ─────────────────────────────────────────────
LAG_DAYS    = [1, 2, 7, 14]        # Lag features for ML/DL models
ROLLING_WIN = [7, 30]              # Rolling mean/std windows

# ── Deep-learning settings ────────────────────────────────────────────────────
DL_LOOKBACK = 30       # Number of past days used as input sequence
DL_EPOCHS   = 60
DL_BATCH    = 32
DL_UNITS    = 64

# ── Matplotlib style ──────────────────────────────────────────────────────────
PALETTE = {
    "solar":   "#F4A261",
    "wind":    "#2A9D8F",
    "demand":  "#264653",
    "forecast":"#E76F51",
    "pred":    "#8338EC",
    "actual":  "#023E8A",
    "error":   "#E63946",
}

FIGSIZE_WIDE  = (14, 5)
FIGSIZE_TALL  = (14, 8)
FIGSIZE_SQ    = (10, 8)
DPI           = 130

