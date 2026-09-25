"""Central configuration. Every threshold used by the analysis lives here."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.getenv("DYNAMICDS_DATA_DIR", os.getenv("AUTODS_DATA_DIR", ROOT / "data")))

# ---- upload limits (configurable via env) --------------------------------
MAX_UPLOAD_MB = int(os.getenv("DYNAMICDS_MAX_UPLOAD_MB", os.getenv("AUTODS_MAX_UPLOAD_MB", "50")))
MAX_ROWS = int(os.getenv("DYNAMICDS_MAX_ROWS", os.getenv("AUTODS_MAX_ROWS", "500000")))
MAX_COLS = int(os.getenv("DYNAMICDS_MAX_COLS", os.getenv("AUTODS_MAX_COLS", "300")))
ALLOWED_EXTENSIONS = {".csv", ".xlsx"}

# ---- placeholder tokens (compared after strip + lowercase) ------------------
PLACEHOLDER_TOKENS = {
    "nan", "null", "none", "na", "n/a", "n.a.", "unknown", "undefined", "missing",
    "not available", "not applicable", "?", "??", "-", "--", "#n/a", "nil",
}

# ---- missingness bands (percent) ---------------------------------------------
MISSING_LOW = 5.0        # (0, 5)   -> low
MISSING_MODERATE = 20.0  # [5, 20)  -> moderate
MISSING_HIGH = 50.0      # [20, 50) -> high ; >= 50 -> very high
ROW_MISSING_FLAG = 0.5   # fraction of a row's cells missing to flag the row

# ---- skewness ------------------------------------------------------------------
SKEW_SYMMETRIC = 0.5     # |skew| < 0.5  -> approximately symmetric
SKEW_STRONG = 1.0        # 0.5-1 moderate ; >= 1 strong

# ---- correlation ------------------------------------------------------------------
CORR_LOW = 0.3           # |r| < 0.3 low ; 0.3-0.7 moderate ; >= 0.7 high
CORR_HIGH = 0.7
CORR_WARN = 0.9          # pairs at/above this raise a WARNING
CORR_MAX_COLS = 40
LEAKAGE_ASSOC = 0.95     # feature-target association at/above this is suspicious

# ---- identifiers / cardinality / constants -----------------------------------------
ID_STRONG_UNIQUE = 0.99          # id-like name and >= 99% unique
ID_NAME_POSSIBLE_UNIQUE = 0.5    # id-like name and >= 50% unique
ID_POSSIBLE_UNIQUE = 0.90        # no id-like name, string/integer, >= 90% unique
ID_MIN_ROWS = 30                 # uniqueness-only rules need at least this many rows
CARD_LOW = 10                    # unique count <= 10 -> low ; <= 50 medium ; > 50 high
CARD_HIGH = 50
RARE_CATEGORY_SHARE = 0.01
NEAR_CONSTANT_SHARE = 0.95

# ---- outliers ----------------------------------------------------------------------------
OUTLIER_IQR_K = 1.5
OUTLIER_WARN_PCT = 5.0

# ---- text -------------------------------------------------------------------------------------
TEXT_MIN_AVG_LEN = 40
TEXT_MIN_MEDIAN_TOKENS = 5
TEXT_MIN_UNIQUE_RATIO = 0.3

# ---- visualisation sampling --------------------------------------------------------------
VIS_SAMPLE_ROWS = 5000
ASSOC_SAMPLE_ROWS = 100_000
SHAPIRO_MAX = 5000

# ---- modelling ----------------------------------------------------------------------------------
OHE_MAX_CATEGORIES = 30
IMBALANCE_MINORITY_SHARE = 0.2
MIN_ROWS_MODEL = 30
MIN_ROWS_FORECAST = 30
RANDOM_STATE = 42
FORECAST_TRAIN_FRAC = 0.70
FORECAST_VAL_FRAC = 0.15
MODEL_STATUSES = ["Development", "Candidate", "Production", "Archived"]
