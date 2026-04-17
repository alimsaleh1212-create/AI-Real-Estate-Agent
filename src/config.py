"""Central configuration for the AI Real Estate Agent.

Loads environment variables, defines path constants, and declares
feature metadata. All other modules import from here — no scattered
os.getenv() calls elsewhere.

GOOGLE_API_KEY is validated lazily via get_google_api_key() so that
modules which do not use the Gemini API (e.g. ml_pipeline, predictor)
can import config without requiring the key to be set.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# LLM — model name and runtime tunables
# ---------------------------------------------------------------------------

GEMINI_MODEL: str = "gemini-2.5-flash"

# Generation temperatures: 0.0 = deterministic (extraction), 0.7 = creative (text)
LLM_EXTRACTION_TEMPERATURE: float = 0.0
LLM_TEXT_TEMPERATURE: float = 0.7

# Token budget per call (10-field JSON ≈ 60 tokens; narrative ≈ 150 tokens)
LLM_MAX_OUTPUT_TOKENS: int = 1024

# LRU cache capacities for repeated identical prompts
LLM_EXTRACTION_CACHE_SIZE: int = 256
LLM_INTENT_CACHE_SIZE: int = 512
LLM_INSIGHTS_CACHE_SIZE: int = 256

# Maximum characters accepted from user input before truncation
MAX_QUERY_LENGTH: int = 500


def get_google_api_key() -> str:
    """Return the Gemini API key, raising if it is not configured.

    Call this only from modules that actually use the Gemini API
    (llm_chain.py). ml_pipeline.py and predictor.py must not call it.

    Returns:
        The API key string.

    Raises:
        RuntimeError: If GOOGLE_API_KEY is not set in the environment.
    """
    key: str = os.getenv("GOOGLE_API_KEY", "")
    if not key:
        raise RuntimeError(
            "GOOGLE_API_KEY environment variable is not set. "
            "Copy .env.example to .env and add your Gemini API key."
        )
    return key


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).parent.parent
DATA_RAW_PATH: Path = PROJECT_ROOT / "data" / "raw" / "AmesHousing.csv"
DATA_PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed"
MODEL_PATH: Path = PROJECT_ROOT / "models" / "best_model_v1.joblib"
STATS_PATH: Path = PROJECT_ROOT / "models" / "training_stats.json"

# ---------------------------------------------------------------------------
# Feature metadata — confirmed by Notebook 03 statistical tests.
#
# FEATURE_TYPES maps each of the 10 selected feature names to its encoding
# strategy, which drives the ColumnTransformer in ml_pipeline.py:
#   "numeric"  → SimpleImputer(median)      + StandardScaler
#   "ordinal"  → SimpleImputer(most_frequent) + OrdinalEncoder(explicit_order)
#   "nominal"  → SimpleImputer(most_frequent) + OneHotEncoder
# ---------------------------------------------------------------------------

FEATURE_TYPES: dict[str, str] = {
    "OverallQual": "numeric",
    "TotalSF": "numeric",
    "GarageCars": "numeric",
    "TotalBath": "numeric",
    "YearBuilt": "numeric",
    "TotalBsmtSF": "numeric",
    "KitchenQual": "ordinal",
    "BsmtQual": "ordinal",
    "ExterQual": "ordinal",
    "Neighborhood": "nominal",
}

# Ordered list preserving the column order used during training.
SELECTED_FEATURES: list[str] = list(FEATURE_TYPES.keys())

# Explicit ordinal orderings (low → high) for OrdinalEncoder.
# Every ordinal feature in FEATURE_TYPES must have an entry here.
_QUALITY_ORDER: list[str] = ["None", "Po", "Fa", "TA", "Gd", "Ex"]

ORDINAL_ORDERS: dict[str, list[str]] = {
    "KitchenQual": _QUALITY_ORDER,
    "BsmtQual": _QUALITY_ORDER,
    "ExterQual": _QUALITY_ORDER,
}

# ---------------------------------------------------------------------------
# Quality codes — single source for ordinal validation, UI widgets, and charts
# ---------------------------------------------------------------------------

# Standard quality codes (no-basement excluded); used by KitchenQual & ExterQual.
QUALITY_CODES: list[str] = ["Po", "Fa", "TA", "Gd", "Ex"]

# Basement quality also accepts "None" (house has no basement).
BSMT_QUALITY_CODES: list[str] = ["None"] + QUALITY_CODES

# Short labels for charts and tables.
QUALITY_LABELS: dict[str, str] = {
    "None": "No Basement",
    "Po": "Poor",
    "Fa": "Fair",
    "TA": "Typical",
    "Gd": "Good",
    "Ex": "Excellent",
}

# Long labels for UI dropdowns (code prefix kept for quick recognition).
QUALITY_DISPLAY: dict[str, str] = {
    "None": "None (no basement)",
    "Po": "Po — Poor",
    "Fa": "Fa — Fair",
    "TA": "TA — Typical/Average",
    "Gd": "Gd — Good",
    "Ex": "Ex — Excellent",
}

# ---------------------------------------------------------------------------
# Neighborhoods — sorted tuple for UI selectbox; frozenset for O(1) validation
# ---------------------------------------------------------------------------

NEIGHBORHOODS: tuple[str, ...] = (
    "Blmngtn", "Blueste", "BrDale", "BrkSide", "ClearCr", "CollgCr",
    "Crawfor", "Edwards", "Gilbert", "IDOTRR", "MeadowV", "Mitchel",
    "NAmes", "NoRidge", "NPkVill", "NridgHt", "NWAmes", "OldTown",
    "SWISU", "Sawyer", "SawyerW", "Somerst", "StoneBr", "Timber", "Veenker",
)
VALID_NEIGHBORHOODS: frozenset[str] = frozenset(NEIGHBORHOODS)

# ---------------------------------------------------------------------------
# Numeric feature bounds — used by Pydantic validation and UI widget clamps
# ---------------------------------------------------------------------------

# (min, max) inclusive; derived from dataset range + domain knowledge.
NUMERIC_FEATURE_BOUNDS: dict[str, tuple[float, float]] = {
    "OverallQual":  (1,    10),
    "TotalSF":      (300,  12_000),
    "GarageCars":   (0,    4),
    "TotalBath":    (0.0,  7.0),
    "YearBuilt":    (1872, 2010),
    "TotalBsmtSF":  (0,    6_000),
}

# Valid discrete values for the GarageCars selectbox.
VALID_GARAGE_CARS: frozenset[int] = frozenset(range(5))  # 0–4

# ---------------------------------------------------------------------------
# Feature display labels — used by UI forms, tables, and chart axes
# ---------------------------------------------------------------------------

FEATURE_LABELS: dict[str, str] = {
    "OverallQual": "Overall Quality (1–10)",
    "TotalSF": "Total Floor Area (sqft)",
    "GarageCars": "Garage Capacity (cars)",
    "TotalBath": "Total Bathrooms",
    "YearBuilt": "Year Built",
    "TotalBsmtSF": "Basement Area (sqft)",
    "KitchenQual": "Kitchen Quality",
    "BsmtQual": "Basement Quality",
    "ExterQual": "Exterior Material Quality",
    "Neighborhood": "Neighborhood",
}

# Human-readable metadata consumed by the Streamlit UI and LLM prompts.
# Keys: description, type, unit (optional), min, max, example.
FEATURE_DEFINITIONS: dict[str, dict[str, str]] = {
    "OverallQual": {
        "description": "Overall material and finish quality",
        "type": "numeric",
        "unit": "rating 1–10",
        "min": "1",
        "max": "10",
        "example": "7",
    },
    "TotalSF": {
        "description": "Total floor area (basement + 1st floor + 2nd floor)",
        "type": "numeric",
        "unit": "sqft",
        "min": "500",
        "max": "10000",
        "example": "2000",
    },
    "GarageCars": {
        "description": "Garage capacity in number of cars",
        "type": "numeric",
        "unit": "cars",
        "min": "0",
        "max": "4",
        "example": "2",
    },
    "TotalBath": {
        "description": "Total bathrooms (full + 0.5×half, above and below grade)",
        "type": "numeric",
        "unit": "bathrooms",
        "min": "0",
        "max": "7",
        "example": "2.5",
    },
    "YearBuilt": {
        "description": "Original construction year",
        "type": "numeric",
        "unit": "year",
        "min": "1872",
        "max": "2010",
        "example": "1995",
    },
    "TotalBsmtSF": {
        "description": "Total basement area (0 if no basement)",
        "type": "numeric",
        "unit": "sqft",
        "min": "0",
        "max": "6000",
        "example": "1000",
    },
    "KitchenQual": {
        "description": "Kitchen quality",
        "type": "ordinal",
        "unit": "Po / Fa / TA / Gd / Ex",
        "min": "Po",
        "max": "Ex",
        "example": "Gd",
    },
    "BsmtQual": {
        "description": "Basement quality (None if no basement)",
        "type": "ordinal",
        "unit": "None / Po / Fa / TA / Gd / Ex",
        "min": "None",
        "max": "Ex",
        "example": "TA",
    },
    "ExterQual": {
        "description": "Exterior material quality",
        "type": "ordinal",
        "unit": "Po / Fa / TA / Gd / Ex",
        "min": "Po",
        "max": "Ex",
        "example": "TA",
    },
    "Neighborhood": {
        "description": "Physical location within Ames city limits",
        "type": "nominal",
        "unit": "neighborhood code",
        "min": "",
        "max": "",
        "example": "CollgCr",
    },
}
