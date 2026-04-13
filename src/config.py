"""Central configuration for the AI Real Estate Agent.

Loads environment variables, defines path constants, and declares
feature metadata. All other modules import from here — no scattered
os.getenv() calls elsewhere.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Security: fail fast if required secrets are missing
# ---------------------------------------------------------------------------

GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
if not GOOGLE_API_KEY:
    raise RuntimeError(
        "GOOGLE_API_KEY environment variable is not set. "
        "Copy .env.example to .env and add your Gemini API key."
    )

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

GEMINI_MODEL: str = "gemini-2.0-flash"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).parent.parent
DATA_RAW_PATH: Path = PROJECT_ROOT / "data" / "raw" / "AmesHousing.csv"
MODEL_PATH: Path = PROJECT_ROOT / "models" / "best_model_v1.joblib"
STATS_PATH: Path = PROJECT_ROOT / "models" / "training_stats.json"

# ---------------------------------------------------------------------------
# Feature metadata — populated after Block 2 (feature selection notebook).
# These are placeholders; update once statistical tests confirm the final 10.
# ---------------------------------------------------------------------------

# Maps feature name to its encoding type for the ColumnTransformer.
# "numeric"  → SimpleImputer(median) + StandardScaler
# "ordinal"  → SimpleImputer(most_frequent) + OrdinalEncoder
# "nominal"  → SimpleImputer(most_frequent) + OneHotEncoder
FEATURE_TYPES: dict[str, str] = {
    # Confirmed by statistical tests — see notebooks/eda_and_pipeline.ipynb
    # TODO: update after Block 2 feature selection is complete
}

# Ordered list of selected feature names (matches DataFrame column order).
SELECTED_FEATURES: list[str] = list(FEATURE_TYPES.keys())

# Human-readable metadata used by the UI and LLM prompts.
# Each entry: {description, type, unit, min, max, example}
FEATURE_DEFINITIONS: dict[str, dict[str, str]] = {
    # TODO: populate after Block 2 confirms the final 10 features
}