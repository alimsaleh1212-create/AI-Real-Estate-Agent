"""Pydantic schemas for the AI Real Estate Agent.

Defines the data contracts between LLM Stage 1, the ML model,
LLM Stage 2, and the FastAPI layer. All I/O validation happens here.

Schemas:
    ExtractedFeatures   — Stage 1 output; 10 optional feature fields.
    PredictionRequest   — /predict endpoint input.
    PredictionResponse  — /predict endpoint output.
    InsightRequest      — /insights endpoint input (bonus).
    InsightResponse     — /insights endpoint output (bonus).
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, model_validator


class Confidence(str, Enum):
    """Extraction status for a single feature."""

    EXTRACTED = "extracted"
    MISSING = "missing"


# Feature names in the order used by the ML pipeline.
_FEATURE_NAMES: list[str] = [
    "OverallQual",
    "TotalSF",
    "GarageCars",
    "TotalBath",
    "YearBuilt",
    "TotalBsmtSF",
    "KitchenQual",
    "BsmtQual",
    "ExterQual",
    "Neighborhood",
]


class ExtractedFeatures(BaseModel):
    """Features extracted from a natural-language property description.

    Stage 1 (Gemini) produces a JSON object with these fields. Each field
    is Optional — None means the feature was not mentioned in the query
    and will be filled by the pipeline's median/mode imputer at inference.

    The ``confidence`` field is computed automatically after validation;
    it must not be included in the LLM response JSON.
    """

    # ------------------------------------------------------------------
    # Numeric features
    # ------------------------------------------------------------------

    OverallQual: Optional[int] = Field(
        default=None,
        ge=1,
        le=10,
        description="Overall material and finish quality (1=Very Poor … 10=Very Excellent)",
    )
    TotalSF: Optional[float] = Field(
        default=None,
        ge=0,
        description="Total floor area: basement + 1st floor + 2nd floor (sqft)",
    )
    GarageCars: Optional[int] = Field(
        default=None,
        ge=0,
        le=5,
        description="Garage capacity in number of cars (0 = no garage)",
    )
    TotalBath: Optional[float] = Field(
        default=None,
        ge=0,
        description="Total bathrooms (full + 0.5×half, above and below grade)",
    )
    YearBuilt: Optional[int] = Field(
        default=None,
        ge=1800,
        le=2025,
        description="Original construction year",
    )
    TotalBsmtSF: Optional[float] = Field(
        default=None,
        ge=0,
        description="Total basement area in sqft (0 if no basement)",
    )

    # ------------------------------------------------------------------
    # Ordinal features  (valid values: Po / Fa / TA / Gd / Ex)
    # BsmtQual also accepts "None" (no basement present)
    # ------------------------------------------------------------------

    KitchenQual: Optional[str] = Field(
        default=None,
        description="Kitchen quality: Po / Fa / TA / Gd / Ex",
    )
    BsmtQual: Optional[str] = Field(
        default=None,
        description="Basement quality: None / Po / Fa / TA / Gd / Ex",
    )
    ExterQual: Optional[str] = Field(
        default=None,
        description="Exterior material quality: Po / Fa / TA / Gd / Ex",
    )

    # ------------------------------------------------------------------
    # Nominal feature
    # ------------------------------------------------------------------

    Neighborhood: Optional[str] = Field(
        default=None,
        description="Physical location within Ames city limits (e.g. CollgCr, NridgHt)",
    )

    # ------------------------------------------------------------------
    # Computed field — populated by model_validator, not by the LLM
    # ------------------------------------------------------------------

    confidence: dict[str, Confidence] = Field(
        default_factory=dict,
        description="Per-feature extraction status; computed automatically.",
    )

    @model_validator(mode="after")
    def _compute_confidence(self) -> "ExtractedFeatures":
        """Populate confidence from which feature fields are non-None."""
        self.confidence = {
            name: (
                Confidence.EXTRACTED
                if getattr(self, name) is not None
                else Confidence.MISSING
            )
            for name in _FEATURE_NAMES
        }
        return self

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def missing_features(self) -> list[str]:
        """Feature names not mentioned in the query."""
        return [k for k, v in self.confidence.items() if v == Confidence.MISSING]

    @property
    def extracted_features(self) -> list[str]:
        """Feature names successfully extracted from the query."""
        return [k for k, v in self.confidence.items() if v == Confidence.EXTRACTED]

    @property
    def is_complete(self) -> bool:
        """True when all 10 features have been extracted or filled by the user."""
        return len(self.missing_features) == 0

    def to_feature_dict(self) -> dict[str, object]:
        """Return a plain dict of feature name → value (None for missing).

        Used to build the pandas DataFrame fed to the ML model. None values
        become NaN, which the Pipeline's SimpleImputer handles.
        """
        return {name: getattr(self, name) for name in _FEATURE_NAMES}


# ---------------------------------------------------------------------------
# Request / Response schemas for FastAPI endpoints
# ---------------------------------------------------------------------------


class PredictionRequest(BaseModel):
    """Input to the POST /predict endpoint."""

    query: str = Field(
        ...,
        min_length=3,
        description="Natural-language property description from the user.",
    )


class PredictionResponse(BaseModel):
    """Output from the POST /predict endpoint.

    ``error`` is non-None only when the pipeline failed gracefully;
    in that case predicted_price and interpretation will be None.
    """

    query: str
    extracted_features: ExtractedFeatures
    predicted_price: Optional[float] = None
    interpretation: Optional[str] = None
    error: Optional[str] = None


class InsightRequest(BaseModel):
    """Input to the POST /insights endpoint (bonus market-analysis route)."""

    query: str = Field(
        ...,
        min_length=3,
        description="Natural-language market question (e.g. 'average price in NridgHt?').",
    )


class InsightResponse(BaseModel):
    """Output from the POST /insights endpoint."""

    query: str
    intent: str = Field(description="Classified intent: 'prediction' or 'analysis'.")
    answer: Optional[str] = None
    error: Optional[str] = None
