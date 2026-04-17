"""Shared domain schema for the AI Real Estate Agent.

Defines the central data contract between LLM Stage 1, the ML predictor,
and LLM Stage 2. FastAPI request/response schemas live in app/schemas.py.

Schemas:
    ExtractedFeatures — Stage 1 output; 10 optional feature fields.
"""

import logging
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, model_validator

from src.config import (
    BSMT_QUALITY_CODES,
    NUMERIC_FEATURE_BOUNDS,
    QUALITY_CODES,
    SELECTED_FEATURES,
    VALID_NEIGHBORHOODS,
)

logger = logging.getLogger(__name__)


class Confidence(str, Enum):
    """Extraction status for a single feature."""

    EXTRACTED = "extracted"
    MISSING = "missing"


# Feature names in the order used by the ML pipeline.
_FEATURE_NAMES: list[str] = SELECTED_FEATURES


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
        description="Overall quality rating (1=Very Poor, 10=Very Excellent)",
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

    @model_validator(mode="before")
    @classmethod
    def _coerce_out_of_range_to_none(cls, data: Any) -> Any:
        """Nullify any field whose extracted value falls outside its domain-valid range.

        Runs before Pydantic's per-field validation so out-of-range values (e.g.
        OverallQual=12, TotalSF=130, KitchenQual="excellent") do not raise a
        ValidationError and abort the entire extraction.  Each field is treated as
        missing; the UI will show it as unextracted and prompt the user to fill it in.

        Covers all 10 extracted features:
          - Numeric: bounds from dataset/config (min, max inclusive)
          - Ordinal: must be one of the accepted code strings
          - Nominal: Neighborhood must be a known Ames neighbourhood code
        """
        if not isinstance(data, dict):
            return data

        # ── Numeric bounds (inclusive) — sourced from config.NUMERIC_FEATURE_BOUNDS ──
        for field, (lo, hi) in NUMERIC_FEATURE_BOUNDS.items():
            val = data.get(field)
            if val is None:
                continue
            try:
                v = float(val)
            except (TypeError, ValueError):
                logger.info("Extracted %s=%r is not numeric — treating as missing", field, val)
                data[field] = None
                continue
            if v < lo or v > hi:
                logger.info(
                    "Extracted %s=%s is outside domain range [%s, %s] — treating as missing",
                    field, val, lo, hi,
                )
                data[field] = None

        # ── Ordinal valid sets — sourced from config quality code lists ──────
        _QUALITY_SET = frozenset(QUALITY_CODES)
        _BSMT_SET    = frozenset(BSMT_QUALITY_CODES)
        _ORDINAL_CHECKS: dict[str, frozenset[str]] = {
            "KitchenQual": _QUALITY_SET,
            "BsmtQual":    _BSMT_SET,
            "ExterQual":   _QUALITY_SET,
        }
        for field, valid_set in _ORDINAL_CHECKS.items():
            val = data.get(field)
            if val is not None and val not in valid_set:
                logger.info(
                    "Extracted %s=%r is not a valid ordinal code %s — treating as missing",
                    field, val, sorted(valid_set),
                )
                data[field] = None

        # ── Neighborhood valid set — sourced from config.VALID_NEIGHBORHOODS ─
        nbhd = data.get("Neighborhood")
        if nbhd is not None and nbhd not in VALID_NEIGHBORHOODS:
            logger.info(
                "Extracted Neighborhood=%r is not a known Ames neighbourhood — treating as missing",
                nbhd,
            )
            data["Neighborhood"] = None

        return data

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
