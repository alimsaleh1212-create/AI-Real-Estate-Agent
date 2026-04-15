"""Predict router — feature definitions and price prediction pipeline."""

import logging
from typing import Any

from fastapi import APIRouter

from src.config import FEATURE_DEFINITIONS
from src.llm_chain import ExtractionError, extract_features, predict_and_interpret
from src.predictor import get_stats, predict_price
from src.schemas import (
    ExtractedFeatures,
    PredictionRequest,
    PredictionResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/features")
async def features() -> dict[str, Any]:
    """Return feature definitions for the UI gap-filling form.

    Returns:
        Dict mapping feature name → metadata (description, type, unit, example).
    """
    return FEATURE_DEFINITIONS


@router.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest) -> PredictionResponse:
    """Full prediction pipeline: LLM extraction → ML price → LLM interpretation.

    Returns a PredictionResponse with an ``error`` field on failure —
    never raises an unhandled exception to the caller.

    Args:
        request: Contains the user's natural-language property description.

    Returns:
        PredictionResponse with extracted_features, predicted_price,
        interpretation, and optional error string.
    """
    # Stage 1 — extract structured features from the query
    try:
        extracted = extract_features(request.query)
    except ExtractionError as exc:
        logger.error("Extraction failed for query=%r: %s", request.query[:60], exc)
        return PredictionResponse(
            query=request.query,
            extracted_features=ExtractedFeatures(),
            error=(
                "Could not parse property features from your description. "
                "Try being more specific (e.g. '3 beds, 2000 sqft, built 2001')."
            ),
        )
    except Exception as exc:
        logger.error("Unexpected extraction error: %s", exc)
        return PredictionResponse(
            query=request.query,
            extracted_features=ExtractedFeatures(),
            error="An unexpected error occurred. Please try again.",
        )

    # ML prediction — None fields imputed by pipeline's SimpleImputer
    try:
        price = predict_price(extracted)
    except Exception as exc:
        logger.error("Prediction failed: %s", exc)
        return PredictionResponse(
            query=request.query,
            extracted_features=extracted,
            error=f"Price prediction failed: {exc}",
        )

    # Stage 2 — generate plain-English interpretation
    try:
        stats = get_stats()
        interpretation = predict_and_interpret(extracted, price, stats)
    except Exception as exc:
        logger.error("Interpretation failed: %s", exc)
        return PredictionResponse(
            query=request.query,
            extracted_features=extracted,
            predicted_price=price,
            error="Price prediction succeeded but interpretation is unavailable.",
        )

    return PredictionResponse(
        query=request.query,
        extracted_features=extracted,
        predicted_price=price,
        interpretation=interpretation,
    )
