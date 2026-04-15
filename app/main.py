"""FastAPI application for the AI Real Estate Agent.

Routes:
    GET  /health    — liveness/readiness probe
    GET  /features  — feature definitions for the Streamlit gap-filling form
    POST /predict   — full pipeline: Stage 1 (LLM) → ML → Stage 2 (LLM)
    POST /insights  — bonus: intent-classify then answer a market question
"""

import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

from fastapi import FastAPI

from src.config import FEATURE_DEFINITIONS, get_google_api_key
from src.llm_chain import (
    ExtractionError,
    classify_intent,
    extract_features,
    generate_market_insights,
    predict_and_interpret,
)
from src.predictor import get_stats, load_model, load_stats, predict_price
from src.schemas import (
    ExtractedFeatures,
    InsightRequest,
    InsightResponse,
    PredictionRequest,
    PredictionResponse,
)

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Validate API key and load model + stats at startup."""
    get_google_api_key()  # raises RuntimeError if GOOGLE_API_KEY not set
    load_model()
    load_stats()
    logger.info("AI Real Estate Agent ready.")
    yield
    logger.info("AI Real Estate Agent shutting down.")


app = FastAPI(
    title="AI Real Estate Agent",
    description=(
        "Predict Ames, Iowa home prices from natural-language property "
        "descriptions using Gemini LLM feature extraction and a "
        "GradientBoosting ML model."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> dict[str, str]:
    """Liveness/readiness probe — returns 200 when the server is running."""
    return {"status": "ok"}


@app.get("/features")
async def features() -> dict[str, Any]:
    """Return feature definitions for the UI gap-filling form.

    Returns:
        Dict mapping feature name → metadata (description, type, unit, example).
    """
    return FEATURE_DEFINITIONS


@app.post("/predict", response_model=PredictionResponse)
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
        # Return the price even when interpretation fails
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


@app.post("/insights", response_model=InsightResponse)
async def insights(request: InsightRequest) -> InsightResponse:
    """Bonus: classify intent and answer a market question from training stats.

    Redirects prediction-intent queries to the /predict endpoint rather
    than attempting to generate market insights from them.

    Args:
        request: Contains a natural-language market question.

    Returns:
        InsightResponse with intent, answer, and optional error.
    """
    intent = classify_intent(request.query)

    if intent == "prediction":
        return InsightResponse(
            query=request.query,
            intent=intent,
            answer=(
                "Your query looks like a price prediction request. "
                "Send it to the /predict endpoint instead."
            ),
        )

    try:
        stats = get_stats()
        answer = generate_market_insights(request.query, stats)
    except Exception as exc:
        logger.error("Market insights failed: %s", exc)
        return InsightResponse(
            query=request.query,
            intent=intent,
            error=f"Could not generate market insights: {exc}",
        )

    return InsightResponse(
        query=request.query,
        intent=intent,
        answer=answer,
    )
