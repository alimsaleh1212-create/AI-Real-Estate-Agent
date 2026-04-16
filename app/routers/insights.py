"""Insights router — intent classification and market analysis."""

import logging

from fastapi import APIRouter

from src.database import log_insight
from src.llm_chain import classify_intent, generate_market_insights
from src.predictor import get_stats

from app.schemas import InsightRequest, InsightResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/insights", response_model=InsightResponse)
async def insights(request: InsightRequest) -> InsightResponse:
    """Classify intent and answer a market question from training stats.

    Redirects prediction-intent queries to the /predict endpoint rather
    than attempting to generate market insights from them.

    Args:
        request: Contains a natural-language market question.

    Returns:
        InsightResponse with intent, answer, and optional error.
    """
    intent = classify_intent(request.query)

    if intent == "prediction":
        answer = (
            "Your query looks like a price prediction request. "
            "Send it to the /predict endpoint instead."
        )
        log_insight(request.query, intent, answer, None)
        return InsightResponse(query=request.query, intent=intent, answer=answer)

    try:
        stats = get_stats()
        answer = generate_market_insights(request.query, stats)
    except Exception as exc:
        logger.error("Market insights failed: %s", exc)
        err = f"Could not generate market insights: {exc}"
        log_insight(request.query, intent, None, err)
        return InsightResponse(query=request.query, intent=intent, error=err)

    log_insight(request.query, intent, answer, None)
    return InsightResponse(query=request.query, intent=intent, answer=answer)
