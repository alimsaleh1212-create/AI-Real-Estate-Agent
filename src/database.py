"""Supabase persistence layer for the AI Real Estate Agent.

Logs every prediction and insight query to Supabase tables so that
usage can be analysed and predictions can be audited.

The client is initialised lazily from SUPABASE_URL and SUPABASE_KEY.
If either variable is missing the module silently no-ops — the app
works normally without Supabase configured.

Tables (created via the schema in docs/supabase_schema.sql):
    predictions — one row per /predict pipeline run
    insights    — one row per /insights query
"""

import logging
import os
from typing import Any, Optional

from src.schemas import ExtractedFeatures

logger = logging.getLogger(__name__)

# Module-level singleton; None until first successful initialisation.
_client: Optional[Any] = None


def _get_client() -> Optional[Any]:
    """Return a cached Supabase client, or None if not configured.

    Returns:
        Supabase client instance, or None when env vars are absent.
    """
    global _client
    if _client is not None:
        return _client

    url = os.getenv("SUPABASE_URL", "")
    key = os.getenv("SUPABASE_KEY", "")
    if not url or not key:
        return None

    try:
        from supabase import create_client  # imported lazily
        _client = create_client(url, key)
        logger.info("Supabase client initialised (url=%s…)", url[:30])
    except Exception as exc:
        logger.warning("Could not initialise Supabase client: %s", exc)

    return _client


def log_prediction(
    query: str,
    extracted: ExtractedFeatures,
    predicted_price: Optional[float],
    interpretation: Optional[str],
    error: Optional[str],
) -> None:
    """Insert one row into the predictions table.

    Never raises — a failed DB write must not break the prediction flow.

    Args:
        query: User's original natural-language description.
        extracted: Stage 1 output (feature values, some may be None).
        predicted_price: ML model output in dollars, or None on failure.
        interpretation: LLM Stage 2 text, or None on failure.
        error: Error message when the pipeline failed gracefully.
    """
    client = _get_client()
    if client is None:
        return

    record: dict[str, Any] = {
        "query": query,
        "overall_qual": extracted.OverallQual,
        "total_sf": extracted.TotalSF,
        "garage_cars": extracted.GarageCars,
        "total_bath": extracted.TotalBath,
        "year_built": extracted.YearBuilt,
        "total_bsmt_sf": extracted.TotalBsmtSF,
        "kitchen_qual": extracted.KitchenQual,
        "bsmt_qual": extracted.BsmtQual,
        "exter_qual": extracted.ExterQual,
        "neighborhood": extracted.Neighborhood,
        "extracted_count": len(extracted.extracted_features),
        "predicted_price": predicted_price,
        "interpretation": interpretation,
        "error": error,
    }

    try:
        client.table("predictions").insert(record).execute()
        logger.debug("Logged prediction to Supabase (price=%s)", predicted_price)
    except Exception as exc:
        logger.warning("Failed to log prediction to Supabase: %s", exc)


def log_insight(
    query: str,
    intent: str,
    answer: Optional[str],
    error: Optional[str],
) -> None:
    """Insert one row into the insights table.

    Never raises — a failed DB write must not break the insights flow.

    Args:
        query: User's natural-language market question.
        intent: Classified intent ('analysis' or 'prediction').
        answer: LLM-generated market answer, or None on failure.
        error: Error message when insight generation failed.
    """
    client = _get_client()
    if client is None:
        return

    record: dict[str, Any] = {
        "query": query,
        "intent": intent,
        "answer": answer,
        "error": error,
    }

    try:
        client.table("insights").insert(record).execute()
        logger.debug("Logged insight to Supabase (intent=%s)", intent)
    except Exception as exc:
        logger.warning("Failed to log insight to Supabase: %s", exc)
