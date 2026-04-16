"""LLM chain for the AI Real Estate Agent.

Wraps the Google Gemini API for two-stage processing:
  Stage 1 — extract structured features from a natural-language description.
  Stage 2 — interpret a price prediction in plain English.

Also exposes bonus functions for intent classification and market insights.

The Gemini client is lazily initialised on first call so that modules that
import llm_chain in tests can mock the client before it is created.
"""

import json
import logging
import re
from functools import lru_cache
from typing import Any

import google.generativeai as genai

from src.config import GEMINI_MODEL, get_google_api_key
from src.prompts import (
    EXTRACTION_PROMPT_V1,
    EXTRACTION_PROMPT_V2,
    INSIGHTS_PROMPT,
    INTENT_PROMPT,
    INTERPRETATION_PROMPT,
)
from src.schemas import ExtractedFeatures

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class ExtractionError(Exception):
    """Raised when Stage 1 feature extraction fails after retries."""


class InterpretationError(Exception):
    """Raised when Stage 2 interpretation generation fails."""


# ---------------------------------------------------------------------------
# Lazy Gemini client
# ---------------------------------------------------------------------------

# google-generativeai has no type stubs; use Any for the model handle.
_model: Any = None


def _get_model() -> Any:  # -> genai.GenerativeModel
    """Return the module-level Gemini client, initialising it on first call.

    Returns:
        Configured GenerativeModel instance.
    """
    global _model
    if _model is None:
        genai.configure(api_key=get_google_api_key())  # type: ignore[attr-defined]
        _model = genai.GenerativeModel(GEMINI_MODEL)  # type: ignore[attr-defined]
        logger.debug("Gemini client initialised (model=%s)", GEMINI_MODEL)
    return _model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EXTRACTION_CONFIG: Any = genai.GenerationConfig(  # type: ignore[attr-defined]
    temperature=0.0,  # deterministic for structured extraction
    response_mime_type="application/json",
    max_output_tokens=1024,  # 10-field JSON ≈ 60 tokens; 1024 gives headroom without exfiltration risk
)

_TEXT_CONFIG: Any = genai.GenerationConfig(  # type: ignore[attr-defined]
    temperature=0.7,
    max_output_tokens=1024,  # 3-4 sentences needs ~150 tokens; 1024 is a generous cap
)


def _call_gemini(prompt: str, config: Any) -> str:
    """Send a prompt to Gemini and return the raw text response.

    Args:
        prompt: The complete prompt string.
        config: GenerationConfig controlling temperature and response format.

    Returns:
        Raw text from the model response.

    Raises:
        Exception: Re-raises any API-level errors after logging.
    """
    try:
        response = _get_model().generate_content(prompt, generation_config=config)
        return str(response.text)
    except Exception as exc:
        logger.error("Gemini API call failed: %s", exc)
        raise


@lru_cache(maxsize=256)
def _cached_extraction(prompt: str) -> str:
    """LRU-cached wrapper for deterministic (temperature=0) extraction calls.

    Caches up to 256 distinct prompts in memory. Safe to cache because the
    extraction config uses temperature=0.0, so the response is deterministic.
    Repeated identical descriptions (common in testing and demos) skip the
    Gemini round-trip entirely.

    Args:
        prompt: Fully-formatted extraction prompt (query already embedded).

    Returns:
        Raw Gemini response text.
    """
    logger.debug("LRU cache miss — calling Gemini for extraction prompt")
    return _call_gemini(prompt, _EXTRACTION_CONFIG)


@lru_cache(maxsize=512)
def _cached_intent(prompt: str) -> str:
    """LRU-cached wrapper for intent classification calls.

    Intent classification is deterministic and cheap, but fired on every
    query. Caching means repeated questions (e.g. 'What is the average
    price?') never touch the API after the first call.

    Args:
        prompt: Fully-formatted intent prompt.

    Returns:
        Raw Gemini response text.
    """
    logger.debug("LRU cache miss — calling Gemini for intent prompt")
    return _call_gemini(prompt, _TEXT_CONFIG)


@lru_cache(maxsize=256)
def _cached_insights(prompt: str) -> str:
    """LRU-cached wrapper for market-insights calls.

    Market questions are often repeated ('median price?', 'best neighborhood?').
    Caching avoids redundant LLM calls for identical question+stats combos.

    Args:
        prompt: Fully-formatted insights prompt (query + stats already embedded).

    Returns:
        Raw Gemini response text.
    """
    logger.debug("LRU cache miss — calling Gemini for insights prompt")
    return _call_gemini(prompt, _TEXT_CONFIG)


def _parse_extraction_response(raw: str) -> dict[str, Any]:
    """Parse a JSON string returned by Stage 1.

    Handles models that wrap the JSON in markdown code fences despite
    response_mime_type='application/json' being set.

    Also recovers from truncated JSON (e.g. ``'{"OverallQual": null, "'``)
    by scanning backwards for the last complete key-value pair, closing the
    object there, and parsing what was salvaged.  Any recovered value that is
    out-of-range will be caught by the ExtractedFeatures model_validator.

    Args:
        raw: Raw text from Gemini.

    Returns:
        Parsed dict (may be partial if the response was truncated).

    Raises:
        json.JSONDecodeError: If the text cannot be parsed even after recovery.
    """
    text = raw.strip()
    # Strip optional markdown fence
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(line for line in lines if not line.startswith("```")).strip()

    # Primary parse — the normal path
    try:
        result: dict[str, Any] = json.loads(text)
        return result
    except json.JSONDecodeError:
        pass

    # Fallback: truncated JSON recovery.
    # Strategy 1: just close the object (works when the last value is complete
    #   but the closing brace is missing, e.g. '{"OverallQual": null').
    # Strategy 2: walk backwards for the last comma that ends a complete
    #   key-value pair, close there (handles '{"OverallQual": null, "...').
    logger.warning(
        "Primary JSON parse failed — attempting truncated-response recovery "
        "(raw length=%d, preview=%r)",
        len(text),
        text[:80],
    )

    # Strategy 1: append closing brace
    candidate = text.rstrip().rstrip(",") + "}"
    try:
        recovered: dict[str, Any] = json.loads(candidate)
        logger.info(
            "Truncated JSON recovered (strategy 1): kept %d key(s)",
            len(recovered),
        )
        return recovered
    except json.JSONDecodeError:
        pass

    # Strategy 2: scan backwards for last comma between complete pairs
    for i in range(len(text) - 1, 0, -1):
        if text[i] == ",":
            candidate = text[:i] + "}"
            try:
                recovered = json.loads(candidate)
                logger.info(
                    "Truncated JSON recovered (strategy 2): kept %d key(s)",
                    len(recovered),
                )
                return recovered
            except json.JSONDecodeError:
                continue

    # Nothing salvageable — re-raise with the original error
    raise json.JSONDecodeError(
        "Could not parse or recover JSON from Gemini response",
        text,
        0,
    )


def _format_features_text(features: ExtractedFeatures) -> str:
    """Render extracted features as a readable bullet list for Stage 2.

    Args:
        features: Validated ExtractedFeatures instance.

    Returns:
        Multi-line string of feature: value pairs (missing features omitted).
    """
    lines = []
    for name in features.extracted_features:
        value = getattr(features, name)
        # Sanitize string values before Stage 2 embedding (secondary injection)
        if isinstance(value, str):
            value = _sanitize_feature_string(value)
        lines.append(f"  - {name}: {value}")
    if features.missing_features:
        lines.append(
            f"  - (imputed from training data: {', '.join(features.missing_features)})"
        )
    return "\n".join(lines)


_MAX_QUERY_LEN: int = 500
_INJECTION_PATTERN: re.Pattern[str] = re.compile(
    r"(ignore|disregard|override|forget|system|instruction|prompt|previous)\s",
    re.IGNORECASE,
)
_SAFE_FEATURE_CHARS: re.Pattern[str] = re.compile(r"[^\w\s,.\-]")


def _sanitize_query(query: str) -> str:
    """Truncate and strip control characters from a user query before prompt injection.

    Caps length, removes null bytes and carriage returns that could break prompt
    structure, and logs a warning if injection-pattern keywords are detected.
    Does not reject queries — logs only, so legitimate edge-case phrasing still works.

    Args:
        query: Raw user input string.

    Returns:
        Sanitized string safe for interpolation into any prompt template.
    """
    sanitized = query[:_MAX_QUERY_LEN]
    sanitized = sanitized.replace("\x00", "").replace("\r", " ").replace("\t", " ")
    if _INJECTION_PATTERN.search(sanitized):
        logger.warning(
            "Suspicious injection pattern detected in query (first 60 chars): %s",
            sanitized[:60],
        )
    return sanitized


def _sanitize_feature_string(value: str) -> str:
    """Strip characters outside alphanumeric and basic punctuation from a feature value.

    Prevents secondary prompt injection when Stage 1 string outputs (e.g., Neighborhood)
    are embedded into Stage 2 prompts. Ordinal values are already enum-validated by
    Pydantic; this targets free-text fields.

    Args:
        value: String feature value produced by Stage 1 extraction.

    Returns:
        Sanitized string containing only word chars, spaces, commas, dots, hyphens.
    """
    return _SAFE_FEATURE_CHARS.sub("", value)


def _format_stats_text(stats: dict[str, Any]) -> str:
    """Render training stats as a readable block for Stage 2 / Insights.

    Args:
        stats: Dict from training_stats.json (expects sale_price_stats key).

    Returns:
        Multi-line string of stat: value pairs.
    """
    sp = stats.get("sale_price_stats", stats)
    return (
        f"  Median sale price : ${sp['median']:,.0f}\n"
        f"  Mean sale price   : ${sp['mean']:,.0f}\n"
        f"  Std deviation     : ${sp['std']:,.0f}\n"
        f"  Min sale price    : ${sp['min']:,.0f}\n"
        f"  Max sale price    : ${sp['max']:,.0f}\n"
        f"  25th percentile   : ${sp['q25']:,.0f}\n"
        f"  75th percentile   : ${sp['q75']:,.0f}"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_features(
    query: str,
    prompt_version: str = "v2",
) -> ExtractedFeatures:
    """Stage 1 — extract structured property features from a natural-language query.

    Uses response_mime_type='application/json' to guide Gemini output.
    Retries once on JSON parse or Pydantic validation failure before raising.

    Args:
        query: Raw user description (e.g. "3-bed ranch, big garage, built 2001").
        prompt_version: "v1" (direct instruction) or "v2" (few-shot, default).

    Returns:
        ExtractedFeatures with non-None values for each mentioned feature and
        auto-computed confidence dict.

    Raises:
        ExtractionError: If both attempts fail to produce a valid schema.
    """
    template = EXTRACTION_PROMPT_V1 if prompt_version == "v1" else EXTRACTION_PROMPT_V2
    prompt = template.format(query=_sanitize_query(query))

    last_exc: Exception = ExtractionError("No attempts made")
    for attempt in range(1, 3):  # two attempts total
        try:
            # Attempt 1 uses the LRU cache (temperature=0, response is deterministic).
            # Attempt 2 bypasses the cache — the cached response was already invalid.
            raw = _cached_extraction(prompt) if attempt == 1 else _call_gemini(prompt, _EXTRACTION_CONFIG)
            data = _parse_extraction_response(raw)
            features = ExtractedFeatures(**data)
            logger.info(
                "Extraction OK (attempt=%d, version=%s, extracted=%d/10)",
                attempt,
                prompt_version,
                len(features.extracted_features),
            )
            return features
        except (json.JSONDecodeError, ValueError) as exc:
            last_exc = exc
            logger.warning(
                "Extraction attempt %d failed (parse error): %s — retrying",
                attempt,
                exc,
            )
        except Exception as exc:
            # API errors (rate limit, network) — do not retry
            raise ExtractionError(f"Gemini API error: {exc}") from exc

    raise ExtractionError(
        f"Feature extraction failed after 2 attempts: {last_exc}"
    ) from last_exc


def predict_and_interpret(
    features: ExtractedFeatures,
    predicted_price: float,
    stats: dict[str, Any],
) -> str:
    """Stage 2 — generate a plain-English interpretation of a price prediction.

    Args:
        features: Extracted (and optionally gap-filled) property features.
        predicted_price: ML model output in dollars.
        stats: Training stats dict (from training_stats.json).

    Returns:
        3–4 sentence narrative suitable for display in the UI.

    Raises:
        InterpretationError: If Gemini fails to generate a response.
    """
    sp = stats.get("sale_price_stats", stats)
    prompt = INTERPRETATION_PROMPT.format(
        predicted_price=predicted_price,
        features_text=_format_features_text(features),
        median=sp["median"],
        mean=sp["mean"],
        q25=sp["q25"],
        q75=sp["q75"],
        price_min=sp["min"],
        price_max=sp["max"],
    )

    try:
        interpretation = _call_gemini(prompt, _TEXT_CONFIG).strip()
        logger.info(
            "Interpretation generated (price=$%.0f, length=%d chars)",
            predicted_price,
            len(interpretation),
        )
        return interpretation
    except Exception as exc:
        raise InterpretationError(f"Interpretation failed: {exc}") from exc


def classify_intent(query: str) -> str:
    """Bonus — classify a query as 'prediction' or 'analysis'.

    Args:
        query: Raw user query.

    Returns:
        'prediction' or 'analysis' (lowercase, stripped).

    Raises:
        ValueError: If the model returns an unexpected value.
    """
    prompt = INTENT_PROMPT.format(query=_sanitize_query(query))
    try:
        raw = _cached_intent(prompt).strip().lower()
    except Exception as exc:
        logger.error("Intent classification failed: %s", exc)
        return "prediction"  # safe default

    intent = raw.split()[0] if raw else "prediction"
    if intent not in ("prediction", "analysis"):
        logger.warning("Unexpected intent '%s', defaulting to 'prediction'", intent)
        intent = "prediction"

    logger.info("Intent classified: '%s' for query='%s'", intent, query[:60])
    return intent


def generate_market_insights(query: str, stats: dict[str, Any]) -> str:
    """Bonus — answer a market question using pre-computed training stats.

    Args:
        query: Natural-language market question.
        stats: Training stats dict (from training_stats.json).

    Returns:
        2–3 sentence answer citing specific numbers.

    Raises:
        Exception: Re-raises API errors after logging.
    """
    prompt = INSIGHTS_PROMPT.format(
        query=_sanitize_query(query),
        stats_text=_format_stats_text(stats),
    )
    try:
        answer = _cached_insights(prompt).strip()
        logger.info("Market insight generated (%d chars)", len(answer))
        return answer
    except Exception as exc:
        logger.error("Market insights failed: %s", exc)
        raise
