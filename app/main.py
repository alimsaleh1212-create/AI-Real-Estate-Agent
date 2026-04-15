"""FastAPI application factory for the AI Real Estate Agent.

Routes are split across router modules:
    app/routers/health.py   — GET  /health
    app/routers/predict.py  — GET  /features, POST /predict
    app/routers/insights.py — POST /insights
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI

from src.config import get_google_api_key
from src.predictor import load_model, load_stats

from app.routers import health, insights, predict

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

app.include_router(health.router)
app.include_router(predict.router)
app.include_router(insights.router)
