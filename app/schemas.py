"""FastAPI request/response schemas for the AI Real Estate Agent.

These Pydantic models define the API contracts for each endpoint.
The shared domain model (ExtractedFeatures) lives in src/schemas.py
because it is consumed by both the src/ layer and this app/ layer.
"""

from typing import Optional

from pydantic import BaseModel, Field

from src.schemas import ExtractedFeatures


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
    """Input to the POST /insights endpoint."""

    query: str = Field(
        ...,
        min_length=3,
        description="Natural-language market question (e.g. 'avg price in NridgHt?').",
    )


class InsightResponse(BaseModel):
    """Output from the POST /insights endpoint."""

    query: str
    intent: str = Field(description="Classified intent: 'prediction' or 'analysis'.")
    answer: Optional[str] = None
    error: Optional[str] = None
