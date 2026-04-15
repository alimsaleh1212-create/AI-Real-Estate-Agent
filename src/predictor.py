"""Predictor module — loads trained model at startup and runs inference.

Maintains two module-level singletons initialised once at startup:
    _pipeline   — fitted sklearn Pipeline (preprocessor + GBR model)
    _stats      — training statistics dict from training_stats.json

Call load_model() and load_stats() once before any predict_price() call.
"""

import json
import logging
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.config import MODEL_PATH, SELECTED_FEATURES, STATS_PATH
from src.schemas import ExtractedFeatures

logger = logging.getLogger(__name__)

_pipeline: Optional[Pipeline] = None
_stats: Optional[dict[str, Any]] = None


def load_model(path: Path = MODEL_PATH) -> None:
    """Load the trained sklearn Pipeline from disk into module state.

    Thread-safe for read-only access after loading. Call once at startup.

    Args:
        path: Path to the serialised .joblib file.

    Raises:
        FileNotFoundError: If the model file does not exist.
    """
    global _pipeline
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found: {path}. "
            "Run `uv run python -m src.ml_pipeline` to train and save."
        )
    _pipeline = joblib.load(path)
    logger.info("Model loaded from %s", path)


def load_stats(path: Path = STATS_PATH) -> None:
    """Load training statistics from a JSON file into module state.

    Args:
        path: Path to training_stats.json.

    Raises:
        FileNotFoundError: If the stats file does not exist.
    """
    global _stats
    if not path.exists():
        raise FileNotFoundError(
            f"Stats not found: {path}. "
            "Run `uv run python -m src.ml_pipeline` to generate."
        )
    with open(path) as f:
        _stats = json.load(f)
    logger.info("Training stats loaded from %s", path)


def get_stats() -> dict[str, Any]:
    """Return the loaded training stats dict.

    Returns:
        Dict with sale_price_stats and model metadata.

    Raises:
        RuntimeError: If load_stats() has not been called.
    """
    if _stats is None:
        raise RuntimeError("Stats not loaded. Call load_stats() at startup.")
    return _stats


def predict_price(features: ExtractedFeatures) -> float:
    """Predict sale price from extracted property features.

    None fields in ExtractedFeatures become NaN in the DataFrame; the
    pipeline's SimpleImputer fills them with training-set medians/modes.
    Target was log1p-transformed during training — expm1 reverses this.

    Args:
        features: Validated ExtractedFeatures (may have None fields).

    Returns:
        Predicted sale price in USD.

    Raises:
        RuntimeError: If load_model() has not been called.
        ValueError: If the model predicts a non-positive price.
    """
    if _pipeline is None:
        raise RuntimeError("Model not loaded. Call load_model() at startup.")

    row = features.to_feature_dict()
    df = pd.DataFrame([row], columns=SELECTED_FEATURES)

    log_pred: "np.ndarray[Any, np.dtype[np.floating[Any]]]" = _pipeline.predict(df)
    price = float(np.expm1(log_pred[0]))

    if price <= 0:
        raise ValueError(f"Model returned non-positive price: {price}")

    logger.debug(
        "Predicted $%.0f (extracted=%d/10)",
        price,
        len(features.extracted_features),
    )
    return price
