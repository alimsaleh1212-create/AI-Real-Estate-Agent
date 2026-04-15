"""Tests for src/predictor.py.

Covers: load_model, load_stats, get_stats, predict_price — all external I/O
is mocked so tests run without disk access or a trained model.
"""

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import src.predictor as predictor_module
from src.predictor import get_stats, load_model, load_stats, predict_price
from src.schemas import ExtractedFeatures


@pytest.fixture(autouse=True)
def reset_predictor_state() -> Any:
    """Reset module-level singletons before each test."""
    predictor_module._pipeline = None
    predictor_module._stats = None
    yield
    predictor_module._pipeline = None
    predictor_module._stats = None


# ---------------------------------------------------------------------------
# load_model
# ---------------------------------------------------------------------------


class TestLoadModel:
    def test_raises_if_file_missing(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="Model not found"):
            load_model(path=tmp_path / "missing.joblib")

    def test_sets_pipeline_on_success(self, tmp_path: Path) -> None:
        fake_path = tmp_path / "model.joblib"
        fake_path.touch()
        mock_pipeline = MagicMock()
        with patch("src.predictor.joblib.load", return_value=mock_pipeline):
            load_model(path=fake_path)
        assert predictor_module._pipeline is mock_pipeline

    def test_logs_success(self, tmp_path: Path, caplog: Any) -> None:
        fake_path = tmp_path / "model.joblib"
        fake_path.touch()
        with patch("src.predictor.joblib.load", return_value=MagicMock()):
            import logging
            with caplog.at_level(logging.INFO, logger="src.predictor"):
                load_model(path=fake_path)
        assert "Model loaded" in caplog.text


# ---------------------------------------------------------------------------
# load_stats
# ---------------------------------------------------------------------------


class TestLoadStats:
    def test_raises_if_file_missing(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="Stats not found"):
            load_stats(path=tmp_path / "missing.json")

    def test_sets_stats_on_success(self, tmp_path: Path) -> None:
        stats_data = {"sale_price_stats": {"median": 160000}}
        fake_path = tmp_path / "stats.json"
        fake_path.write_text(json.dumps(stats_data))
        load_stats(path=fake_path)
        assert predictor_module._stats is not None
        assert predictor_module._stats["sale_price_stats"]["median"] == 160000


# ---------------------------------------------------------------------------
# get_stats
# ---------------------------------------------------------------------------


class TestGetStats:
    def test_raises_if_not_loaded(self) -> None:
        with pytest.raises(RuntimeError, match="Stats not loaded"):
            get_stats()

    def test_returns_stats_when_loaded(self, tmp_path: Path) -> None:
        stats_data = {"sale_price_stats": {"median": 160000}}
        fake_path = tmp_path / "stats.json"
        fake_path.write_text(json.dumps(stats_data))
        load_stats(path=fake_path)
        result = get_stats()
        assert result["sale_price_stats"]["median"] == 160000


# ---------------------------------------------------------------------------
# predict_price
# ---------------------------------------------------------------------------


class TestPredictPrice:
    def _make_features(self, **kwargs: Any) -> ExtractedFeatures:
        return ExtractedFeatures(**kwargs)

    def test_raises_if_model_not_loaded(self) -> None:
        features = self._make_features()
        with pytest.raises(RuntimeError, match="Model not loaded"):
            predict_price(features)

    def test_returns_positive_float(self) -> None:
        mock_pipeline = MagicMock()
        # Pipeline predicts log1p(200000) ≈ 12.21; expm1 reverses to ~200000
        log_price = np.log1p(200000.0)
        mock_pipeline.predict.return_value = np.array([log_price])
        predictor_module._pipeline = mock_pipeline

        features = self._make_features(OverallQual=7, TotalSF=2000.0)
        result = predict_price(features)

        assert isinstance(result, float)
        assert result > 0

    def test_result_close_to_expected(self) -> None:
        mock_pipeline = MagicMock()
        log_price = np.log1p(180000.0)
        mock_pipeline.predict.return_value = np.array([log_price])
        predictor_module._pipeline = mock_pipeline

        features = self._make_features()
        result = predict_price(features)

        assert abs(result - 180000.0) < 1.0  # rounding tolerance

    def test_passes_correct_columns_to_pipeline(self) -> None:
        from src.config import SELECTED_FEATURES

        mock_pipeline = MagicMock()
        mock_pipeline.predict.return_value = np.array([np.log1p(150000.0)])
        predictor_module._pipeline = mock_pipeline

        features = self._make_features(OverallQual=5)
        predict_price(features)

        call_args = mock_pipeline.predict.call_args[0][0]
        assert list(call_args.columns) == SELECTED_FEATURES

    def test_raises_on_nonpositive_prediction(self) -> None:
        mock_pipeline = MagicMock()
        # expm1(-inf) → -1 which triggers the ValueError
        mock_pipeline.predict.return_value = np.array([-1000.0])
        predictor_module._pipeline = mock_pipeline

        with pytest.raises(ValueError, match="non-positive"):
            predict_price(self._make_features())

    def test_none_features_passed_as_nan_to_pipeline(self) -> None:
        import pandas as pd

        mock_pipeline = MagicMock()
        mock_pipeline.predict.return_value = np.array([np.log1p(150000.0)])
        predictor_module._pipeline = mock_pipeline

        features = self._make_features()  # all None
        predict_price(features)

        df_arg = mock_pipeline.predict.call_args[0][0]
        assert isinstance(df_arg, pd.DataFrame)
        # None values should be NaN in the DataFrame
        assert df_arg.isnull().all().all()
