"""Tests for src/ml_pipeline.py and src/config.py.

ml_pipeline tests cover the importable helper functions (build_preprocessor,
build_pipeline, _compute_metrics, compute_training_stats, save_model,
save_training_stats). The __main__ script entry-point is not tested here —
it is verified by running the notebook.

config tests cover get_google_api_key() success/failure paths.
"""

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.config import get_google_api_key
from src.ml_pipeline import (
    _compute_metrics,
    build_pipeline,
    build_preprocessor,
    compute_training_stats,
    save_model,
    save_training_stats,
)


# ---------------------------------------------------------------------------
# src/config.py — get_google_api_key
# ---------------------------------------------------------------------------


class TestGetGoogleApiKey:
    def test_returns_key_when_set(self, monkeypatch: Any) -> None:
        monkeypatch.setenv("GOOGLE_API_KEY", "test-api-key-123")
        assert get_google_api_key() == "test-api-key-123"

    def test_raises_when_key_missing(self, monkeypatch: Any) -> None:
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="GOOGLE_API_KEY"):
            get_google_api_key()

    def test_raises_when_key_empty_string(self, monkeypatch: Any) -> None:
        monkeypatch.setenv("GOOGLE_API_KEY", "")
        with pytest.raises(RuntimeError, match="GOOGLE_API_KEY"):
            get_google_api_key()


# ---------------------------------------------------------------------------
# build_preprocessor
# ---------------------------------------------------------------------------


class TestBuildPreprocessor:
    def test_returns_column_transformer(self) -> None:
        from sklearn.compose import ColumnTransformer
        ct = build_preprocessor()
        assert isinstance(ct, ColumnTransformer)

    def test_has_three_sub_transformers(self) -> None:
        ct = build_preprocessor()
        names = [name for name, _, _ in ct.transformers]
        assert set(names) == {"num", "ord", "nom"}

    def test_numeric_columns_assigned_correctly(self) -> None:
        ct = build_preprocessor()
        numeric_cols = [cols for name, _, cols in ct.transformers if name == "num"][0]
        assert "OverallQual" in numeric_cols
        assert "TotalSF" in numeric_cols

    def test_ordinal_columns_assigned_correctly(self) -> None:
        ct = build_preprocessor()
        ordinal_cols = [cols for name, _, cols in ct.transformers if name == "ord"][0]
        assert "KitchenQual" in ordinal_cols
        assert "ExterQual" in ordinal_cols

    def test_nominal_columns_assigned_correctly(self) -> None:
        ct = build_preprocessor()
        nominal_cols = [cols for name, _, cols in ct.transformers if name == "nom"][0]
        assert "Neighborhood" in nominal_cols


# ---------------------------------------------------------------------------
# build_pipeline
# ---------------------------------------------------------------------------


class TestBuildPipeline:
    def test_returns_sklearn_pipeline(self) -> None:
        mock_model = MagicMock()
        pipe = build_pipeline(mock_model)
        assert isinstance(pipe, Pipeline)

    def test_pipeline_has_two_steps(self) -> None:
        mock_model = MagicMock()
        pipe = build_pipeline(mock_model)
        assert list(pipe.named_steps.keys()) == ["preprocessor", "model"]

    def test_model_step_is_the_provided_model(self) -> None:
        mock_model = MagicMock()
        pipe = build_pipeline(mock_model)
        assert pipe.named_steps["model"] is mock_model


# ---------------------------------------------------------------------------
# _compute_metrics
# ---------------------------------------------------------------------------


class TestComputeMetrics:
    def test_perfect_predictions_r2_is_one(self) -> None:
        y = pd.Series([100000.0, 200000.0, 300000.0])
        metrics = _compute_metrics(y, np.array([100000.0, 200000.0, 300000.0]))
        assert abs(metrics["r2"] - 1.0) < 1e-9
        assert abs(metrics["rmse"]) < 1e-6
        assert abs(metrics["mae"]) < 1e-6

    def test_returns_required_keys(self) -> None:
        y = pd.Series([100000.0, 200000.0])
        metrics = _compute_metrics(y, np.array([110000.0, 190000.0]))
        assert set(metrics.keys()) == {"rmse", "mae", "r2"}

    def test_rmse_positive(self) -> None:
        y = pd.Series([100000.0, 200000.0])
        metrics = _compute_metrics(y, np.array([110000.0, 190000.0]))
        assert metrics["rmse"] > 0

    def test_bad_predictions_r2_below_one(self) -> None:
        y = pd.Series([100000.0, 200000.0, 300000.0])
        metrics = _compute_metrics(y, np.array([200000.0, 100000.0, 400000.0]))
        assert metrics["r2"] < 1.0


# ---------------------------------------------------------------------------
# compute_training_stats
# ---------------------------------------------------------------------------


class TestComputeTrainingStats:
    def _make_data(self) -> tuple[pd.DataFrame, pd.Series]:
        X = pd.DataFrame({"OverallQual": [5, 7, 8]})
        y = pd.Series([120000.0, 180000.0, 250000.0])
        return X, y

    def test_returns_sale_price_stats_key(self) -> None:
        X, y = self._make_data()
        stats = compute_training_stats(X, y)
        assert "sale_price_stats" in stats

    def test_sale_price_stats_has_required_keys(self) -> None:
        X, y = self._make_data()
        sp = compute_training_stats(X, y)["sale_price_stats"]
        for key in ("mean", "median", "std", "min", "max", "q25", "q75"):
            assert key in sp

    def test_median_correct(self) -> None:
        X, y = self._make_data()
        sp = compute_training_stats(X, y)["sale_price_stats"]
        assert sp["median"] == 180000.0

    def test_extra_keys_merged(self) -> None:
        X, y = self._make_data()
        stats = compute_training_stats(X, y, extra={"model_name": "GBR"})
        assert stats["model_name"] == "GBR"

    def test_train_size_correct(self) -> None:
        X, y = self._make_data()
        stats = compute_training_stats(X, y)
        assert stats["train_size"] == 3

    def test_selected_features_included(self) -> None:
        X, y = self._make_data()
        stats = compute_training_stats(X, y)
        assert "selected_features" in stats


# ---------------------------------------------------------------------------
# save_model / save_training_stats
# ---------------------------------------------------------------------------


def _minimal_pipeline() -> Pipeline:
    """Return a real (picklable) sklearn Pipeline for joblib serialisation tests."""
    from sklearn.preprocessing import StandardScaler
    return Pipeline([("scaler", StandardScaler())])


class TestSaveModel:
    def test_creates_joblib_file(self, tmp_path: Path) -> None:
        import joblib
        path = tmp_path / "model.joblib"
        save_model(_minimal_pipeline(), path=path)
        assert path.exists()
        assert joblib.load(path) is not None

    def test_creates_parent_directory(self, tmp_path: Path) -> None:
        path = tmp_path / "subdir" / "model.joblib"
        save_model(_minimal_pipeline(), path=path)
        assert path.parent.exists()


class TestSaveTrainingStats:
    def test_writes_valid_json(self, tmp_path: Path) -> None:
        path = tmp_path / "stats.json"
        stats = {"sale_price_stats": {"median": 160000}}
        save_training_stats(stats, path=path)
        loaded = json.loads(path.read_text())
        assert loaded["sale_price_stats"]["median"] == 160000

    def test_creates_parent_directory(self, tmp_path: Path) -> None:
        path = tmp_path / "models" / "stats.json"
        save_training_stats({"key": "val"}, path=path)
        assert path.parent.exists()
