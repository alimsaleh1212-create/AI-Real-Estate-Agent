"""Tests for schemas (src/schemas.py and app/schemas.py).

Covers: ExtractedFeatures validation, confidence computation, properties,
to_feature_dict column ordering, request/response schemas.
"""

import pytest
from pydantic import ValidationError

from app.schemas import (
    InsightRequest,
    InsightResponse,
    PredictionRequest,
    PredictionResponse,
)
from src.schemas import Confidence, ExtractedFeatures

_FEATURE_NAMES = [
    "OverallQual", "TotalSF", "GarageCars", "TotalBath", "YearBuilt",
    "TotalBsmtSF", "KitchenQual", "BsmtQual", "ExterQual", "Neighborhood",
]


class TestExtractedFeaturesDefaults:
    def test_all_none_by_default(self) -> None:
        f = ExtractedFeatures()
        for name in _FEATURE_NAMES:
            assert getattr(f, name) is None

    def test_missing_features_all_ten(self) -> None:
        f = ExtractedFeatures()
        assert len(f.missing_features) == 10
        assert set(f.missing_features) == set(_FEATURE_NAMES)

    def test_extracted_features_empty(self) -> None:
        f = ExtractedFeatures()
        assert f.extracted_features == []

    def test_is_complete_false_when_all_none(self) -> None:
        assert ExtractedFeatures().is_complete is False


class TestExtractedFeaturesConfidence:
    def test_confidence_computed_for_all_fields(self) -> None:
        f = ExtractedFeatures(OverallQual=7)
        assert set(f.confidence.keys()) == set(_FEATURE_NAMES)

    def test_extracted_field_marked_extracted(self) -> None:
        f = ExtractedFeatures(OverallQual=7)
        assert f.confidence["OverallQual"] == Confidence.EXTRACTED

    def test_missing_field_marked_missing(self) -> None:
        f = ExtractedFeatures(OverallQual=7)
        assert f.confidence["TotalSF"] == Confidence.MISSING

    def test_extracted_features_list_correct(self) -> None:
        f = ExtractedFeatures(OverallQual=7, TotalSF=2000.0, KitchenQual="Gd")
        assert set(f.extracted_features) == {"OverallQual", "TotalSF", "KitchenQual"}

    def test_missing_features_list_correct(self) -> None:
        f = ExtractedFeatures(OverallQual=7)
        assert "OverallQual" not in f.missing_features
        assert len(f.missing_features) == 9

    def test_is_complete_true_when_all_filled(self) -> None:
        f = ExtractedFeatures(
            OverallQual=7,
            TotalSF=2000.0,
            GarageCars=2,
            TotalBath=2.5,
            YearBuilt=1995,
            TotalBsmtSF=1000.0,
            KitchenQual="Gd",
            BsmtQual="TA",
            ExterQual="TA",
            Neighborhood="CollgCr",
        )
        assert f.is_complete is True


class TestExtractedFeaturesValidation:
    def test_overall_qual_too_low_raises(self) -> None:
        with pytest.raises(ValidationError):
            ExtractedFeatures(OverallQual=0)

    def test_overall_qual_too_high_raises(self) -> None:
        with pytest.raises(ValidationError):
            ExtractedFeatures(OverallQual=11)

    def test_total_sf_negative_raises(self) -> None:
        with pytest.raises(ValidationError):
            ExtractedFeatures(TotalSF=-1.0)

    def test_year_built_min_raises(self) -> None:
        with pytest.raises(ValidationError):
            ExtractedFeatures(YearBuilt=1799)

    def test_year_built_max_raises(self) -> None:
        with pytest.raises(ValidationError):
            ExtractedFeatures(YearBuilt=2026)

    def test_garage_cars_negative_raises(self) -> None:
        with pytest.raises(ValidationError):
            ExtractedFeatures(GarageCars=-1)


class TestToFeatureDict:
    def test_returns_all_ten_keys(self) -> None:
        d = ExtractedFeatures().to_feature_dict()
        assert list(d.keys()) == _FEATURE_NAMES

    def test_preserves_column_order(self) -> None:
        d = ExtractedFeatures(OverallQual=8, Neighborhood="OldTown").to_feature_dict()
        assert list(d.keys()) == _FEATURE_NAMES

    def test_none_values_pass_through(self) -> None:
        d = ExtractedFeatures(OverallQual=8).to_feature_dict()
        assert d["OverallQual"] == 8
        assert d["TotalSF"] is None

    def test_all_none_when_empty(self) -> None:
        d = ExtractedFeatures().to_feature_dict()
        assert all(v is None for v in d.values())


class TestRequestResponseSchemas:
    def test_prediction_request_valid(self) -> None:
        r = PredictionRequest(query="3 bed house")
        assert r.query == "3 bed house"

    def test_prediction_request_too_short_raises(self) -> None:
        with pytest.raises(ValidationError):
            PredictionRequest(query="ab")

    def test_prediction_response_no_error(self) -> None:
        r = PredictionResponse(
            query="test",
            extracted_features=ExtractedFeatures(),
            predicted_price=200000.0,
            interpretation="A great house.",
        )
        assert r.error is None
        assert r.predicted_price == 200000.0

    def test_prediction_response_error_state(self) -> None:
        r = PredictionResponse(
            query="test",
            extracted_features=ExtractedFeatures(),
            error="Something went wrong.",
        )
        assert r.predicted_price is None
        assert r.interpretation is None
        assert r.error is not None

    def test_insight_request_valid(self) -> None:
        r = InsightRequest(query="What is the median price?")
        assert r.query == "What is the median price?"

    def test_insight_response_with_answer(self) -> None:
        r = InsightResponse(
            query="avg price?", intent="analysis", answer="$179,000."
        )
        assert r.error is None
