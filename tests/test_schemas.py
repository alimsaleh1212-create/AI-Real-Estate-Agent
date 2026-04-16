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
    """Out-of-range values are coerced to None (treated as missing) rather than
    raising ValidationError.  This allows the extraction to succeed with the
    valid features while flagging only the implausible ones as unextracted."""

    def test_overall_qual_too_low_becomes_none(self) -> None:
        f = ExtractedFeatures(OverallQual=0)
        assert f.OverallQual is None
        assert "OverallQual" in f.missing_features

    def test_overall_qual_too_high_becomes_none(self) -> None:
        f = ExtractedFeatures(OverallQual=11)
        assert f.OverallQual is None
        assert "OverallQual" in f.missing_features

    def test_total_sf_below_min_becomes_none(self) -> None:
        # TotalSF < 300 sqft is unrealistic for Ames housing
        f = ExtractedFeatures(TotalSF=130.0)
        assert f.TotalSF is None
        assert "TotalSF" in f.missing_features

    def test_total_sf_negative_becomes_none(self) -> None:
        f = ExtractedFeatures(TotalSF=-1.0)
        assert f.TotalSF is None

    def test_year_built_min_becomes_none(self) -> None:
        f = ExtractedFeatures(YearBuilt=1799)
        assert f.YearBuilt is None
        assert "YearBuilt" in f.missing_features

    def test_year_built_max_becomes_none(self) -> None:
        f = ExtractedFeatures(YearBuilt=2026)
        assert f.YearBuilt is None

    def test_garage_cars_negative_becomes_none(self) -> None:
        f = ExtractedFeatures(GarageCars=-1)
        assert f.GarageCars is None

    def test_garage_cars_too_high_becomes_none(self) -> None:
        f = ExtractedFeatures(GarageCars=6)
        assert f.GarageCars is None

    def test_invalid_kitchen_qual_becomes_none(self) -> None:
        f = ExtractedFeatures(KitchenQual="excellent")
        assert f.KitchenQual is None

    def test_invalid_bsmt_qual_becomes_none(self) -> None:
        f = ExtractedFeatures(BsmtQual="Good")
        assert f.BsmtQual is None

    def test_invalid_exter_qual_becomes_none(self) -> None:
        f = ExtractedFeatures(ExterQual="Average")
        assert f.ExterQual is None

    def test_invalid_neighborhood_becomes_none(self) -> None:
        f = ExtractedFeatures(Neighborhood="Downtown")
        assert f.Neighborhood is None

    def test_valid_values_pass_through(self) -> None:
        """A fully valid extraction must not be modified."""
        f = ExtractedFeatures(
            OverallQual=8, TotalSF=2100.0, GarageCars=2, TotalBath=2.5,
            YearBuilt=1998, TotalBsmtSF=900.0, KitchenQual="Ex",
            BsmtQual="Gd", ExterQual="Gd", Neighborhood="NridgHt",
        )
        assert len(f.extracted_features) == 10
        assert f.OverallQual == 8
        assert f.TotalSF == 2100.0
        assert f.Neighborhood == "NridgHt"

    def test_mixed_query_extracts_valid_only(self) -> None:
        """Simulates the original bug: OverallQual=12, TotalSF=130 in one query."""
        f = ExtractedFeatures(
            OverallQual=12, TotalSF=130, GarageCars=2, TotalBath=2.5,
            YearBuilt=1998, TotalBsmtSF=900.0, KitchenQual="Ex",
            BsmtQual="Gd", ExterQual="Gd", Neighborhood="NridgHt",
        )
        assert f.OverallQual is None
        assert f.TotalSF is None
        assert len(f.extracted_features) == 8  # 10 - 2 out-of-range


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
