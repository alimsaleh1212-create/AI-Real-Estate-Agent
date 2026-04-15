"""Tests for FastAPI endpoints (app/routers/).

All external dependencies (model loading, LLM calls) are mocked so tests
run without disk access or a Gemini API key.
"""

from typing import Generator
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.schemas import ExtractedFeatures


# ---------------------------------------------------------------------------
# App fixture — patches all startup side-effects
# ---------------------------------------------------------------------------


@pytest.fixture()
def client() -> Generator[TestClient, None, None]:
    """Create a TestClient with model/stats loading and API key mocked out."""
    mock_stats = {
        "sale_price_stats": {
            "median": 160000,
            "mean": 179000,
            "std": 80000,
            "min": 34900,
            "max": 755000,
            "q25": 129400,
            "q75": 210000,
        }
    }
    with (
        patch("app.main.get_google_api_key", return_value="fake-key"),
        patch("app.main.load_model"),
        patch("app.main.load_stats"),
        patch("app.routers.predict.get_stats", return_value=mock_stats),
        patch("app.routers.insights.get_stats", return_value=mock_stats),
    ):
        from app.main import app
        with TestClient(app) as c:
            yield c


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    def test_returns_200(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_returns_ok_status(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# /features
# ---------------------------------------------------------------------------


class TestFeaturesEndpoint:
    def test_returns_200(self, client: TestClient) -> None:
        resp = client.get("/features")
        assert resp.status_code == 200

    def test_contains_expected_keys(self, client: TestClient) -> None:
        resp = client.get("/features")
        data = resp.json()
        assert "OverallQual" in data
        assert "Neighborhood" in data
        assert len(data) == 10


# ---------------------------------------------------------------------------
# POST /predict
# ---------------------------------------------------------------------------


class TestPredictEndpoint:
    def _full_extraction(self) -> ExtractedFeatures:
        return ExtractedFeatures(
            OverallQual=7,
            TotalSF=2000.0,
            GarageCars=2,
            TotalBath=2.5,
            YearBuilt=1998,
            TotalBsmtSF=800.0,
            KitchenQual="Gd",
            BsmtQual="TA",
            ExterQual="TA",
            Neighborhood="CollgCr",
        )

    def test_success_returns_200(self, client: TestClient) -> None:
        with (
            patch("app.routers.predict.extract_features",
                  return_value=self._full_extraction()),
            patch("app.routers.predict.predict_price", return_value=220000.0),
            patch("app.routers.predict.predict_and_interpret",
                  return_value="Nice house."),
        ):
            resp = client.post("/predict", json={"query": "3 bed house with garage"})
        assert resp.status_code == 200

    def test_success_response_fields(self, client: TestClient) -> None:
        with (
            patch("app.routers.predict.extract_features",
                  return_value=self._full_extraction()),
            patch("app.routers.predict.predict_price", return_value=220000.0),
            patch("app.routers.predict.predict_and_interpret",
                  return_value="Nice house."),
        ):
            resp = client.post("/predict", json={"query": "3 bed house with garage"})
        data = resp.json()
        assert data["predicted_price"] == 220000.0
        assert data["interpretation"] == "Nice house."
        assert data["error"] is None

    def test_extraction_failure_returns_error_field(self, client: TestClient) -> None:
        from src.llm_chain import ExtractionError
        err = ExtractionError("bad json")
        with patch("app.routers.predict.extract_features", side_effect=err):
            resp = client.post("/predict", json={"query": "vague description here"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["error"] is not None
        assert data["predicted_price"] is None

    def test_prediction_failure_returns_error_field(self, client: TestClient) -> None:
        with (
            patch("app.routers.predict.extract_features",
                  return_value=self._full_extraction()),
            patch("app.routers.predict.predict_price",
                  side_effect=RuntimeError("model error")),
        ):
            resp = client.post("/predict", json={"query": "3 bed house"})
        data = resp.json()
        assert data["error"] is not None
        assert data["predicted_price"] is None

    def test_interpretation_failure_still_returns_price(
        self, client: TestClient
    ) -> None:
        with (
            patch("app.routers.predict.extract_features",
                  return_value=self._full_extraction()),
            patch("app.routers.predict.predict_price", return_value=180000.0),
            patch(
                "app.routers.predict.predict_and_interpret",
                side_effect=RuntimeError("LLM error"),
            ),
        ):
            resp = client.post("/predict", json={"query": "3 bed house"})
        data = resp.json()
        assert data["predicted_price"] == 180000.0
        assert data["error"] is not None
        assert data["interpretation"] is None

    def test_query_too_short_returns_422(self, client: TestClient) -> None:
        resp = client.post("/predict", json={"query": "ab"})
        assert resp.status_code == 422

    def test_unexpected_exception_returns_error_field(
        self, client: TestClient
    ) -> None:
        with patch("app.routers.predict.extract_features",
                   side_effect=Exception("unexpected")):
            resp = client.post("/predict", json={"query": "3 bed house"})
        data = resp.json()
        assert data["error"] is not None


# ---------------------------------------------------------------------------
# POST /insights
# ---------------------------------------------------------------------------


class TestInsightsEndpoint:
    def test_analysis_intent_returns_answer(self, client: TestClient) -> None:
        with (
            patch("app.routers.insights.classify_intent", return_value="analysis"),
            patch(
                "app.routers.insights.generate_market_insights",
                return_value="The median is $160,000.",
            ),
        ):
            resp = client.post(
                "/insights", json={"query": "What is the median price?"}
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["intent"] == "analysis"
        assert data["answer"] == "The median is $160,000."
        assert data["error"] is None

    def test_prediction_intent_redirects(self, client: TestClient) -> None:
        with patch("app.routers.insights.classify_intent", return_value="prediction"):
            resp = client.post("/insights", json={"query": "3 bed house price?"})
        data = resp.json()
        assert data["intent"] == "prediction"
        assert "/predict" in data["answer"]

    def test_insights_api_failure_returns_error_field(
        self, client: TestClient
    ) -> None:
        with (
            patch("app.routers.insights.classify_intent", return_value="analysis"),
            patch(
                "app.routers.insights.generate_market_insights",
                side_effect=RuntimeError("quota exceeded"),
            ),
        ):
            resp = client.post(
                "/insights", json={"query": "What is the average price?"}
            )
        data = resp.json()
        assert data["error"] is not None
        assert data["answer"] is None
