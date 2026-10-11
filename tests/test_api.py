import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import main
from app.settings import ARTIFACT_DIR, DATASET_FILE, EXAMPLES_FILE

EXAMPLES = {item["id"]: item for item in json.loads(EXAMPLES_FILE.read_text(encoding="utf-8"))}


@pytest.fixture
def client():
    main.app.state.load_beto = False  # Tests must not depend on the optional BETO download.
    with TestClient(main.app) as connection:
        yield connection


class FakeBeto:
    """Declared test double: exercises the BETO route without torch or model weights."""
    threshold, max_length, summary = 0.5, 384, {"base_model": "fake-beto"}

    def predict_probabilities(self, frame):
        return np.where(frame.seccion == "fundamentos", 0.9, 0.1)

    def truncated_paragraphs(self, frame):
        return 0


def post(client, text, **extra):
    return client.post("/api/extract-ratio", json={"text": text, **extra})


def test_health_reports_available_methods(client):
    health = client.get("/api/health").json()
    assert health["status"] == "ok" and health["methods"] == ["rule", "tfidf"] and health["default_method"] == "tfidf"


def test_valid_case_matches_the_evaluated_model(client):
    result = post(client, EXAMPLES["valid"]["text"]).json()
    assert result["method"] == "tfidf" and result["summary"]["selected"] >= 1
    stored = pd.read_csv(ARTIFACT_DIR / "tfidf_predictions_test.csv")
    dataset = pd.read_csv(DATASET_FILE, usecols=["id_resolucion", "resolucion"]).drop_duplicates()
    resolution_id = dataset[dataset.resolucion == EXAMPLES["valid"]["resolucion"]].id_resolucion.iloc[0]
    expected = stored[stored.id_resolucion == resolution_id].sort_values("idx").probability.to_numpy()
    assert np.allclose([item["probability"] for item in result["paragraphs"]], expected)
    assert result["ratio_text"] == "\n\n".join(item["text"] for item in result["paragraphs"] if item["selected"])


def test_difficult_case_flags_paragraphs_for_review(client):
    result = post(client, EXAMPLES["difficult"]["text"]).json()
    assert result["summary"]["review"] >= 1
    assert all(abs(item["probability"] - result["threshold"]) <= result["review_margin"] for item in result["paragraphs"] if item["review"])


def test_outside_text_gets_warnings(client):
    result = post(client, EXAMPLES["outside"]["text"]).json()
    assert any("Fundamentos" in warning for warning in result["warnings"])


@pytest.mark.parametrize("payload", [
    {}, {"text": EXAMPLES["invalid"]["text"]}, {"text": "1" * 2000}, {"text": None}, {"text": 12345},
    {"text": "x" * 400_001}, {"text": EXAMPLES["valid"]["text"], "method": "gpt"},
    {"text": EXAMPLES["valid"]["text"], "es_ratio": [1, 2]}, {"text": EXAMPLES["valid"]["text"], "ficha": "..."},
])
def test_invalid_or_leaking_inputs_are_rejected(client, payload):
    assert client.post("/api/extract-ratio", json=payload).status_code == 422


def test_text_without_paragraph_structure_is_rejected(client):
    response = post(client, "palabra " * 300)
    assert response.status_code == 422 and "párrafos" in response.json()["detail"]


def test_beto_unavailable_returns_503(client):
    response = post(client, EXAMPLES["valid"]["text"], method="beto")
    assert response.status_code == 503 and "BETO" in response.json()["detail"]


def test_beto_route_with_test_double(client):
    main.app.state.beto = FakeBeto()
    try:
        assert client.get("/api/health").json()["default_method"] == "beto"
        result = post(client, EXAMPLES["valid"]["text"]).json()
        assert result["method"] == "beto"
        assert all(item["selected"] == item["rule_selected"] for item in result["paragraphs"])
    finally:
        main.app.state.beto = None


def test_rule_method_always_available(client):
    result = post(client, EXAMPLES["valid"]["text"], method="rule").json()
    assert all(item["selected"] == item["rule_selected"] for item in result["paragraphs"])


def test_missing_model_returns_503(client):
    saved = main.app.state.tfidf
    main.app.state.tfidf = None
    try:
        assert post(client, EXAMPLES["valid"]["text"], method="tfidf").status_code == 503
        assert client.get("/api/health").json()["status"] == "model_missing"
    finally:
        main.app.state.tfidf = saved


def test_frontend_and_examples_are_served(client):
    assert client.get("/").status_code == 200
    assert client.get("/static/app.js").status_code == 200
    assert len(client.get("/api/examples").json()) == 5
    assert "test" in client.get("/api/metrics").json()
