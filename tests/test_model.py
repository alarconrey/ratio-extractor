import json

import joblib
import numpy as np
import pandas as pd
import pytest

from app.evaluation import resolution_scores
from app.settings import ARTIFACT_DIR, SPLITS_FILE
from app.tfidf_model import predict_probabilities
from app.train import load_partitions


@pytest.fixture(scope="module")
def parts():
    return load_partitions()


@pytest.fixture(scope="module")
def bundle():
    return joblib.load(ARTIFACT_DIR / "model.joblib")


@pytest.fixture(scope="module")
def report():
    return json.loads((ARTIFACT_DIR / "metrics.json").read_text(encoding="utf-8"))


def test_splits_are_disjoint_temporal_and_exclude_weak_alignment():
    splits = pd.read_csv(SPLITS_FILE)
    assert splits.id_resolucion.is_unique
    used = splits[splits.split != "excluded"]
    assert used[used.split == "test"].gestion.min() >= 2023
    assert used[used.split.isin(["train", "validation"])].gestion.max() <= 2022


def test_no_resolution_in_two_partitions(parts):
    ids = {name: set(frame.id_resolucion) for name, frame in parts.items()}
    assert not ids["train"] & ids["validation"] and not ids["train"] & ids["test"] and not ids["validation"] & ids["test"]


def test_model_and_report_come_from_the_same_training(bundle, report):
    assert bundle["model_version"] == report["model_version"]
    assert bundle["threshold"] == report["tfidf_selection"]["threshold"]


def test_threshold_is_the_best_validation_candidate(report):
    selection = report["tfidf_selection"]
    best = max(selection["candidates"], key=lambda row: row["validation_resolution_f1"])
    assert selection["threshold"] == best["threshold"]


def test_saved_pipeline_reproduces_stored_test_predictions(parts, bundle):
    test = parts["test"]
    sample = test[test.id_resolucion.isin(test.id_resolucion.unique()[:5])]
    stored = pd.read_csv(ARTIFACT_DIR / "tfidf_predictions_test.csv")
    expected = sample[["id_resolucion", "idx"]].merge(stored, on=["id_resolucion", "idx"]).probability.to_numpy()
    assert np.allclose(predict_probabilities(bundle["pipeline"], sample), expected)


def test_reported_f1_matches_stored_predictions(parts, bundle, report):
    test = parts["test"]
    stored = test[["id_resolucion", "idx"]].merge(pd.read_csv(ARTIFACT_DIR / "tfidf_predictions_test.csv"), on=["id_resolucion", "idx"])
    scores = resolution_scores(test, stored.probability.to_numpy() >= bundle["threshold"])
    assert scores.mean() == pytest.approx(report["test"]["tfidf"]["resolution_f1"])


def test_tfidf_beats_rule_on_test(report):
    assert report["test"]["tfidf"]["resolution_f1"] > report["test"]["rule"]["resolution_f1"]
