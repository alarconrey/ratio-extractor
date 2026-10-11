"""Train TF-IDF candidates, choose on validation, freeze the threshold, then score the test set once."""

import hashlib
import json
import time

import joblib
import pandas as pd

from app.evaluation import choose_threshold
from app.settings import ARTIFACT_DIR, DATASET_FILE, SPLITS_FILE, TFIDF_C_CANDIDATES, THRESHOLD_CANDIDATES
from app.tfidf_model import build_pipeline, predict_probabilities, add_context


def dataset_sha256():
    return hashlib.sha256(DATASET_FILE.read_bytes()).hexdigest()


def load_partitions():
    if not DATASET_FILE.exists():
        raise SystemExit(f"Missing {DATASET_FILE}. See README (Datos).")
    data = pd.read_csv(DATASET_FILE)
    splits = pd.read_csv(SPLITS_FILE)
    data = data.merge(splits[["id_resolucion", "split"]], on="id_resolucion", how="inner")
    data["texto"] = data.texto.fillna("")
    return {name: data[data.split == name].reset_index(drop=True) for name in ["train", "validation", "test"]}


def main():
    parts = load_partitions()
    train, validation, test = parts["train"], parts["validation"], parts["test"]
    print(f"Train {train.id_resolucion.nunique()} res / {len(train)} paragraphs; validation {validation.id_resolucion.nunique()}; test {test.id_resolucion.nunique()}.")
    train_with_context = add_context(train)
    candidates = []
    for use_context in [False, True]:
        for regularization_c in TFIDF_C_CANDIDATES:
            started = time.time()
            pipeline = build_pipeline(regularization_c, use_context).fit(train_with_context, train_with_context.es_ratio)
            probability = predict_probabilities(pipeline, validation)
            threshold, scores = choose_threshold(validation, probability, THRESHOLD_CANDIDATES)
            candidates.append({"use_context": use_context, "c": regularization_c, "threshold": threshold,
                               "validation_resolution_f1": scores[threshold], "pipeline": pipeline, "validation_probability": probability})
            print(f"context={use_context} C={regularization_c}: validation F1 {scores[threshold]:.4f} at threshold {threshold} ({time.time() - started:.0f}s)", flush=True)
    # Best validation F1; ties go to the simpler configuration (no context, smaller C).
    best = max(candidates, key=lambda item: (round(item["validation_resolution_f1"], 4), not item["use_context"], -item["c"]))

    # Frozen from here: the test set is scored once with the chosen pipeline and threshold.
    test_probability = predict_probabilities(best["pipeline"], test)
    digest = dataset_sha256()
    bundle = {
        "pipeline": best["pipeline"], "threshold": best["threshold"], "model_version": f"tfidf-v1-{digest[:8]}",
        "dataset_sha256": digest, "config": {"use_context": best["use_context"], "c": best["c"]},
    }
    ARTIFACT_DIR.mkdir(exist_ok=True)
    joblib.dump(bundle, ARTIFACT_DIR / "model.joblib", compress=3)
    validation[["id_resolucion", "idx"]].assign(probability=best["validation_probability"]).to_csv(ARTIFACT_DIR / "tfidf_predictions_validation.csv", index=False)
    test[["id_resolucion", "idx"]].assign(probability=test_probability).to_csv(ARTIFACT_DIR / "tfidf_predictions_test.csv", index=False)
    selection = [{key: item[key] for key in ["use_context", "c", "threshold", "validation_resolution_f1"]} for item in candidates]
    (ARTIFACT_DIR / "tfidf_selection.json").write_text(json.dumps({"candidates": selection, "selected": bundle["config"], "threshold": best["threshold"],
                                                                   "model_version": bundle["model_version"]}, indent=2), encoding="utf-8")
    print(f"Selected context={best['use_context']} C={best['c']} threshold={best['threshold']}.")

    from app import report  # Imported late so training alone does not depend on the report module.
    report.main()


if __name__ == "__main__":
    main()
