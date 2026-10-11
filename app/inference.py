"""From a resolution text to a proposed ratio decidendi, paragraph by paragraph."""

import pandas as pd

from app.baseline import has_foundations_section, rule_predict
from app.settings import MAX_PARAGRAPHS, MIN_PARAGRAPHS, REVIEW_MARGIN
from app.text import looks_like_html, paragraph_features, split_paragraphs
from app.tfidf_model import predict_probabilities

METHOD_NAMES = {"rule": "Regla de sección", "tfidf": "TF-IDF + regresión logística", "beto": "BETO con ajuste fino"}


class InputProblem(ValueError):
    """The text passed validation but cannot be processed as a resolution."""


def prepare(text):
    paragraphs = split_paragraphs(text)
    if len(paragraphs) < MIN_PARAGRAPHS:
        raise InputProblem(f"Se detectaron {len(paragraphs)} párrafos; se necesitan al menos {MIN_PARAGRAPHS}. "
                           "Pega la resolución completa: en texto plano, un párrafo por línea.")
    if len(paragraphs) > MAX_PARAGRAPHS:
        raise InputProblem(f"Se detectaron {len(paragraphs)} párrafos; el máximo admitido es {MAX_PARAGRAPHS}.")
    frame = pd.DataFrame(paragraph_features(paragraphs))
    frame.insert(0, "id_resolucion", 0)
    return frame


def extract_ratio(text, method, tfidf_bundle, beto=None):
    frame = prepare(text)
    if method == "beto":
        probabilities, threshold, version = beto.predict_probabilities(frame), beto.threshold, beto.summary.get("base_model", "beto")
    elif method == "tfidf":
        probabilities, threshold, version = predict_probabilities(tfidf_bundle["pipeline"], frame), tfidf_bundle["threshold"], tfidf_bundle["model_version"]
    else:
        probabilities, threshold, version = rule_predict(frame).astype(float), 0.5, "rule-v1"
    selected = probabilities >= threshold
    review = (abs(probabilities - threshold) <= REVIEW_MARGIN) if method != "rule" else selected & False
    rule_selected = rule_predict(frame)

    warnings = []
    if not looks_like_html(text):
        warnings.append("Texto plano: se tomó cada línea como un párrafo. El modelo se entrenó con el HTML de la base; la división puede diferir.")
    if not has_foundations_section(frame):
        warnings.append("No se detectó la sección «Fundamentos jurídicos». Puede no ser una resolución del Tribunal o tener otra estructura; revisa con más cuidado.")
    if not selected.any():
        warnings.append("Ningún párrafo superó el umbral. Revisa los párrafos marcados en amarillo o elige la ratio manualmente.")
    if method == "beto":
        truncated = beto.truncated_paragraphs(frame)
        if truncated:
            warnings.append(f"{truncated} párrafo(s) superan {beto.max_length} tokens; el modelo solo leyó su comienzo.")

    paragraphs = [{
        "idx": int(row.idx), "text": row.texto, "section": row.seccion, "words": int(row.n_palabras),
        "probability": float(probability), "selected": bool(chosen), "review": bool(flag), "rule_selected": bool(rule_choice),
    } for row, probability, chosen, flag, rule_choice in zip(frame.itertuples(), probabilities, selected, review, rule_selected)]
    return {
        "method": method, "method_name": METHOD_NAMES[method], "model_version": version, "threshold": float(threshold),
        "review_margin": REVIEW_MARGIN if method != "rule" else 0.0,
        "paragraphs": paragraphs, "ratio_text": "\n\n".join(item["text"] for item in paragraphs if item["selected"]),
        "summary": {"paragraphs": len(paragraphs), "selected": int(selected.sum()), "review": int(review.sum()),
                    "selected_words": int(frame.n_palabras[selected].sum()), "total_words": int(frame.n_palabras.sum())},
        "warnings": warnings, "input_format": "html" if looks_like_html(text) else "text",
    }
