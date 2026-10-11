import numpy as np
import pandas as pd
import pytest

from app.evaluation import choose_threshold, mcnemar_exact, resolution_scores, wilson_interval
from app.text import detect_heading, paragraph_features, split_paragraphs, words


def test_html_split_keeps_innermost_blocks_only():
    html = "<div><p>Primer párrafo de prueba.</p><li><p>Dentro de lista</p></li><p>  Tercero   con   espacios </p></div>"
    assert split_paragraphs(html) == ["Primer párrafo de prueba.", "Dentro de lista", "Tercero con espacios"]


def test_plain_text_uses_one_paragraph_per_line():
    assert split_paragraphs("Uno largo\n\nDos largo\nab\nTres largo") == ["Uno largo", "Dos largo", "Tres largo"]


@pytest.mark.parametrize("text, expected", [
    ("II.- Fundamentos Jurídicos del Fallo.-", "Fundamentos Jurídicos del Fallo"),
    ("POR TANTO: la Sala resuelve", "POR TANTO"),
    ("1.- Que, conforme al art. 68 de la Ley", None),
    ("Que, el Tribunal advierte lo siguiente", None),
])
def test_heading_detection(text, expected):
    assert detect_heading(text) == expected


def test_features_assign_sections_in_order():
    rows = paragraph_features(["AUTO N° 1/2023", "I. ANTECEDENTES PROCESALES", "Texto de antecedentes.", "II. FUNDAMENTOS JURÍDICOS", "Texto de fundamentos.", "POR TANTO: se declara"])
    assert [row["seccion"] for row in rows] == ["encabezado", "antecedentes", "antecedentes", "fundamentos", "fundamentos", "por_tanto"]
    assert rows[0]["pos_rel"] == 0 and rows[-1]["pos_rel"] == 1


def test_words_ignore_quotes_and_periods():
    assert words("“Art. 68”, Ley N° 1715.") == ["art", "68", "ley", "n", "1715"]


def test_resolution_scores_weight_by_words():
    frame = pd.DataFrame({"id_resolucion": [1, 1, 1, 2, 2], "n_palabras": [10, 30, 60, 50, 50], "es_ratio": [0, 1, 1, 1, 0]})
    scores = resolution_scores(frame, [True, True, False, False, True])
    # Resolution 1: tp=30, predicted=40, gold=90 -> 60/130. Resolution 2: no overlap -> 0.
    assert scores.loc[1] == pytest.approx(60 / 130)
    assert scores.loc[2] == 0


def test_threshold_chosen_on_given_partition_only():
    frame = pd.DataFrame({"id_resolucion": [1, 1, 2, 2], "n_palabras": [10, 10, 10, 10], "es_ratio": [1, 0, 1, 0]})
    threshold, scores = choose_threshold(frame, np.array([0.9, 0.2, 0.7, 0.4]), [0.3, 0.5, 0.8])
    assert threshold == 0.5 and scores[0.5] == 1.0


def test_mcnemar_and_wilson_reference_values():
    result = mcnemar_exact([True, True, True, False, False], [False, False, False, False, True])
    assert (result["only_first"], result["only_second"]) == (3, 1)
    assert result["p_value"] == pytest.approx(0.625)
    low, high = wilson_interval(50, 100)
    assert low == pytest.approx(0.4038, abs=1e-3) and high == pytest.approx(0.5962, abs=1e-3)
