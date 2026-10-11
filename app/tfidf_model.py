"""TF-IDF + logistic regression over paragraphs, with position, section and neighbour context."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

from app.settings import SEED
from app.text import SECTIONS

CONTEXT_CHARS = 300


def add_context(frame):
    """Attach text from the previous and next paragraph of the same resolution."""
    frame = frame.sort_values(["id_resolucion", "idx"]).copy()
    grouped = frame.groupby("id_resolucion").texto
    frame["contexto"] = grouped.shift(1).fillna("").str[:CONTEXT_CHARS] + " || " + grouped.shift(-1).fillna("").str[:CONTEXT_CHARS]
    return frame


def numeric_features(frame):
    position = frame["pos_rel"].to_numpy(dtype=float)
    return np.c_[position, position ** 2, np.log1p(frame["n_palabras"].to_numpy(dtype=float)) / 6]


def build_pipeline(regularization_c, use_context):
    columns = [
        ("text", TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=200_000, sublinear_tf=True), "texto"),
        ("section", OneHotEncoder(categories=[SECTIONS], handle_unknown="ignore"), ["seccion"]),
        ("numeric", FunctionTransformer(numeric_features), ["pos_rel", "n_palabras"]),
    ]
    if use_context:
        columns.append(("context", TfidfVectorizer(ngram_range=(1, 1), min_df=3, max_features=50_000, sublinear_tf=True), "contexto"))
    return Pipeline([
        ("features", ColumnTransformer(columns)),
        ("classifier", LogisticRegression(C=regularization_c, class_weight="balanced", solver="liblinear", max_iter=3000, random_state=SEED)),
    ])


def predict_probabilities(pipeline, frame):
    """Probabilities aligned with the order of `frame` (context is computed internally)."""
    ordered = add_context(frame)
    probability = pd.Series(pipeline.predict_proba(ordered)[:, 1], index=ordered.index)
    return probability.reindex(frame.index).to_numpy()
