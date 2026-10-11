"""Demo scenarios: real test resolutions rebuilt from the dataset, an out-of-domain text and an invalid input.

Resolutions are chosen from the test set by a fixed rule on the TF-IDF result (one good, one difficult).
They illustrate the interface; they are not an additional evaluation.
"""

import html
import json

import pandas as pd

from app.settings import ARTIFACT_DIR, EXAMPLES_FILE
from app.train import load_partitions

OUT_OF_DOMAIN = [
    "REGLAMENTO INTERNO DE LA BIBLIOTECA COMUNITARIA LOS ALAMOS",
    "Artículo 1. Objeto. El presente reglamento establece las condiciones de préstamo, consulta y uso de los espacios de la biblioteca comunitaria, administrada por la junta vecinal del barrio.",
    "Artículo 2. Horario. La biblioteca atiende de lunes a viernes de nueve a dieciocho horas y los sábados de nueve a trece horas. Durante las vacaciones escolares el horario podrá ampliarse mediante aviso publicado en la puerta de ingreso.",
    "Artículo 3. Préstamo de libros. Cada persona inscrita puede retirar hasta tres libros por un plazo de catorce días. El préstamo puede renovarse una vez, siempre que otra persona no haya reservado el mismo ejemplar.",
    "Artículo 4. Devoluciones. Los libros se devuelven en el mostrador principal. Si un libro se devuelve dañado, la junta vecinal acordará con la persona responsable su reposición o reparación.",
    "Artículo 5. Sala de lectura. La sala de lectura es un espacio de silencio. Se permite el uso de computadoras portátiles y teléfonos en modo silencioso. No se permite consumir alimentos junto a los estantes.",
    "Artículo 6. Actividades. La biblioteca organiza talleres de lectura para niñas y niños los miércoles por la tarde y un club de lectura para personas adultas el último sábado de cada mes.",
    "Artículo 7. Disposición final. Cualquier situación no prevista en este reglamento será resuelta por la directiva de la junta vecinal, que informará su decisión en la siguiente asamblea.",
]


def as_html(paragraphs):
    return "\n".join(f"<p>{html.escape(text)}</p>" for text in paragraphs)


def pick(table, condition, order):
    candidates = table[condition].sort_values(order)
    if candidates.empty:
        raise SystemExit("No resolution matches the example rule.")
    return candidates.iloc[0]


def main():
    test = load_partitions()["test"]
    scores = pd.read_csv(ARTIFACT_DIR / "test_predictions.csv")
    sizes = test.groupby("id_resolucion").agg(n=("idx", "size"), has_foundations=("seccion", lambda s: (s == "fundamentos").any()))
    table = scores.join(sizes, on="id_resolucion")
    moderate = table.n.between(25, 90) & table.has_foundations
    good = pick(table, moderate & (table.f1_tfidf >= 0.9), ["n", "id_resolucion"])
    good_other = pick(table, moderate & (table.f1_tfidf >= 0.9) & (table.tipo_nombre != good.tipo_nombre), ["n", "id_resolucion"])
    hard = pick(table, moderate & table.f1_tfidf.between(0.15, 0.4), ["n", "id_resolucion"])

    examples = []
    for key, title, description, row in [
        ("valid", "Resolución bien extraída", "Caso de prueba (2023–2024) donde TF-IDF recupera casi toda la ratio de la ficha.", good),
        ("valid_other", "Otro tipo de resolución", "Caso de prueba de otro tipo de resolución, también bien extraído.", good_other),
        ("difficult", "Caso difícil", "Caso de prueba donde el modelo difiere bastante de la ficha. Útil para discutir límites.", hard),
    ]:
        rows = test[test.id_resolucion == row.id_resolucion].sort_values("idx")
        examples.append({"id": key, "title": title, "description": description, "resolucion": row.resolucion,
                         "tipo": row.tipo_nombre, "text": as_html(rows.texto.tolist()), "gold_idx": rows.idx[rows.es_ratio == 1].astype(int).tolist()})
    examples.append({"id": "outside", "title": "Texto ajeno", "description": "Un reglamento ficticio, no una resolución. Debe mostrar advertencias.",
                     "resolucion": None, "tipo": None, "text": "\n".join(OUT_OF_DOMAIN), "gold_idx": None})
    examples.append({"id": "invalid", "title": "Entrada inválida", "description": "Texto demasiado corto: la API responde 422.",
                     "resolucion": None, "tipo": None, "text": "AUTO AGROAMBIENTAL PLURINACIONAL S1ª N° 01/2023. Se declara infundado.", "gold_idx": None})
    EXAMPLES_FILE.write_text(json.dumps(examples, ensure_ascii=False, indent=1), encoding="utf-8")
    print("Examples:", ", ".join(f"{item['id']}={item['resolucion']}" for item in examples))


if __name__ == "__main__":
    main()
