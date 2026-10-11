# API

Servidor local en `http://127.0.0.1:8995`. Documentación interactiva generada por FastAPI en `/docs`.

## `GET /api/health`

```json
{"status": "ok", "methods": ["rule", "tfidf"], "default_method": "tfidf",
 "beto_status": "models/beto no existe. Ver README: «Activar BETO».",
 "limits": {"min_characters": 1000, "max_characters": 400000}}
```

`status` es `model_missing` si falta `artifacts/model.joblib`. `beto_status` explica por qué BETO no está disponible.

## `POST /api/extract-ratio`

Entrada (no se aceptan otros campos):

```json
{"text": "<p>AUTO AGROAMBIENTAL PLURINACIONAL ...</p><p>...</p>", "method": "auto"}
```

- `text`: resolución completa, en HTML (como en la base) o texto plano con un párrafo por línea. Entre 1.000 y 400.000 caracteres, mayormente letras.
- `method`: `auto` (BETO si está disponible, si no TF-IDF), `beto`, `tfidf` o `rule`.

Respuesta (resumida):

```json
{
  "method": "tfidf", "method_name": "TF-IDF + regresión logística", "model_version": "tfidf-v1-54b1dd9b",
  "threshold": 0.55, "review_margin": 0.15,
  "paragraphs": [{"idx": 20, "text": "...", "section": "fundamentos", "words": 442,
                  "probability": 0.91, "selected": true, "review": false, "rule_selected": true}],
  "ratio_text": "párrafos seleccionados, separados por una línea en blanco",
  "summary": {"paragraphs": 28, "selected": 2, "review": 1, "selected_words": 582, "total_words": 1539},
  "warnings": [], "input_format": "html"
}
```

## Errores

| Código | Causa |
|---|---|
| 422 | Texto ausente, corto, largo, no textual, con campos extra (por ejemplo una etiqueta o la ficha), método desconocido, o menos de 5 / más de 1.500 párrafos. |
| 503 | El método pedido no está disponible (modelo TF-IDF sin entrenar o BETO sin instalar). |

## Otros

- `GET /api/metrics`: contenido de `artifacts/metrics.json`.
- `GET /api/examples`: escenarios de la interfaz (`data/examples.json`).
- `GET /`: interfaz web.
