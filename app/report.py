"""Compare rule, TF-IDF and (if available) BETO on the same test resolutions; write metrics and report.

Run after training TF-IDF, and again after copying BETO predictions from Colab:
    python -m app.report
"""

import json
import platform
from importlib.metadata import version

import pandas as pd

from app.baseline import rule_predict
from app.evaluation import mcnemar_exact, summarize
from app.settings import ARTIFACT_DIR, BETO_PREDICTIONS_FILE, BETO_SUMMARY_FILE, GOOD_EXTRACTION_F1, RESOLUTIONS_FILE, SPLITS_FILE
from app.train import dataset_sha256, load_partitions

METHOD_NAMES = {"rule": "Regla de sección", "tfidf": "TF-IDF + regresión logística", "beto": "BETO con ajuste fino"}


def attach_predictions(test, path, label):
    predictions = pd.read_csv(path)
    merged = test[["id_resolucion", "idx"]].merge(predictions, on=["id_resolucion", "idx"], how="left")
    if merged.probability.isna().any():
        raise SystemExit(f"{label} predictions do not cover every test paragraph ({int(merged.probability.isna().sum())} missing). Use the same dataset and splits.csv.")
    return merged.probability.to_numpy()


def load_beto(test):
    if not (BETO_PREDICTIONS_FILE.exists() and BETO_SUMMARY_FILE.exists()):
        return None
    # Colab hashes the uncompressed CSV, so the hash differs from DATASET_FILE by design.
    # Comparability is enforced instead by requiring a prediction for every test paragraph.
    summary = json.loads(BETO_SUMMARY_FILE.read_text(encoding="utf-8"))
    return summary, attach_predictions(test, BETO_PREDICTIONS_FILE, "BETO")


def describe_partitions(parts):
    return {name: {"resolutions": int(frame.id_resolucion.nunique()), "paragraphs": int(len(frame)), "positive_share": float(frame.es_ratio.mean()),
                   "years": [int(frame.gestion.min()), int(frame.gestion.max())]} for name, frame in parts.items()}


def main():
    parts = load_partitions()
    test = parts["test"]
    selection = json.loads((ARTIFACT_DIR / "tfidf_selection.json").read_text(encoding="utf-8"))
    tfidf_probability = attach_predictions(test, ARTIFACT_DIR / "tfidf_predictions_test.csv", "TF-IDF")
    methods = {"rule": rule_predict(test), "tfidf": tfidf_probability >= selection["threshold"]}
    beto = load_beto(test)
    if beto:
        beto_summary, beto_probability = beto
        methods["beto"] = beto_probability >= beto_summary["threshold"]

    results, per_resolution = {}, {}
    for name, predicted in methods.items():
        results[name], per_resolution[name] = summarize(test, predicted)
    comparisons = {}
    for first, second in [("rule", "tfidf"), ("tfidf", "beto")]:
        if first in per_resolution and second in per_resolution:
            comparisons[f"{first}_vs_{second}"] = mcnemar_exact(per_resolution[first] >= GOOD_EXTRACTION_F1, per_resolution[second] >= GOOD_EXTRACTION_F1)

    splits = pd.read_csv(SPLITS_FILE)
    resolutions = pd.read_csv(RESOLUTIONS_FILE)
    report = {
        "model_version": selection["model_version"], "dataset_sha256": dataset_sha256(),
        "partitions": describe_partitions(parts), "excluded_weak_alignment": int((splits.split == "excluded").sum()),
        "resolutions_total": int(len(resolutions)), "good_extraction_f1": GOOD_EXTRACTION_F1,
        "tfidf_selection": selection, "beto": beto[0] if beto else None,
        "test": {name: results[name] for name in results}, "comparisons": comparisons,
        "method_names": {name: METHOD_NAMES[name] for name in results},
        "versions": {"python": platform.python_version(), **{name: version(name) for name in ["scikit-learn", "pandas", "numpy", "fastapi"]}},
    }
    (ARTIFACT_DIR / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")

    meta = test.drop_duplicates("id_resolucion").set_index("id_resolucion")[["resolucion", "gestion", "tipo_nombre", "analista"]]
    table = meta.join(pd.DataFrame({f"f1_{name}": scores for name, scores in per_resolution.items()}))
    table.reset_index().to_csv(ARTIFACT_DIR / "test_predictions.csv", index=False)
    write_markdown(report)
    for name in results:
        print(f"{METHOD_NAMES[name]}: resolution F1 {results[name]['resolution_f1']:.3f} | well extracted {results[name]['share_good']:.1%}")


def fmt(value):
    return f"{value:.3f}".replace(".", ",")


def thousands(value):
    return f"{value:,}".replace(",", ".")


def pct(value):
    return f"{value:.1%}".replace(".", ",")


def write_markdown(report):
    test, names = report["test"], report["method_names"]
    parts = report["partitions"]
    lines = [
        "# Resultados del experimento", "",
        "Extracción de la *ratio decidendi*: cada párrafo de la resolución se clasifica como «pertenece a la ratio» o no. "
        "Datos reales de la base `arbol` del Tribunal Agroambiental (2011–2024). Archivo generado por `python -m app.report`.", "",
        "## Particiones", "", "| Partición | Resoluciones | Párrafos | % positivos | Años |", "|---|---:|---:|---:|---|",
    ]
    for name, title in [("train", "Entrenamiento"), ("validation", "Validación"), ("test", "Prueba")]:
        part = parts[name]
        lines.append(f"| {title} | {part['resolutions']} | {thousands(part['paragraphs'])} | {pct(part['positive_share'])} | {part['years'][0]}–{part['years'][1]} |")
    lines += ["", f"La prueba contiene solo resoluciones de 2023–2024, posteriores a todo el entrenamiento. Se excluyeron {report['excluded_weak_alignment']} resoluciones cuya ratio se recupera en menos del 50 % (alineamiento débil).", "",
              "## Prueba: misma colección para todos los métodos", "",
              "| Método | F1 por resolución (IC 95 %) | Mediana | Resoluciones con F1 ≥ 0,8 (IC 95 %) | F1 por párrafo |", "|---|---|---:|---|---:|"]
    for name, result in test.items():
        low, high = result["resolution_f1_ci95"]
        good_low, good_high = result["share_good_ci95"]
        lines.append(f"| {names[name]} | **{fmt(result['resolution_f1'])}** ({fmt(low)}–{fmt(high)}) | {fmt(result['resolution_f1_median'])} | {pct(result['share_good'])} ({pct(good_low)}–{pct(good_high)}) | {fmt(result['paragraph']['f1'])} |")
    lines += ["", "F1 por resolución: F1 ponderado por palabras entre los párrafos extraídos y la ratio de la ficha, promediado entre resoluciones. "
              "Intervalo por bootstrap de resoluciones (1.000 remuestreos); proporción con intervalo de Wilson.", ""]
    if report["comparisons"]:
        lines += ["## Comparación pareada (McNemar exacto)", "", "Resultado binario por resolución: F1 ≥ 0,8.", "", "| Comparación | Solo el primero | Solo el segundo | p |", "|---|---:|---:|---:|"]
        for key, value in report["comparisons"].items():
            first, second = key.split("_vs_")
            lines.append(f"| {names[first]} vs {names[second]} | {value['only_first']} | {value['only_second']} | {value['p_value']:.4f} |")
        lines.append("")
    best = list(test)[-1]
    lines += [f"## Desglose del mejor método disponible ({names[best]})", "", "| Tipo de resolución | Resoluciones | F1 |", "|---|---:|---:|"]
    lines += [f"| {row['group']} | {row['resolutions']} | {fmt(row['resolution_f1'])} |" for row in test[best]["by_type"]]
    lines += ["", "| Analista (idusuario) | Resoluciones | F1 |", "|---|---:|---:|"]
    lines += [f"| {row['group']} | {row['resolutions']} | {fmt(row['resolution_f1'])} |" for row in test[best]["by_analyst"]]
    selection = report["tfidf_selection"]
    lines += ["", "## Selección de TF-IDF en validación", "", "| Contexto vecino | C | Umbral | F1 validación |", "|---|---:|---:|---:|"]
    lines += [f"| {'sí' if row['use_context'] else 'no'} | {row['c']} | {row['threshold']} | {fmt(row['validation_resolution_f1'])} |" for row in selection["candidates"]]
    lines += ["", f"Elegido: contexto {'sí' if selection['selected']['use_context'] else 'no'}, C = {selection['selected']['c']}, umbral {selection['threshold']}."]
    if report["beto"]:
        beto = report["beto"]
        lines += ["", "## BETO", "", f"Modelo base `{beto['base_model']}`, ajuste fino de {beto['epochs']} épocas, tasa {beto['learning_rate']}, longitud máxima {beto['max_length']} tokens, "
                  f"{thousands(beto['train_paragraphs'])} párrafos de entrenamiento submuestreados, umbral {beto['threshold']} elegido en validación, GPU {beto['gpu']}."]
    else:
        lines += ["", "## BETO", "", "Aún no hay predicciones de BETO en `artifacts/`. Ver `notebooks/train_beto_colab.ipynb` y README."]
    lines += ["", "## Límites", "",
              "- La etiqueta proviene de la ficha de un analista. Los analistas seleccionan la ratio con criterios distintos; parte del error medido es diferencia de criterio, no error del modelo.",
              "- Las resoluciones de 2011–2021 provienen de una exportación de precedentes vinculados, no de una muestra aleatoria de esos años.",
              "- 2024 aporta pocas resoluciones con ficha; su desempeño se reporta dentro de la prueba pero no por separado.",
              "- Los puntajes no están calibrados como probabilidades de acierto.", "",
              "## Reproducibilidad", "", f"SHA-256 del dataset: `{report['dataset_sha256']}`. Versión del modelo: `{report['model_version']}`. Versiones: `{report['versions']}`."]
    (ARTIFACT_DIR / "evaluation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
