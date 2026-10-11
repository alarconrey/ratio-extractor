"""Build the Colab notebook for BETO fine-tuning (keeps cells reviewable as plain Python)."""

import json
from pathlib import Path

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(keepends=True)})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.strip().splitlines(keepends=True)})


md("""
# Ajuste fino de BETO para extraer la *ratio decidendi*

Clasificación binaria de párrafos: ¿el párrafo forma parte de la *ratio decidendi* de la resolución?

- **Reutilizado:** BETO preentrenado (`dccuchile/bert-base-spanish-wwm-cased`).
- **Entrenado aquí:** la capa de clasificación y el ajuste fino de todos los pesos.
- **Partición:** la misma de `splits.csv` que usan la regla y TF-IDF en el repositorio.
- **Umbral:** se elige en validación. La prueba se evalúa una sola vez.

**Antes de empezar:** *Entorno de ejecución → Cambiar tipo de entorno → GPU* (T4, L4 o A100).
Sube `datos_colab.zip` a tu Google Drive, en la carpeta indicada en `DRIVE_DIR`.
""")

code("""
!pip -q install "transformers==4.56.2" "datasets>=2.19,<4" "accelerate>=0.33"
""")

code("""
# Paths and fixed choices. Change DRIVE_DIR if you stored the data elsewhere.
from google.colab import drive
drive.mount("/content/drive")

DRIVE_DIR = "/content/drive/MyDrive/capstone_ratio"
DATA_ZIP = f"{DRIVE_DIR}/datos_colab.zip"
OUTPUT_DIR = f"{DRIVE_DIR}/beto-ratio"          # checkpoints and final model survive disconnections

MODEL_NAME = "dccuchile/bert-base-spanish-wwm-cased"
SEED = 42
MAX_LENGTH = 384
CONTEXT_CHARS = 300                  # characters taken from previous and next paragraphs
FOUNDATIONS_NEGATIVE_RATE = 0.5      # share of hard negatives (same section) kept for training
OTHER_NEGATIVE_RATE = 0.15           # share of remaining negatives kept for training
EPOCHS = 2
LEARNING_RATE = 2e-5

!mkdir -p /content/data && unzip -o -q "$DATA_ZIP" -d /content/data && ls -la /content/data
""")

code("""
import hashlib, json, random, platform
import numpy as np, pandas as pd, torch

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
batch_size = 32 if any(name in gpu for name in ["A100", "L4", "H100"]) else 16
print(f"GPU: {gpu} | bf16: {use_bf16} | batch: {batch_size}")

data_path = "/content/data/dataset_ratio_parrafos.csv"
data_sha256 = hashlib.sha256(open(data_path, "rb").read()).hexdigest()
paragraphs = pd.read_csv(data_path)
splits = pd.read_csv("/content/data/splits.csv")
paragraphs = paragraphs.merge(splits[["id_resolucion", "split"]], on="id_resolucion")
paragraphs = paragraphs[paragraphs.split != "excluded"].copy()
paragraphs["texto"] = paragraphs.texto.fillna("")
print(paragraphs.groupby("split").agg(resoluciones=("id_resolucion", "nunique"), parrafos=("idx", "size"), positivos=("es_ratio", "sum")))
""")

md("""
## Entrada del modelo

Cada párrafo se acompaña de su sección, su posición y un fragmento de los párrafos vecinos.
Esta misma función se usa en la API (`app/beto_model.py`), para que el modelo vea en producción lo mismo que vio al entrenar.
""")

code("""
def build_pair_inputs(frame):
    \"\"\"Return (segment_a, segment_b) lists. Must match app/beto_model.py exactly.\"\"\"
    frame = frame.sort_values(["id_resolucion", "idx"])
    previous = frame.groupby("id_resolucion").texto.shift(1).fillna("")
    following = frame.groupby("id_resolucion").texto.shift(-1).fillna("")
    position = (frame.pos_rel * 100).round().astype(int).astype(str)
    segment_a = "sección: " + frame.seccion.fillna("otro") + " | posición: " + position + "% | " + frame.texto
    segment_b = "anterior: " + previous.str[:CONTEXT_CHARS] + " || siguiente: " + following.str[:CONTEXT_CHARS]
    return frame, segment_a.tolist(), segment_b.tolist()


train_all = paragraphs[paragraphs.split == "train"]
validation = paragraphs[paragraphs.split == "validation"]
test = paragraphs[paragraphs.split == "test"]

# Context comes from the full resolution, so build inputs before subsampling.
train_all, train_a, train_b = build_pair_inputs(train_all)
train_all = train_all.assign(segment_a=train_a, segment_b=train_b)
rng = np.random.default_rng(SEED)
positive = train_all.es_ratio == 1
foundations = (train_all.seccion == "fundamentos") & ~positive
keep = positive | (foundations & (rng.random(len(train_all)) < FOUNDATIONS_NEGATIVE_RATE)) | (~positive & ~foundations & (rng.random(len(train_all)) < OTHER_NEGATIVE_RATE))
train = train_all[keep]
print(f"Entrenamiento submuestreado: {len(train):,} párrafos, {train.es_ratio.mean():.1%} positivos")
""")

code("""
from datasets import Dataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def to_dataset(frame, segment_a, segment_b):
    dataset = Dataset.from_dict({"segment_a": segment_a, "segment_b": segment_b, "labels": frame.es_ratio.astype(int).tolist()})
    return dataset.map(lambda batch: tokenizer(batch["segment_a"], batch["segment_b"], truncation="longest_first", max_length=MAX_LENGTH),
                       batched=True, remove_columns=["segment_a", "segment_b"])


validation, validation_a, validation_b = build_pair_inputs(validation)
test, test_a, test_b = build_pair_inputs(test)
train_ds = to_dataset(train, train.segment_a.tolist(), train.segment_b.tolist())
validation_ds = to_dataset(validation, validation_a, validation_b)
test_ds = to_dataset(test, test_a, test_b)
lengths = np.array([len(ids) for ids in train_ds["input_ids"]])
print(f"Tokens por ejemplo: mediana {np.median(lengths):.0f}, p95 {np.percentile(lengths, 95):.0f}, truncados al límite {(lengths >= MAX_LENGTH).mean():.1%}")
""")

md("""
## Entrenamiento

Hiperparámetros estándar de ajuste fino de BERT. Se guarda el mejor punto de control según el F1 por párrafo en validación.
""")

code("""
from transformers import AutoModelForSequenceClassification, DataCollatorWithPadding, Trainer, TrainingArguments


def compute_metrics(prediction):
    logits, labels = prediction
    predicted = logits.argmax(axis=1)
    true_positive = ((predicted == 1) & (labels == 1)).sum()
    precision = true_positive / max((predicted == 1).sum(), 1)
    recall = true_positive / max((labels == 1).sum(), 1)
    return {"precision": precision, "recall": recall, "f1": 0.0 if true_positive == 0 else 2 * precision * recall / (precision + recall)}


model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
arguments = TrainingArguments(
    output_dir=OUTPUT_DIR, seed=SEED, learning_rate=LEARNING_RATE, num_train_epochs=EPOCHS,
    per_device_train_batch_size=batch_size, per_device_eval_batch_size=batch_size * 4,
    warmup_ratio=0.1, weight_decay=0.01, bf16=use_bf16, fp16=torch.cuda.is_available() and not use_bf16,
    group_by_length=True, dataloader_num_workers=2,
    eval_strategy="epoch", save_strategy="epoch", save_total_limit=2,
    load_best_model_at_end=True, metric_for_best_model="f1", report_to="none", logging_steps=100,
)
trainer = Trainer(model=model, args=arguments, train_dataset=train_ds, eval_dataset=validation_ds,
                  processing_class=tokenizer, data_collator=DataCollatorWithPadding(tokenizer), compute_metrics=compute_metrics)
train_result = trainer.train()
print(train_result)
""")

md("""
## Umbral en validación y evaluación única en prueba

La métrica principal es la misma del repositorio: F1 de palabras por resolución (macro).
""")

code("""
# group_by_length=True also reorders evaluation batches in transformers 4.56, so predictions
# would not match the dataset order. Predict with a separate Trainer that keeps the order.
predictor = Trainer(
    model=trainer.model,  # best checkpoint already loaded by load_best_model_at_end
    args=TrainingArguments(output_dir="/content/predict", per_device_eval_batch_size=batch_size * 4,
                           bf16=use_bf16, fp16=torch.cuda.is_available() and not use_bf16,
                           group_by_length=False, report_to="none"),
    processing_class=tokenizer, data_collator=DataCollatorWithPadding(tokenizer),
)


def probabilities(dataset):
    output = predictor.predict(dataset)
    # Guard: predictions must come back in dataset order.
    assert (output.label_ids == np.array(dataset["labels"])).all(), "Prediction order does not match the dataset"
    return torch.softmax(torch.tensor(output.predictions, dtype=torch.float32), dim=1)[:, 1].numpy()


def resolution_f1(frame, predicted):
    \"\"\"Word-weighted F1 per resolution, averaged over resolutions.\"\"\"
    frame = frame.assign(predicted=np.asarray(predicted, dtype=bool), gold=frame.es_ratio.astype(bool))
    scores = []
    for _, group in frame.groupby("id_resolucion"):
        words = group.n_palabras
        true_positive = (words * (group.predicted & group.gold)).sum()
        total = (words * group.predicted).sum() + (words * group.gold).sum()
        scores.append(0.0 if true_positive == 0 else 2 * true_positive / total)
    return float(np.mean(scores)), np.array(scores)


validation_probability = probabilities(validation_ds)
candidates = np.round(np.arange(0.10, 0.951, 0.05), 2)
validation_scores = {float(threshold): resolution_f1(validation, validation_probability >= threshold)[0] for threshold in candidates}
threshold = max(validation_scores, key=validation_scores.get)
print(f"Umbral elegido en validación: {threshold} (F1 {validation_scores[threshold]:.3f})")

# Threshold is frozen. The test set is evaluated once.
test_probability = probabilities(test_ds)
test_mean, test_scores = resolution_f1(test, test_probability >= threshold)
print(f"PRUEBA · F1 de palabras por resolución: {test_mean:.3f} | resoluciones con F1 >= 0,8: {(test_scores >= 0.8).mean():.1%}")
""")

code("""
# Breakdowns for the report.
test_eval = test.assign(predicted=test_probability >= threshold)
for column in ["tipo_nombre", "analista"]:
    rows = []
    for value, group in test_eval.groupby(column):
        rows.append({column: value, "resoluciones": group.id_resolucion.nunique(), "f1": resolution_f1(group, group.predicted)[0]})
    print(pd.DataFrame(rows).sort_values("resoluciones", ascending=False).to_string(index=False), "\\n")
""")

md("""
## Guardar el modelo y las predicciones

El repositorio lee `beto_predictions_test.csv` y `beto_summary.json` para la tabla de resultados, y el modelo para la API.
""")

code("""
import transformers

final_dir = f"{OUTPUT_DIR}/final"
trainer.save_model(final_dir)
tokenizer.save_pretrained(final_dir)
for name, frame, probability in [("validation", validation, validation_probability), ("test", test, test_probability)]:
    frame[["id_resolucion", "idx"]].assign(probability=probability).to_csv(f"{OUTPUT_DIR}/beto_predictions_{name}.csv", index=False)
summary = {
    "base_model": MODEL_NAME, "seed": SEED, "max_length": MAX_LENGTH, "context_chars": CONTEXT_CHARS,
    "epochs": EPOCHS, "learning_rate": LEARNING_RATE, "batch_size": batch_size,
    "foundations_negative_rate": FOUNDATIONS_NEGATIVE_RATE, "other_negative_rate": OTHER_NEGATIVE_RATE,
    "train_paragraphs": int(len(train)), "threshold": threshold, "validation_threshold_scores": validation_scores,
    "test_resolution_f1": test_mean, "test_share_f1_at_least_0_8": float((test_scores >= 0.8).mean()),
    "dataset_sha256": data_sha256, "gpu": gpu, "train_runtime_seconds": train_result.metrics.get("train_runtime"),
    "versions": {"python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__},
}
json.dump(summary, open(f"{OUTPUT_DIR}/beto_summary.json", "w"), indent=2, ensure_ascii=False)
print(json.dumps(summary, indent=2, ensure_ascii=False))
""")

md("""
## (Opcional) Publicar el modelo en Hugging Face

Crea un token con permiso de escritura en https://huggingface.co/settings/tokens. Anota la revisión que se imprime: va en `models-lock.json` del repositorio.
""")

code("""
from huggingface_hub import HfApi, login

HF_REPO = "tu-usuario/beto-ratio-decidendi"   # change this
login()
api = HfApi()
api.create_repo(HF_REPO, private=True, exist_ok=True)
commit = api.upload_folder(folder_path=final_dir, repo_id=HF_REPO, commit_message="BETO fine-tuned for ratio decidendi paragraphs")
print("Revisión para models-lock.json:", commit.oid)
""")

notebook = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": []}, "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 0}
Path(__file__).with_name("train_beto_colab.ipynb").write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
print("Notebook written.")
