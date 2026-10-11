"""Optional BETO classifier. Loaded only if models/beto and artifacts/beto_summary.json exist.

torch and transformers are imported lazily, so the rest of the system runs without them
(install requirements-beto.txt to enable this model).
"""

import json
import threading

import numpy as np

from app.settings import BETO_DIR, BETO_SUMMARY_FILE


def build_pair_inputs(frame, context_chars):
    """Return (segment_a, segment_b) in frame order. Must match notebooks/train_beto_colab.ipynb."""
    frame = frame.sort_values(["id_resolucion", "idx"])
    previous = frame.groupby("id_resolucion").texto.shift(1).fillna("")
    following = frame.groupby("id_resolucion").texto.shift(-1).fillna("")
    position = (frame.pos_rel * 100).round().astype(int).astype(str)
    segment_a = "sección: " + frame.seccion.fillna("otro") + " | posición: " + position + "% | " + frame.texto
    segment_b = "anterior: " + previous.str[:context_chars] + " || siguiente: " + following.str[:context_chars]
    return frame, segment_a.tolist(), segment_b.tolist()


class BetoModel:
    def __init__(self, model, tokenizer, summary, torch_module):
        self.model, self.tokenizer, self.summary, self.torch = model, tokenizer, summary, torch_module
        self.threshold = float(summary["threshold"])
        self.max_length = int(summary.get("max_length", 384))
        self.context_chars = int(summary.get("context_chars", 300))
        self.lock = threading.Lock()  # One inference at a time on CPU.

    @classmethod
    def try_load(cls):
        """Return (model, None) or (None, reason) without raising."""
        if not (BETO_DIR / "config.json").exists():
            return None, "models/beto no existe. Ver README: «Activar BETO»."
        if not BETO_SUMMARY_FILE.exists():
            return None, "Falta artifacts/beto_summary.json (umbral elegido en validación)."
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError:
            return None, "Faltan torch/transformers. Instala requirements-beto.txt."
        torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
        tokenizer = AutoTokenizer.from_pretrained(BETO_DIR, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(BETO_DIR, local_files_only=True).eval()
        summary = json.loads(BETO_SUMMARY_FILE.read_text(encoding="utf-8"))
        return cls(model, tokenizer, summary, torch), None

    def predict_probabilities(self, frame, batch_size=32):
        ordered, segment_a, segment_b = build_pair_inputs(frame, self.context_chars)
        probabilities = []
        with self.lock, self.torch.no_grad():
            for start in range(0, len(segment_a), batch_size):
                encoded = self.tokenizer(segment_a[start:start + batch_size], segment_b[start:start + batch_size],
                                         truncation="longest_first", max_length=self.max_length, padding=True, return_tensors="pt")
                logits = self.model(**encoded).logits
                probabilities.append(self.torch.softmax(logits.float(), dim=1)[:, 1].numpy())
        result = np.concatenate(probabilities) if probabilities else np.array([])
        # Map back to the caller's row order.
        position = {index: offset for offset, index in enumerate(ordered.index)}
        return np.array([result[position[index]] for index in frame.index])

    def truncated_paragraphs(self, frame):
        """Paragraphs whose own text exceeds the token limit (the model only sees their beginning)."""
        lengths = [len(ids) for ids in self.tokenizer(frame.texto.tolist(), add_special_tokens=True)["input_ids"]]
        return int(sum(length > self.max_length for length in lengths))
