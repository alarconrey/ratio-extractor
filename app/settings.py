"""Paths and choices fixed before looking at test results."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "artifacts"
MODEL_DIR = ROOT / "models"
FRONTEND_DIR = ROOT / "frontend"

DATASET_FILE = DATA_DIR / "dataset_ratio_parrafos.csv.gz"
SPLITS_FILE = DATA_DIR / "splits.csv"
RESOLUTIONS_FILE = DATA_DIR / "resumen_resoluciones.csv"
EXAMPLES_FILE = DATA_DIR / "examples.json"
BETO_DIR = MODEL_DIR / "beto"
BETO_PREDICTIONS_FILE = ARTIFACT_DIR / "beto_predictions_test.csv"
BETO_SUMMARY_FILE = ARTIFACT_DIR / "beto_summary.json"

SEED = 42
LABEL_OVERLAP = 0.5            # a paragraph is "ratio" if >= 50% of its 8-word sequences appear in the ratio
MIN_RECOVERED_RATIO = 0.5      # resolutions whose ratio is recovered below this are excluded (weak alignment)
GOOD_EXTRACTION_F1 = 0.8       # a resolution counts as "well extracted" at or above this F1
REVIEW_MARGIN = 0.15           # paragraphs within +/- this margin of the threshold are flagged for review
THRESHOLD_CANDIDATES = [round(0.10 + 0.05 * step, 2) for step in range(18)]  # 0.10 ... 0.95
TFIDF_C_CANDIDATES = [1.0, 4.0]
BOOTSTRAP_SAMPLES = 1000

MIN_CHARACTERS = 1000
MAX_CHARACTERS = 400_000
MIN_PARAGRAPHS = 5
MAX_PARAGRAPHS = 1500

TYPE_NAMES = {
    1: "Sentencia Agroambiental Nacional", 2: "Auto Interlocutorio Definitivo", 3: "Auto Nacional Agroambiental",
    4: "Sentencia Agroambiental Plurinacional", 5: "Auto Agroambiental Plurinacional",
    6: "Sentencia Agraria Nacional", 7: "Auto Nacional Agraria",
}
