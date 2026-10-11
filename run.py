"""Start the local system; train TF-IDF and build the demo examples first if they are missing."""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="Run the ratio decidendi extractor.")
    parser.add_argument("--rebuild", action="store_true", help="Retrain TF-IDF and rewrite the report before starting.")
    parser.add_argument("--port", type=int, default=8995, help="Local HTTP port (default: 8995).")
    args = parser.parse_args()
    artifacts = [ROOT / "artifacts" / name for name in ["model.joblib", "metrics.json", "tfidf_predictions_test.csv"]]
    if args.rebuild or not all(path.exists() for path in artifacts):
        subprocess.run([sys.executable, "-m", "app.train"], cwd=ROOT, check=True)
    if args.rebuild or not (ROOT / "data" / "examples.json").exists():
        subprocess.run([sys.executable, "-m", "app.examples"], cwd=ROOT, check=True)
    print(f"Open http://127.0.0.1:{args.port}. Press Ctrl+C to stop.", flush=True)
    subprocess.run([sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(args.port)], cwd=ROOT, check=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
    except subprocess.CalledProcessError as error:
        raise SystemExit(f"A step failed with exit code {error.returncode}. See the message above and README.md.") from None
