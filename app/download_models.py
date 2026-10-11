"""Download the fine-tuned BETO pinned in models-lock.json into models/beto (optional).

Usage:
    python -m app.download_models

If the Hugging Face repository is private, run `huggingface-cli login` first.
Alternatively, copy the `final` folder produced by the Colab notebook to models/beto.
"""

import json

from app.settings import BETO_DIR, ROOT


def main():
    from huggingface_hub import snapshot_download  # Installed with requirements-beto.txt.

    lock = json.loads((ROOT / "models-lock.json").read_text(encoding="utf-8"))["beto"]
    if not lock.get("repo") or "TU-USUARIO" in lock["repo"]:
        raise SystemExit("models-lock.json has no published repository yet. Copy the Colab 'final' folder to models/beto instead.")
    print(f"Downloading {lock['repo']} at {lock['revision']}...", flush=True)
    snapshot_download(lock["repo"], revision=lock["revision"], local_dir=BETO_DIR,
                      allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"])
    print("BETO ready in models/beto. Restart the server.")


if __name__ == "__main__":
    main()
