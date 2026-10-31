"""Prepare an allowlisted container context for Azure without research data or secrets."""
import argparse
import json
from pathlib import Path
import shutil

ASR_MODEL_NAME = "whisper-small-mnsc-sg-15k-heldout"
# Serving files only; training_args.bin (a pickle) is not needed to load the model.
ASR_FILES = ("config.json", "generation_config.json", "model.safetensors", "processor_config.json",
             "tokenizer.json", "tokenizer_config.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path, help="New, nonexistent directory")
    parser.add_argument("--asr-model", type=Path, default=None,
                        help=f"Optional saved ASR model folder; copied to mvp/models/{ASR_MODEL_NAME}/. "
                             "Without it the package serves classification only (transcription returns 503).")
    args = parser.parse_args()
    templates = Path(__file__).resolve().parent
    mvp = templates.parents[1]
    manifest = json.loads((mvp / "models/manifest.json").read_text())
    names = ["__init__.py", "app.py", "audio.py", "features.py", "models.py", "embeddings.py",
             "asr_decoding.py", "transcription.py",
             "requirements.txt", "models/manifest.json",
             "static/index.html", "static/app.js", "static/recorder.js", "static/styles.css"]
    for key in ("model_file", "ensemble_file"):
        model_name = Path(manifest[key])
        if model_name.is_absolute() or ".." in model_name.parts or model_name.parts[0] != "models":
            raise ValueError(f"{key} must be a relative path inside models/")
        names.append(str(model_name))
        metadata_name = model_name.with_name(model_name.stem + "_metadata.json")
        if (mvp / metadata_name).is_file():
            names.append(str(metadata_name))
    for name in names:
        if not (mvp / name).is_file():
            raise FileNotFoundError(mvp / name)
    asr_model = args.asr_model.resolve() if args.asr_model else None
    if asr_model is not None:
        for name in ASR_FILES:
            if not (asr_model / name).is_file():
                raise FileNotFoundError(asr_model / name)
    destination = args.destination.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    for name in names:
        target = destination / "mvp" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(mvp / name, target)
    if asr_model is not None:
        target = destination / "mvp/models" / ASR_MODEL_NAME
        target.mkdir(parents=True)
        for name in ASR_FILES:
            shutil.copy2(asr_model / name, target / name)
    manifest.pop("source", None)
    (destination / "mvp/models/manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n")
    for name in ("Dockerfile", "compose.yaml", "Caddyfile"):
        shutil.copy2(templates / name, destination / name)
    (destination / ".dockerignore").write_text(
        "**/__pycache__/\n**/.venv/\n**/.pytest_cache/\n.git/\n")
    asr = "ASR model included" if asr_model is not None else "no ASR model (transcription unavailable)"
    print(f"Prepared {destination}; ML and Ensemble artifacts included; {asr}.")


if __name__ == "__main__":
    main()
