# Singlish Accent Classification, Transcription & Lexicon Correction

A speech system for Singaporean English. An audio clip is passed through a
three-stage **cascade**:

1. **Accent gate** — a supervised classifier (SVM / logistic regression / random
   forest, plus a hybrid feature-fusion model) decides whether the speaker is
   using Singaporean English.
2. **Transcription** — if a Singaporean-English accent is detected, a fine-tuned
   Whisper-small model (`whisper-small-mnsc-sg-15k-heldout`) transcribes the clip.
3. **Singlish lexicon correction** — the transcript is corrected and expanded
   against a LanceDB vector database of Singlish terms, particles, and acronyms,
   using phonetic (double-metaphone) and semantic (embedding) matching.

The web UI shows the classification result and a **before/after** view of the
transcript so the lexicon's effect is visible.

## Repository layout

```
Singlish/                 # the application
  mvp/                    # FastAPI app, models loader, transcription, features
    app.py                # endpoints: /api/classify, /api/transcribe, /api/health
    static/               # vanilla HTML/CSS/JS front end
    models/               # model artifacts (NOT in git — see "Models" below)
  build_lexicon.py        # builds the LanceDB lexicon from the JSON
  lexicon_lookup.py       # phonetic + semantic lookup / correction
  singlish_lexicon.json   # the Singlish lexicon source (tracked)
  run_app.py              # launcher for the hosted (Kubernetes) application
```

> **Models and caches are intentionally excluded from git** (see `.gitignore`).
> They are too large for a normal repository and are distributed separately.

## Prerequisites

- Python 3.10+
- ~3 GB free disk and ~2–3 GB free RAM (CPU is fine; no GPU required)
- Internet access on first run (to download the stock `openai/whisper-base.en`
  encoder used by the fusion model)

## Setup

### 1. Create a virtual environment and install dependencies

```bash
cd Singlish
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Download the models

The model artifacts are published as release assets. Download the three archives
and unzip them **from the `Singlish/` folder** so they land in `mvp/models/`:

```bash
# from the Singlish/ directory
REL=https://github.com/chupchai/singlish-gc2-group29/releases/download/models-20261029
curl -LO $REL/ml.zip            # svm_65_c_261024.joblib + metadata
curl -LO $REL/ensemble.zip      # fusion_svm_c_261029.joblib + metadata
curl -LO $REL/transcription.zip # whisper-small-mnsc-sg-15k-heldout/
unzip -o ml.zip && unzip -o ensemble.zip && unzip -o transcription.zip
rm ml.zip ensemble.zip transcription.zip
```

Verify (each line should list the file):

```bash
ls mvp/models/svm_65_c_261024.joblib
ls mvp/models/fusion_svm_c_261029.joblib
ls mvp/models/whisper-small-mnsc-sg-15k-heldout/config.json
```

### 3. Build the Singlish lexicon database

```bash
# from the Singlish/ directory
python build_lexicon.py singlish_lexicon.json
```

This creates the LanceDB table used by the correction stage (default location
`lexicon_db/`, overridable with the `LEXICON_DB` environment variable).

## Running the app locally

From the `Singlish/` directory, run the API with uvicorn and open the UI:

```bash
uvicorn mvp.app:app --host 127.0.0.1 --port 8000
```

Then visit <http://127.0.0.1:8000>. Microphone recording works on `localhost`
(a secure context); classification, transcription, and the before/after
lexicon correction are all available from the page.

## Configuration (environment variables)

| Variable | Purpose | Default |
|---|---|---|
| `ASR_MODEL_PATH` | Path to the Whisper model folder (must contain `config.json`) | auto-detected at `mvp/models/whisper-small-mnsc-sg-15k-heldout/` |
| `LEXICON_DB` | Path to the LanceDB lexicon database | `lexicon_db/` (local) |
| `LEXICON_TABLE` | LanceDB table name | `singlish` |
| `HF_HUB_OFFLINE` | Set to `1` to run fully offline after the first run | unset |

## Notes

- The first startup downloads `openai/whisper-base.en` (for the fusion model's
  512-value encoder embedding). After that you can set `HF_HUB_OFFLINE=1`.
- Do **not** commit models, audio, datasets, caches, or credentials — the
  `.gitignore` is configured to keep them out.

## Training & evaluation

Model development evidence (training guides, notebooks, hyperparameters, and
evaluation results) is kept alongside the application for reference:

- **Classifier training guides & notebooks:** [`Singlish/training/`](Singlish/training)
- **Fine-tuned Whisper evaluation & docs:** [`Singlish/whisper_fine-tuned_transcription/`](Singlish/whisper_fine-tuned_transcription)

These folders are documentation only and are not required to run the app.

## Upstream / team repository

The accent classifier, fusion ensemble, fine-tuned Whisper model, and the base
web application were developed by the team in the upstream repository, whose
full commit history reflects the overall effort spent on the project:

- **Team repository:** <https://github.com/nuswohc/Singlish>

This repository builds on that work and adds the Singlish lexicon vector-database
correction stage (LanceDB) and the accent-gated before/after transcript display.
