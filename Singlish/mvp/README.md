# Singapore English Accent Classification

A FastAPI demo that records microphone audio or accepts WAV/FLAC uploads,
compares two test release candidates and shows a transcript of the recording:

| Display name | Saved model | Inputs |
|---|---|---|
| **ML** | `svm_65_c_261024.joblib` | 65 selected acoustic features |
| **Ensemble** | `fusion_svm_c_261024.joblib` | The same 65 features + 512 Whisper values |
| **Transcript** | `whisper-small-mnsc-sg-15k-heldout` (fine-tuned Whisper small) | The full upload, see [Transcription](#transcription) |

The page shows the shared prediction when both models agree, or **Inconclusive**
with a retry prompt when they disagree. Both cards show the predicted accent and its
calibrated probability, with **Agree** or **Disagree** based on their class labels.
Both follow the notebooks' rule: Singapore when `p_singapore >= 0.50`, including
ties. Agreement is not a confidence measure.

Sources: [E1 calibration notebook](../results/E1_Feature_Engineering_20261024/E1_svm_calibrated_probability.ipynb)
and [E4 calibration notebook](../results/E4_Hybrid_Ensemble_20261024/E4_03_whisper_svm_calibrated_probability.ipynb).
Both artifacts use the 40,000-clip study. The date suffix distinguishes them from
the previous exports; matching metadata uses the same dated stem.
The local MVP uses the **24 Oct models** with loudness normalization. The
[hosted Azure demo](https://singlish-demo-4e04e8d0.eastasia.cloudapp.azure.com/)
uses the **25 Oct models** without loudness normalization.

## Setup

Python 3.12 or later is required; Python 3.14 matches the serving environment.
Use trusted joblib artifacts only, since loading them executes Python code.
Copy both models and their matching metadata from the completed
`E1_Feature_EngineeringV4/outputs/` studies below, using your local project path.
From the repository root:

```bash
study_root=/path/to/E1_Feature_EngineeringV4
ml_source="$study_root/outputs/e1_feature_study_accent_40000/svm_calibrated_probability_testing"
ensemble_source="$study_root/outputs/svm65_fusion_comparison_40000/calibrated_probability_testing"
cp "$ml_source/svm_65_calibrated.joblib" mvp/models/svm_65_c_261024.joblib
cp "$ml_source/svm_65_calibrated_metadata.json" mvp/models/svm_65_c_261024_metadata.json
cp "$ensemble_source/fusion_svm_calibrated.joblib" mvp/models/fusion_svm_c_261024.joblib
cp "$ensemble_source/fusion_svm_calibrated_metadata.json" mvp/models/fusion_svm_c_261024_metadata.json
python3 -m venv mvp/.venv
# Linux CPU serving: install this first to avoid unnecessary CUDA dependencies.
mvp/.venv/bin/python -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
mvp/.venv/bin/python -m pip install -r mvp/requirements.txt
./mvp/run.sh
```

On Windows (PowerShell), run from the folder that contains `mvp`:

```powershell
mvp\.venv\Scripts\python -m uvicorn mvp.app:app --host 127.0.0.1 --port 8000
```

Open **http://127.0.0.1:8000**. Set `PORT=8001` to use another port.
The first startup downloads `openai/whisper-base.en`, pinned to revision
`911407f4214e0e1d82085af863093ec0b66f9cd6`. Subsequent starts can use
`HF_HUB_OFFLINE=1` once the weights are cached. Both classifiers and Whisper must
load before the API is ready. Allow at least 4 GB host RAM; see
[Azure setup](deploy/azure/README.md). With the ASR model loaded, the measured
Windows CPU peak working set was about 1.8 GB (0.65 GB without it); see
[Transcription](#transcription).

### ASR model

Transcription uses the fine-tuned `whisper-small-mnsc-sg-15k-heldout` model
(about 1 GB, not in the GitHub release and never committed). The app looks for it
in this order:

1. `ASR_MODEL_PATH`, if set. Relative paths resolve from `mvp/`. The folder must
   contain `config.json`, otherwise transcription is disabled.
2. `mvp/models/whisper-small-mnsc-sg-15k-heldout/` (git-ignored).
3. `../whisper_fine-tuned_transcription/models/whisper-small-mnsc-sg-15k-heldout/`,
   the sibling ASR project, for local development.

If no model is found or it fails to load, the app still starts and classifies.
`/api/health` reports `asr.ready: false` and `/api/transcribe` returns 503.

To bundle the model (for example for deployment), copy only its serving files
into `mvp/models/whisper-small-mnsc-sg-15k-heldout/`: `config.json`,
`generation_config.json`, `model.safetensors`, `processor_config.json`,
`tokenizer.json` and `tokenizer_config.json` (`training_args.bin` is not needed).
The Azure packager does this with `--asr-model`; see
[Azure setup](deploy/azure/README.md). Never edit the model's files, including
`generation_config.json`: the decoding options are passed at request time.

## Using the demo

- **Record:** allow microphone access and read this sentence:
  **“They tried to keep it a secret but he was able to figure out everything”**.
  Stop and preview your recording. Recording stops automatically at 10 seconds;
  **Cancel** preserves previously selected audio.
- **Upload:** choose or drop a PCM WAV or lossless FLAC file.
- Listen to the preview, then select **Classify recording** to run both models.
  After the accent result appears, the page requests the transcript and shows it
  in the **Transcript** box. A transcription error is shown there and does not
  affect the accent result.

Use clear English speech from one speaker. Files must be at most 10 MiB and
10 seconds. Clips shorter than three seconds **after silence trimming** are
rejected with a request to retry with a longer recording, before either model
runs. Longer clips are classified on the first six seconds after trimming, with
a notice; the transcript covers the whole clip.

Microphone access requires HTTPS or localhost and a supported browser.
Stopping, cancelling or leaving the page releases the microphone. Audio is sent
only when you classify (one request for classification, then one for the
transcript), processed per request, and not retained.

## Model behavior

Shared preprocessing matches E1: mono → 16 kHz (soxr HQ) → 40 dB frame-RMS
boundary trimming (25 ms frames, 10 ms hop, 100 ms edge padding) → require three
seconds and crop to six → reject RMS below 1e-5 → normalize to −20 dBFS RMS
with a 0.99 peak limit → PCM16 FLAC encode/decode in memory, matching the
24 Oct training audio. Internal pauses remain.

The app extracts 91 acoustic features once and selects each model's 65 metadata
columns in the saved order. Ensemble concatenates the raw selected values with a
512-value frozen Whisper encoder embedding. Whisper uses the same waveform,
standard 30-second padding and a mean over real 20 ms encoder steps only
(`ceil(samples / 320)`). The saved fusion pipeline handles scaling, Whisper PCA
to 32 components, branch weighting, SVM classification and sigmoid calibration.
No fitting occurs in the app. CPU float32 embeddings can differ slightly from
the research GPU embeddings. Inference is serialized to bound memory use.

The classifiers assess accent characteristics, not nationality or identity.
They can be wrong and do not verify English speech or reject all noise/music.

## Transcription

- **Model:** `whisper-small-mnsc-sg-15k-heldout`, Whisper small fine-tuned on
  IMDA MNSC PART3 (Singapore conversational English), the final model of the
  sibling `whisper_fine-tuned_transcription` project.
- **Decoding:** English `transcribe` task with temperature fallback, as in that
  project's evaluation and `transcribe.py`. Each clip is decoded greedily first and
  re-decoded at temperatures 0.2, 0.4, 0.6, 0.8 and 1.0 only when the output fails
  the checks: compression ratio above 1.35 (zlib ratio of the token IDs) or average
  log-probability below −1.0. The seed is reset to 42 for every request, so the same
  upload gives the same transcript. Audio is split into ≤ 30 s windows (uploads are
  at most 10 s, so normally one window). Why: on held-out PART3 speakers the word
  error rate is 0.232 with fallback vs 0.336 with plain greedy decoding. The
  checks catch repetition loops and re-decode those clips.
- **Audio:** the transcriber gets the full upload (mono, 16 kHz, silent edges
  trimmed with the same 40 dB rule), not the classifier's six-second crop, and no
  loudness normalization. Clips under three seconds after trimming get the same
  `invalid_duration` error as classification. Transcription preserves waveform
  gain and uses the full clip; classification normalizes and crops to six seconds.
- **Serialization:** transcription and classification share one lock, so only one
  model runs at a time.
- **Measured CPU cost** (Windows, Intel i9-11900H laptop CPU, float32, two torch
  threads as set by the app, 9.5 s real PART3 clips): about 3.3–5.3 s per transcription
  (real-time factor 0.35–0.56), on top of about 0.7 s for classification. The ASR
  model adds about 1.1 GB to the peak working set (1.8 GB vs 0.65 GB) and about
  1.2 GB of private memory after requests (1.6 GB at peak). Startup took 8–26 s with the ASR model vs 10 s without,
  depending on the disk cache. GPU was not measured; the serving image is CPU-only.
- **transformers 4.57.6:** the model was saved with transformers 5, whose
  `tokenizer_config.json` 4.57.6 cannot read directly. `transcription.py` builds the
  tokenizer in memory instead and turns off `clean_up_tokenization_spaces`, which
  4.57.6 would otherwise apply ("oh!wow!" instead of "oh !wow!"). With both fixes the
  generated token IDs and texts match transformers 5.16.1 on the checked clips, so
  the pinned version is kept. The model files are not modified.
- **Vendored decoding:** `asr_decoding.py` is a copy of the needed parts of the ASR
  project's `scripts/whisper_decoding.py` and `scripts/transcribe.py`, because the
  deployed app does not have the sibling folder. Keep it in sync:
  `tests/test_asr_decoding_sync.py` compares constants, `generate_kwargs()` and the
  function source when the sibling project exists, and skips otherwise.

## Model swaps

`MODEL_PATH` / `MODEL_VERSION` configure ML; `ENSEMBLE_MODEL_PATH` /
`ENSEMBLE_MODEL_VERSION` configure Ensemble. Relative paths resolve from `mvp/`.
Versions default to filename stems. Restart after replacing files; no checksum
update is required. Keep each model and its matching metadata together, and
match audio preprocessing to its training release.

ML accepts an estimator or a study dictionary containing `model`, `columns` and
`feature_names`. Selected-feature estimators need `<stem>_metadata.json` with
`columns` and `feature_names`; full 91-feature estimators can omit metadata.
Ensemble uses the E4 calibrated metadata schema: `acoustic_columns_in_91`,
`acoustic_feature_names`, `whisper_width`, `positive_label` and
`whisper_provenance`. Its input width is 577. Both calibrated exports contain
one refitted SVM. Uncalibrated ML overrides use the original SVM decision and
return `probability: null`, shown as a dash in the UI.

For deployment, update `model_file` / `ensemble_file` in the manifest and rebuild.
The deployment packager includes both artifacts and their metadata.

`ASR_MODEL_PATH` / `ASR_MODEL_VERSION` configure transcription (see
[ASR model](#asr-model)). The version defaults to the model folder name.

## API

- `GET /` — recording/upload page.
- `GET /api/health` — readiness, both model versions and the `asr` status.
- `POST /api/classify` — multipart form with exactly one `audio` WAV/FLAC file.
- `POST /api/transcribe` — the same multipart form; returns the transcript.

```bash
curl -F 'audio=@recording.wav' http://127.0.0.1:8000/api/classify
curl -F 'audio=@recording.wav' http://127.0.0.1:8000/api/transcribe
```

Success returns `prediction` (the Ensemble result), `final_model: "ensemble"`,
`models.ml`, `models.ensemble`, `agreement`, `model_version`, `audio` and `warnings`.
Each prediction contains `class_id`, `label`, `display_label`, `probability` (of
the predicted class) and the original `decision_score`. Labels follow the
calibrated 0.50 threshold, so they can differ from the raw score's sign. The UI
shows probabilities.

Audio metadata contains `duration_seconds`, `classified_duration_seconds` and
`truncated`; processed duration is not verified speech duration. Errors return
`error.code` and `error.message`: 413 for size limits, 415 for unsupported
formats, 422 for invalid audio/requests (including `invalid_duration`) and 500
for inference failure. There is no silent fallback to ML if Ensemble fails.

`/api/transcribe` is separate from classification, so a transcription failure
cannot change a classification result. It applies the same upload validation and
error format, and returns:

```json
{"transcript": "where is your true home? my true home will be my current home.",
 "model_version": "whisper-small-mnsc-sg-15k-heldout",
 "decoding": "temperature fallback",
 "audio": {"duration_seconds": 9.5, "transcribed_duration_seconds": 9.43},
 "warnings": []}
```

`transcribed_duration_seconds` is the length after edge trimming. An empty
`transcript` comes with the warning "No speech was recognised in this recording."
Errors: 413, 415 and 422 as above, 500 `transcription_failed`, and 503
`transcription_unavailable` when the ASR model is not loaded.

`/api/health` keeps `status: "ready"` while classification works and adds `asr`:
`{"ready": true, "model_version": "...", "decoding": "temperature fallback", "device": "cpu"}`,
or `{"ready": false, "model_version": null, "decoding": "temperature fallback", "reason": "model_not_found"}`
(`reason` is `model_not_found` or `load_failed`).

## Verification

```bash
mvp/.venv/bin/python -m pip install -r mvp/requirements-dev.txt
mvp/.venv/bin/python -m pytest mvp/tests -q
node --test mvp/tests/*.cjs
```

All automated tests live in `mvp/tests/`:

| File | Checks |
|---|---|
| `test_api.py` | Both upload endpoints, validation, classifier parity, health and errors |
| `test_audio.py` | Trimming, duration, normalized PCM16 classification and original-gain ASR audio |
| `test_models.py` | Saved model contracts, feature order, calibrated thresholds and Whisper pooling |
| `test_transcription.py` | ASR model paths, decoding, locking and tokenizer compatibility |
| `test_asr_decoding_sync.py` | Optional comparison with the sibling ASR project's source |
| `test_ui.cjs`, `test_recorder.cjs` | Results, transcripts, recording and microphone cleanup |

`conftest.py` shares the audio factory and API stubs. API tests run the real
saved classifiers with stub Whisper/ASR models, so they need no downloads.
The source comparison skips when the sibling ASR project is absent; the real
tokenizer check skips when its model files are absent.

For an end-to-end check, upload a real speech clip using the actual Whisper
encoder. The offline E1 verifier also checks acoustic feature parity:

```bash
mvp/.venv/bin/python -m mvp.scripts.verify_study /path/to/E1_Feature_EngineeringV4 --output mvp/verification.json
```

It requires `pyarrow==26.0.0`, which is not a server dependency. If study audio
is stored elsewhere, pass `--data-root /path/to/shared/data` (locally, V4 uses
`/home/mh/nusproj/E1_Feature_EngineeringV3/data`).
