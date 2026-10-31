"""End-to-end Singlish demo service: gate -> transcribe -> correct.

Pipeline per request (POST /api/analyze with one 'audio' file):
  Stage 1  Gate        - MH's 39-feature SVM decides Singapore vs non-Singapore.
  Stage 2  Transcribe  - JJ's fine-tuned Whisper (word timestamps) -- only if SG.
  Stage 3  Correct     - your LanceDB Singlish lexicon fixes/expands words.

Run on Cloudera as an Application:
    uvicorn service:app --host 127.0.0.1 --port $CDSW_APP_PORT

Environment:
    SINGLISH_REPO    path to MH's cloned repo (for the mvp package). default /home/cdsw/Singlish
    WHISPER_DIR      path to JJ's model folder.                     default /home/cdsw/models/whisper
    (LEXICON_* vars are read by lexicon_lookup.)
"""
from __future__ import annotations

import os
import sys
import tempfile

import torch
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import JSONResponse

# Make MH's `mvp` package importable before we import from it.
SINGLISH_REPO = os.environ.get("SINGLISH_REPO", "/home/cdsw/Singlish")
if SINGLISH_REPO not in sys.path:
    sys.path.append(SINGLISH_REPO)

from mvp.audio import AudioError, decode_audio  # noqa: E402
from mvp.features import extract_features        # noqa: E402
from mvp.models import AccentModel               # noqa: E402

from lexicon_lookup import Lexicon               # noqa: E402

WHISPER_DIR = os.environ.get("WHISPER_DIR", "/home/cdsw/models/whisper")

app = FastAPI(title="Singlish end-to-end demo")


@app.on_event("startup")
def load_models() -> None:
    from transformers import (WhisperForConditionalGeneration, WhisperProcessor,
                              pipeline)

    app.state.gate = AccentModel()  # Stage 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = WhisperProcessor.from_pretrained(WHISPER_DIR)
    model = WhisperForConditionalGeneration.from_pretrained(WHISPER_DIR)
    model.config.use_cache = True
    model.to(device).eval()
    app.state.asr = pipeline(  # Stage 2
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=30,
        device=0 if device == "cuda" else -1,
    )

    app.state.lex = Lexicon()  # Stage 3


@app.exception_handler(AudioError)
async def _audio_error(request: Request, err: AudioError):
    return JSONResponse({"error": {"code": err.code, "message": err.message}},
                        status_code=getattr(err, "status", 422))


@app.get("/api/health")
def health():
    return {"status": "ready",
            "model_version": app.state.gate.manifest["model_version"]}


def _transcribe_and_correct(raw_bytes: bytes) -> dict:
    # Whisper reads the full clip from a temp file (independent of the SVM's 6s crop).
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        tmp.write(raw_bytes)
        tmp.flush()
        result = app.state.asr(tmp.name, return_timestamps="word")
    words = result.get("chunks", [])
    corrected = app.state.lex.correct(words)
    return {
        "transcript_raw": result.get("text", "").strip(),
        "transcript": " ".join(c["text"] for c in corrected).strip(),
        "words": corrected,
    }


@app.post("/api/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    """Skip the gate: always transcribe + correct. Useful for testing Stage 2+3."""
    data = await audio.read()
    return _transcribe_and_correct(data)
  
    
@app.post("/api/analyze")
async def analyze(audio: UploadFile = File(...)):
    data = await audio.read()

    # Stage 1 - gate
    waveform, meta, warnings = decode_audio(data)
    verdict = app.state.gate.predict(extract_features(waveform))

    response = {"prediction": verdict, "audio": meta, "warnings": warnings,
                "model_version": app.state.gate.manifest["model_version"]}

    # Stages 2 + 3 - only when the gate says Singaporean
    if verdict["class_id"] == 1:
        response.update(_transcribe_and_correct(data))

    return response
