"""Local WAV/FLAC-upload accent classifier with speech-to-text."""
from contextlib import asynccontextmanager
from functools import partial
import logging
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile
from starlette.exceptions import HTTPException
from . import transcription
from .audio import AudioError, MAX_BYTES, decode_audio, prepare_transcription_audio, read_audio
from .features import extract_features
from .models import AccentModels

ROOT = Path(__file__).resolve().parent
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app):
    app.state.models = AccentModels()
    # ASR is optional: without its model the app still classifies and /api/transcribe returns 503.
    app.state.transcriber = None
    app.state.asr_error = None
    try:
        app.state.transcriber = transcription.Transcriber(lock=app.state.models.lock)
    except FileNotFoundError as error:
        logger.warning("Transcription disabled: %s", error)
        app.state.asr_error = "model_not_found"
    except Exception:
        logger.exception("Transcription model failed to load")
        app.state.asr_error = "load_failed"
    # Singlish lexicon (LanceDB) is optional: if it fails to load, transcription
    # still works and simply returns no corrections.
    app.state.lexicon = None
    try:
        from .lexicon_lookup import Lexicon
        app.state.lexicon = Lexicon()
    except Exception:
        logger.exception("Singlish lexicon disabled")
    yield

app = FastAPI(title="Singapore Accent MVP", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")

@app.exception_handler(AudioError)
async def audio_error(request, error):
    return JSONResponse({"error": {"code": error.code, "message": error.message}}, status_code=error.status)

@app.get("/")
def index():
    # Revalidate the entry page so returning browsers discover new asset URLs.
    return FileResponse(ROOT / "static/index.html", headers={"Cache-Control": "no-cache"})

@app.get("/api/health")
def health(request: Request):
    models = request.app.state.models
    transcriber = request.app.state.transcriber
    if transcriber is None:
        asr = {"ready": False, "model_version": None, "decoding": transcription.DECODING,
               "reason": request.app.state.asr_error}
    else:
        asr = {"ready": True, "model_version": transcriber.version, "decoding": transcriber.decoding,
               "device": transcriber.device}
    return {"status": "ready", "model_version": models.ensemble.version, "models": models.versions(),
            "asr": asr}


def classify(models, data):
    waveform, metadata, warnings = decode_audio(data)
    return {**models.predict(extract_features(waveform), waveform),
            "audio": metadata, "warnings": warnings}


def transcribe(transcriber, lexicon, data):
    # The full upload (trimmed, 16 kHz mono), not the classifier's six-second crop.
    waveform, audio = prepare_transcription_audio(*read_audio(data))
    text = transcriber.transcribe(waveform)
    # Correct against the Singlish lexicon; "transcript" is the corrected text,
    # "transcript_raw" is the original Whisper output, "changes" lists edits.
    corrected, changes = text, []
    if lexicon is not None and text:
        try:
            corrected, changes = lexicon.correct_text(text)
        except Exception:
            logger.exception("Lexicon correction failed")
            corrected, changes = text, []
    return {"transcript": corrected, "transcript_raw": text, "changes": changes,
            "model_version": transcriber.version, "decoding": transcriber.decoding,
            "audio": audio, "warnings": [] if text else ["No speech was recognised in this recording."]}


async def handle_audio_upload(request, work, log_message, error_code, error_message):
    # Read the body with a hard limit before multipart parsing. Multipart overhead
    # has its own allowance; the actual file still has a strict 10 MiB limit.
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > MAX_BYTES + 64 * 1024:
            raise AudioError("The upload exceeds the 10 MiB file limit.", "file_too_large", 413)
    request._body = bytes(body)
    try:
        async with request.form(max_files=1, max_fields=0, max_part_size=MAX_BYTES) as form:
            items = form.multi_items()
            if len(items) != 1 or items[0][0] != "audio" or not isinstance(items[0][1], UploadFile):
                raise AudioError("Upload exactly one WAV or FLAC file in the audio field.", "invalid_request")
            upload = items[0][1]
            data = await upload.read(MAX_BYTES + 1)
            return await run_in_threadpool(work, data)
    except AudioError:
        raise
    except HTTPException as error:
        raise AudioError("Upload exactly one WAV or FLAC file as multipart form data.", "invalid_request") from error
    except Exception:
        logger.exception(log_message)
        return JSONResponse({"error": {"code": error_code, "message": error_message}}, status_code=500)


@app.post("/api/classify")
async def classify_upload(request: Request):
    return await handle_audio_upload(
        request, partial(classify, request.app.state.models), "Accent inference failed",
        "inference_failed", "Classification failed. Please try another WAV or FLAC file.")


@app.post("/api/transcribe")
async def transcribe_upload(request: Request):
    transcriber = request.app.state.transcriber
    if transcriber is None:
        return JSONResponse({"error": {"code": "transcription_unavailable",
                                       "message": "Transcription is unavailable on this server. Classification still works."}},
                            status_code=503)
    # Only query the Singlish lexicon (vector DB) when the caller confirms a
    # Singapore-accent detection via ?correct=1. Otherwise pass no lexicon, so
    # the lookup is never run and the raw Whisper transcript is returned as-is.
    apply_lexicon = request.query_params.get("correct") == "1"
    lexicon = request.app.state.lexicon if apply_lexicon else None
    return await handle_audio_upload(
        request, partial(transcribe, transcriber, lexicon), "Transcription failed",
        "transcription_failed", "Transcription failed. Please try another WAV or FLAC file.")
