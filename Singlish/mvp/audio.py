"""Bounded WAV/FLAC decoding; no recordings are written to disk."""
from io import BytesIO
import soxr
import numpy as np
import soundfile as sf
from .features import SAMPLE_RATE, MIN_SECONDS, MAX_SECONDS, trim_silence, normalize_loudness

MAX_BYTES = 10 * 1024 * 1024
MAX_DURATION = 10
SHORT_CLIP_MESSAGE = "The clip is less than 3 seconds after silence trimming. Please retry with a longer recording."

class AudioError(Exception):
    def __init__(self, message, code="invalid_audio", status=422):
        self.message, self.code, self.status = message, code, status
        super().__init__(message)


def decode_audio(data):
    """Classifier waveform: validated, trimmed, cropped, normalized PCM16 audio."""
    return prepare_classifier_audio(*read_audio(data))


def read_audio(data):
    """Validated mono float32 audio at 16 kHz, plus the soundfile info."""
    if not data:
        raise AudioError("Choose a nonempty WAV or FLAC file.")
    if len(data) > MAX_BYTES:
        raise AudioError("The file exceeds the 10 MiB limit.", "file_too_large", 413)
    is_wav = data[:4] == b"RIFF" and data[8:12] == b"WAVE"
    if not is_wav and data[:4] != b"fLaC":
        raise AudioError("Upload a PCM WAV or FLAC file.", "unsupported_format", 415)
    try:
        info = sf.info(BytesIO(data))
        if info.format not in {"WAV", "FLAC"} or info.subtype not in {"PCM_U8", "PCM_S8", "PCM_16", "PCM_24", "PCM_32"}:
            raise AudioError("Use an uncompressed PCM WAV or lossless FLAC file.", "unsupported_encoding", 415)
        if info.frames <= 0 or info.samplerate <= 0:
            raise AudioError("The file contains no audio.")
        if info.duration > MAX_DURATION:
            raise AudioError("Use a WAV or FLAC recording of 10 seconds or less.", "audio_too_long")
        # Bound decoder allocations independently of the claimed duration.
        if info.channels > 8 or info.samplerate > 192_000 or info.frames * info.channels > 4_000_000:
            raise AudioError("Use a WAV or FLAC file with at most 8 channels and a sample rate up to 192 kHz.")
        y, rate = sf.read(BytesIO(data), dtype="float32", always_2d=False)
        if y.ndim > 1:
            y = y.mean(axis=1)
        if rate != SAMPLE_RATE:
            y = soxr.resample(y, rate, SAMPLE_RATE, quality="HQ").astype(np.float32)
    except AudioError:
        raise
    except (ValueError, RuntimeError, OSError, EOFError) as error:
        raise AudioError("The audio could not be decoded. Export it again as PCM WAV or FLAC.") from error
    if not y.size or not np.isfinite(y).all() or not np.any(y):
        raise AudioError("The recording is empty or silent. Upload clear English speech.")
    return y, info


def prepare_classifier_audio(y, info):
    y = trim_silence(y)
    duration = len(y) / SAMPLE_RATE
    if duration < MIN_SECONDS:
        raise AudioError(SHORT_CLIP_MESSAGE, "invalid_duration")
    truncated = duration > MAX_SECONDS
    warnings = []
    if truncated:
        y = y[:SAMPLE_RATE * MAX_SECONDS]
        warnings.append("Only the first six seconds after silence trimming were classified.")
    try:
        y = normalize_loudness(y)
    except ValueError as error:
        raise AudioError(str(error)) from error
    # Training features used decoded PCM16 FLAC. Reproduce that quantization
    # in memory using the same FLAC encoder (no file is retained).
    stream = BytesIO()
    sf.write(stream, y, SAMPLE_RATE, format="FLAC", subtype="PCM_16")
    stream.seek(0)
    y, _ = sf.read(stream, dtype="float32")
    return y, {
        "duration_seconds": round(info.duration, 4),
        "classified_duration_seconds": round(min(duration, MAX_SECONDS), 4),
        "truncated": truncated,
    }, warnings


def prepare_transcription_audio(y, info):
    """Full upload for ASR: edges trimmed, no 6 s crop and no loudness normalization
    (the ASR model was trained and evaluated on un-normalized audio)."""
    y = trim_silence(y)
    duration = len(y) / SAMPLE_RATE
    if duration < MIN_SECONDS:
        raise AudioError(SHORT_CLIP_MESSAGE, "invalid_duration")
    return y.astype(np.float32), {
        "duration_seconds": round(info.duration, 4),
        "transcribed_duration_seconds": round(duration, 4),
    }
