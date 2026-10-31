"""Shared audio factory and API stubs; saved classifiers still run for real."""
from io import BytesIO

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from mvp.app import app
from mvp import embeddings, transcription


@pytest.fixture(scope="module")
def audio_bytes():
    def make(seconds=3, rate=16000, channels=1, amplitude=0.15, pad=0,
             format="WAV", subtype="PCM_16", envelope=False):
        t = np.arange(int(seconds * rate)) / rate
        y = amplitude * np.sin(2 * np.pi * 180 * t)
        if envelope:
            y *= 0.6 + 0.4 * np.sin(2 * np.pi * 0.7 * t)
        if pad:
            y = np.pad(y, (int(pad * rate), int(pad * rate)))
        if channels > 1:
            y = np.stack([y / (i + 1) for i in range(channels)], axis=1)
        stream = BytesIO()
        sf.write(stream, y, rate, format=format, subtype=subtype)
        return stream.getvalue()
    return make


class FakeTranscriber:
    version, decoding, device = "fake-asr", "temperature fallback", "cpu"

    def __init__(self, lock=None):
        self.lock = lock
        self.waveforms = []
        self.text = "hello from the fake transcriber"
        self.error = None

    def transcribe(self, waveform):
        self.waveforms.append(waveform)
        if self.error:
            raise self.error
        return self.text


@pytest.fixture
def stub_models(monkeypatch):
    # Never download Whisper or load the ~1 GB ASR weights in API tests.
    monkeypatch.setattr(embeddings, "WhisperEmbedding",
                        lambda *args: lambda waveform: np.zeros(512, dtype=np.float32))
    monkeypatch.setattr(transcription, "Transcriber", FakeTranscriber)


@pytest.fixture
def client(stub_models):
    with TestClient(app) as client:
        yield client
