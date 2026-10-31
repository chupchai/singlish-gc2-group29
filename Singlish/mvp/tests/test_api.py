"""Both upload endpoints: validation, real classifier parity and optional ASR."""
from io import BytesIO

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from mvp.app import app
from mvp import transcription
from mvp.audio import MAX_BYTES, decode_audio, read_audio
from mvp.features import extract_features, trim_silence

ENDPOINTS = ["/api/classify", "/api/transcribe"]


@pytest.fixture(scope="module")
def invalid_audio(audio_bytes):
    # One validation matrix for both endpoints; explicit statuses prevent them
    # from passing merely because they return the same incorrect response.
    return {
        "empty": (b"", 422),
        "not_wav": (b"not wav", 415),
        "bad_riff": (b"RIFF1234WAVEbad", 422),
        "bad_flac": (b"fLaCbad", 422),
        "silent": (audio_bytes(amplitude=0), 422),
        "too_short": (audio_bytes(.2), 422),
        "too_long": (audio_bytes(11), 422),
        "float_wav": (audio_bytes(subtype="FLOAT"), 415),
        "too_big": (b"x" * (MAX_BYTES + 1), 413),
    }


@pytest.mark.parametrize("case", ["empty", "not_wav", "bad_riff", "bad_flac", "silent",
                                  "too_short", "too_long", "float_wav", "too_big"])
def test_invalid_audio(client, invalid_audio, case):
    data, status = invalid_audio[case]
    responses = [client.post(route, files={"audio": ("clip.wav", data)}) for route in ENDPOINTS]
    for response in responses:
        assert response.status_code == status
        assert response.json()["error"]["message"]
    assert responses[0].json() == responses[1].json()
    assert client.app.state.transcriber.waveforms == []


@pytest.mark.parametrize("route", ENDPOINTS)
def test_request_contract(client, audio_bytes, route):
    data = audio_bytes()
    for files in [{}, {"wrong": ("clip.wav", data)},
                  [("audio", ("a.wav", data)), ("audio", ("b.wav", data))]]:
        assert client.post(route, files=files).status_code == 422
    assert client.post(route, content=b"bad", headers={"Content-Type": "application/json"}).status_code == 422
    response = client.post(route, content=b"x" * (MAX_BYTES + 65537))
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "file_too_large"


@pytest.mark.parametrize("seconds,status", [(2.5, 422), (3, 200)])
def test_minimum_duration_is_checked_after_trimming(client, audio_bytes, monkeypatch, seconds, status):
    if status == 422:
        def unexpected_inference(*args):
            pytest.fail("Short clips must be rejected before either model runs")
        monkeypatch.setattr(client.app.state.models, "predict", unexpected_inference)
        monkeypatch.setattr(client.app.state.transcriber, "transcribe", unexpected_inference)
    for route in ENDPOINTS:
        response = client.post(route, files={"audio": ("padded.wav", audio_bytes(seconds, pad=1))})
        assert response.status_code == status, response.text
        if status == 422:
            assert response.json()["error"]["code"] == "invalid_duration"
            assert "Please retry" in response.json()["error"]["message"]


@pytest.mark.parametrize("rate,channels,seconds,format", [(16000, 1, 3, "WAV"), (44100, 2, 6, "FLAC")])
def test_classification_matches_saved_pipelines(client, audio_bytes, rate, channels, seconds, format):
    data = audio_bytes(seconds, rate, channels, format=format)
    response = client.post("/api/classify", files={"audio": ("clip." + format.lower(), data)})
    assert response.status_code == 200, response.text
    result = response.json()
    y, metadata, warnings = decode_audio(data)
    vector = extract_features(y)
    ml = client.app.state.models.ml
    values = vector[ml.columns].reshape(1, -1)
    probabilities = ml.pipeline.predict_proba(values)[0]
    expected = int(probabilities[1] >= 0.5)
    prediction = result["models"]["ml"]["prediction"]
    assert prediction["decision_score"] == pytest.approx(ml.decision_estimator.decision_function(values)[0])
    assert prediction["class_id"] == expected
    assert prediction["probability"] == pytest.approx(probabilities[expected])
    assert result["models"]["ml"]["version"] == "svm_65_c_261024"
    assert result["model_version"] == "fusion_svm_c_261024"
    assert result["final_model"] == "ensemble"
    ensemble = client.app.state.models.ensemble.predict(vector, np.zeros(512, dtype=np.float32))
    assert result["prediction"] == result["models"]["ensemble"]["prediction"] == ensemble
    assert result["prediction"]["probability"] >= 0.5
    assert result["agreement"] == ("agree" if expected == ensemble["class_id"] else "disagree")
    assert result["audio"] == metadata and result["warnings"] == warnings
    assert metadata["truncated"] is False and not warnings


def test_content_validation_ignores_filename_and_mime(client, audio_bytes):
    # Convert already quantized WAV samples, so both containers hold identical
    # PCM16 audio (encoding a fresh float signal can round differently).
    wav = audio_bytes(4)
    samples, rate = sf.read(BytesIO(wav), dtype="float32")
    stream = BytesIO()
    sf.write(stream, samples, rate, format="FLAC", subtype="PCM_16")
    flac = stream.getvalue()
    np.testing.assert_array_equal(decode_audio(wav)[0], decode_audio(flac)[0])
    result = client.post("/api/classify", files={"audio": ("clip.wav", flac, "audio/wav")})
    assert result.status_code == 200
    assert result.json() == client.post("/api/classify", files={"audio": ("clip.wav", wav)}).json()


@pytest.mark.parametrize("format,seconds", [("WAV", 8), ("FLAC", 10)])
def test_long_upload_classifies_six_seconds_and_transcribes_full_clip(client, audio_bytes, format, seconds):
    data = audio_bytes(seconds, amplitude=.02, format=format)
    classified = client.post("/api/classify", files={"audio": ("clip.wav", data)})
    transcribed = client.post("/api/transcribe", files={"audio": ("clip.wav", data)})
    assert classified.status_code == transcribed.status_code == 200
    result, body = classified.json(), transcribed.json()
    first_six, _, _ = decode_audio(audio_bytes(6, amplitude=.02, format=format))
    assert result["audio"] == {"duration_seconds": seconds, "classified_duration_seconds": 6, "truncated": True}
    assert result["warnings"] == ["Only the first six seconds after silence trimming were classified."]
    assert result["prediction"] == client.app.state.models.ensemble.predict(
        extract_features(first_six), np.zeros(512, dtype=np.float32))
    expected = trim_silence(read_audio(data)[0])
    np.testing.assert_array_equal(client.app.state.transcriber.waveforms[-1], expected)
    assert body == {"transcript": "hello from the fake transcriber", "model_version": "fake-asr",
                    "decoding": "temperature fallback", "warnings": [],
                    "audio": {"duration_seconds": seconds, "transcribed_duration_seconds": round(len(expected) / 16000, 4)}}


def test_transcription_receives_resampled_stereo_flac(client, audio_bytes):
    data = audio_bytes(10, 44100, 2, amplitude=.02, format="FLAC")
    response = client.post("/api/transcribe", files={"audio": ("clip.flac", data)})
    assert response.status_code == 200
    np.testing.assert_array_equal(client.app.state.transcriber.waveforms[-1], trim_silence(read_audio(data)[0]))
    assert len(client.app.state.transcriber.waveforms[-1]) >= 10 * 16000 - 1


def test_classification_failure_returns_error(client, audio_bytes, monkeypatch):
    def fail(_):
        raise RuntimeError("Encoder unavailable")
    monkeypatch.setattr(client.app.state.models, "embedding", fail)
    response = client.post("/api/classify", files={"audio": ("clip.wav", audio_bytes())})
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "inference_failed"
    assert "prediction" not in response.json()


def test_transcription_failure_keeps_classification_working(client, audio_bytes):
    client.app.state.transcriber.error = RuntimeError("decoder crashed")
    response = client.post("/api/transcribe", files={"audio": ("clip.wav", audio_bytes())})
    assert response.status_code == 500
    assert response.json() == {"error": {"code": "transcription_failed",
                                         "message": "Transcription failed. Please try another WAV or FLAC file."}}
    assert client.post("/api/classify", files={"audio": ("clip.wav", audio_bytes())}).status_code == 200


def test_empty_transcript_returns_warning(client, audio_bytes):
    client.app.state.transcriber.text = ""
    response = client.post("/api/transcribe", files={"audio": ("clip.wav", audio_bytes())})
    assert response.status_code == 200
    assert response.json()["transcript"] == ""
    assert response.json()["warnings"] == ["No speech was recognised in this recording."]


def test_page_health_and_shared_lock(client):
    page = client.get("/")
    assert page.status_code == 200 and page.headers["cache-control"] == "no-cache"
    assert client.get("/static/app.js").status_code == 200
    health = client.get("/api/health").json()
    assert health["status"] == "ready"
    assert health["models"] == {"ml": "svm_65_c_261024", "ensemble": "fusion_svm_c_261024"}
    assert health["asr"] == {"ready": True, "model_version": "fake-asr", "decoding": "temperature fallback", "device": "cpu"}
    assert client.app.state.transcriber.lock is client.app.state.models.lock


@pytest.mark.parametrize("error,reason", [(FileNotFoundError("ASR model not found"), "model_not_found"),
                                         (RuntimeError("corrupt weights"), "load_failed")])
def test_unavailable_asr_keeps_classification_working(stub_models, audio_bytes, monkeypatch, error, reason):
    def unavailable(**kwargs):
        raise error
    monkeypatch.setattr(transcription, "Transcriber", unavailable)
    with TestClient(app) as client:
        health = client.get("/api/health").json()
        assert health["status"] == "ready" and health["model_version"] == "fusion_svm_c_261024"
        assert health["asr"] == {"ready": False, "model_version": None, "decoding": "temperature fallback", "reason": reason}
        response = client.post("/api/transcribe", files={"audio": ("clip.wav", audio_bytes())})
        assert response.status_code == 503
        assert response.json() == {"error": {"code": "transcription_unavailable",
                                             "message": "Transcription is unavailable on this server. Classification still works."}}
        classified = client.post("/api/classify", files={"audio": ("clip.wav", audio_bytes())})
        assert classified.status_code == 200
        assert set(classified.json()) == {"prediction", "final_model", "models", "agreement", "model_version", "audio", "warnings"}
