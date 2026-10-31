"""Transcriber model-path resolution, decoding options and the transformers 4.x loader."""
import os
from threading import Lock
import numpy as np
import pytest
import torch
import mvp.transcription as transcription
from mvp.asr_decoding import generate_kwargs


def fake_model(path):
    path.mkdir(parents=True)
    (path / "config.json").write_text("{}")
    return path


@pytest.fixture
def root(tmp_path, monkeypatch):
    mvp = tmp_path / "mvp"
    mvp.mkdir()
    monkeypatch.setattr(transcription, "ROOT", mvp)
    monkeypatch.delenv("ASR_MODEL_PATH", raising=False)
    return mvp


def test_absolute_override_wins(root, tmp_path, monkeypatch):
    fake_model(root / "models" / transcription.MODEL_NAME)
    custom = fake_model(tmp_path / "elsewhere" / "custom-asr")
    monkeypatch.setenv("ASR_MODEL_PATH", str(custom))
    assert transcription.resolve_model_dir() == custom


def test_relative_override_resolves_from_mvp(root, tmp_path, monkeypatch):
    custom = fake_model(tmp_path / "shared" / "asr")
    monkeypatch.setenv("ASR_MODEL_PATH", os.path.relpath(custom, root))
    assert transcription.resolve_model_dir().resolve() == custom.resolve()


def test_override_without_model_fails(root, tmp_path, monkeypatch):
    (tmp_path / "empty").mkdir()
    fake_model(root / "models" / transcription.MODEL_NAME)
    monkeypatch.setenv("ASR_MODEL_PATH", str(tmp_path / "empty"))
    with pytest.raises(FileNotFoundError):
        transcription.resolve_model_dir()


def test_default_prefers_mvp_models_then_sibling(root):
    sibling = fake_model(root.parent / "whisper_fine-tuned_transcription" / "models" / transcription.MODEL_NAME)
    assert transcription.resolve_model_dir() == sibling
    local = fake_model(root / "models" / transcription.MODEL_NAME)
    assert transcription.resolve_model_dir() == local


def test_missing_model_raises(root):
    with pytest.raises(FileNotFoundError, match="ASR_MODEL_PATH"):
        transcription.resolve_model_dir()


@pytest.fixture
def transcriber():
    t = transcription.Transcriber.__new__(transcription.Transcriber)
    t.processor, t.model = object(), object()
    t.device, t.dtype = "cpu", torch.float32
    t.gen_kwargs = generate_kwargs()
    t.lock = Lock()
    return t


def test_transcribe_uses_seeded_fallback_under_lock(transcriber, monkeypatch):
    calls = []
    monkeypatch.setattr(transcription, "set_seed", lambda seed: calls.append(("seed", seed)))

    def decode(model, processor, audios, sampling_rate, device, dtype, gen_kwargs):
        calls.append(("decode", [len(a) for a in audios], sampling_rate, gen_kwargs, transcriber.lock.locked()))
        return [" first window ", "  second  "], None

    monkeypatch.setattr(transcription, "decode_batch", decode)
    text = transcriber.transcribe(np.zeros(35 * 16000, dtype=np.float32))
    assert text == "first window second"
    assert calls == [("seed", 42),
                     ("decode", [30 * 16000, 5 * 16000], 16000,
                      {"temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0), "compression_ratio_threshold": 1.35,
                       "logprob_threshold": -1.0}, True)]
    assert not transcriber.lock.locked()


@pytest.mark.parametrize("waveform", [np.zeros((2, 16000), dtype=np.float32), np.full(16000, np.nan, dtype=np.float32),
                                      np.zeros(0, dtype=np.float32), np.zeros(16000)],
                         ids=["2d", "nan", "empty", "float64"])
def test_transcribe_rejects_invalid_waveforms(transcriber, waveform):
    with pytest.raises(ValueError):
        transcriber.transcribe(waveform)


def test_version_defaults_to_directory_name(tmp_path, monkeypatch):
    monkeypatch.setattr(transcription, "load_processor", lambda path: object())
    monkeypatch.setattr(transcription, "load_model", lambda path, device: (object(), torch.float32))
    monkeypatch.delenv("ASR_MODEL_VERSION", raising=False)
    t = transcription.Transcriber(model_dir=tmp_path / "my-asr", device="cpu")
    assert (t.version, t.decoding, t.dtype) == ("my-asr", "temperature fallback", torch.float32)
    assert t.gen_kwargs == generate_kwargs()
    monkeypatch.setenv("ASR_MODEL_VERSION", "asr-experiment-3")
    assert transcription.Transcriber(model_dir=tmp_path / "my-asr", device="cpu").version == "asr-experiment-3"


def real_model_dir():
    try:
        return transcription.resolve_model_dir()
    except FileNotFoundError:
        return None


@pytest.mark.skipif(real_model_dir() is None, reason="ASR model not present")
def test_real_processor_loads_under_installed_transformers():
    # Tokenizer and feature extractor only; the weights are not loaded.
    processor = transcription.load_processor(real_model_dir())
    tokenizer = processor.tokenizer
    assert tokenizer.clean_up_tokenization_spaces is False
    assert tokenizer.convert_tokens_to_ids(["<|en|>", "<|transcribe|>", "<|notimestamps|>"]) == [50259, 50359, 50363]
    assert processor.feature_extractor.sampling_rate == 16000
    assert tokenizer.decode(tokenizer.encode(" oh !wow!", add_special_tokens=False)) == " oh !wow!"
