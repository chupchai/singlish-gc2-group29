"""mvp/asr_decoding.py must match the ASR project's decoding code. Runs only where the
sibling whisper_fine-tuned_transcription project exists (local development)."""
import importlib.util
import inspect
import sys
from pathlib import Path
import numpy as np
import pytest
import mvp.asr_decoding as vendored
import mvp.transcription as transcription

ASR = Path(__file__).resolve().parents[2] / "whisper_fine-tuned_transcription" / "scripts"
pytestmark = pytest.mark.skipif(not (ASR / "whisper_decoding.py").is_file(),
                                reason="sibling ASR project not present")


@pytest.fixture(scope="module")
def source():
    # Never write __pycache__ into the read-only ASR project.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(sys, "dont_write_bytecode", True)
        patch.syspath_prepend(str(ASR))
        added = []
        modules = {}
        for name, file in (("whisper_decoding", "whisper_decoding.py"), ("asr_transcribe_source", "transcribe.py")):
            previous = sys.modules.get(name)
            spec = importlib.util.spec_from_file_location(name, ASR / file)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            added.append((name, previous))
            spec.loader.exec_module(module)
            modules[name] = module
        try:
            yield modules["whisper_decoding"], modules["asr_transcribe_source"]
        finally:
            for name, previous in added:
                if previous is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = previous


def test_constants_match(source):
    decoding, transcribe = source
    for name in ("LANGUAGE", "TASK", "SAMPLE_RATE", "FALLBACK_TEMPERATURES", "DEFAULT_COMPRESSION_RATIO_THRESHOLD",
                 "DEFAULT_LOGPROB_THRESHOLD", "DEFAULT_SEED"):
        assert getattr(vendored, name) == getattr(decoding, name), name
    for name in ("CHUNK_SECONDS", "MIN_TAIL_SECONDS", "BATCH_SIZE"):
        assert getattr(vendored, name) == getattr(transcribe, name), name
    assert transcription.MODEL_NAME == transcribe.DEFAULT_MODEL.name


@pytest.mark.parametrize("kwargs", [{}, {"temperature_fallback": False},
                                    {"no_repeat_ngram_size": 3, "max_new_tokens": 50},
                                    {"compression_ratio_threshold": 2.0, "logprob_threshold": -0.5}],
                         ids=["default", "plain", "ngram_max_tokens", "thresholds"])
def test_generate_kwargs_match(source, kwargs):
    decoding, _ = source
    assert vendored.generate_kwargs(**kwargs) == decoding.generate_kwargs(**kwargs)


@pytest.mark.parametrize("name", ["generate_kwargs", "generated_sequences", "select_device", "decode_batch", "chunk_audio"])
def test_function_source_matches(source, name):
    decoding, transcribe = source
    origin = transcribe if name == "chunk_audio" else decoding
    assert inspect.getsource(getattr(vendored, name)) == inspect.getsource(getattr(origin, name))


@pytest.mark.parametrize("length", [0, 1, 480000, 480001, 481600, 1040000])
def test_chunk_audio_outputs_match(source, length):
    _, transcribe = source
    audio = np.arange(length, dtype=np.float32)
    ours, theirs = vendored.chunk_audio(audio, 16000), transcribe.chunk_audio(audio, 16000)
    assert len(ours) == len(theirs)
    assert all(np.array_equal(a, b) for a, b in zip(ours, theirs))
