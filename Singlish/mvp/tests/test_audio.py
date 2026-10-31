"""24 Oct normalized PCM16 preparation and full, original-gain ASR audio."""
from io import BytesIO

import numpy as np
import pytest
import soundfile as sf
import soxr

from mvp.audio import AudioError, decode_audio, prepare_transcription_audio, read_audio
from mvp.features import SAMPLE_RATE, MAX_SECONDS, trim_silence, normalize_loudness

CASES = {
    "3s_16k_wav": {"seconds": 3},
    "6s_44k_stereo_flac": {"seconds": 6, "rate": 44100, "channels": 2, "format": "FLAC"},
    "8s_wav_truncated": {"seconds": 8},
    "10s_flac": {"seconds": 10, "format": "FLAC"},
    "4s_padded": {"seconds": 4, "pad": 1},
    "pcm24_48k": {"seconds": 5, "rate": 48000, "subtype": "PCM_24"},
    "pcm_u8_8k": {"seconds": 4, "rate": 8000, "subtype": "PCM_U8"},
    "quiet_tone": {"seconds": 4, "amplitude": .02},
}


@pytest.mark.parametrize("case", CASES)
def test_classifier_matches_training_preparation(audio_bytes, case):
    data = audio_bytes(**CASES[case], envelope=True)
    # Decode independently so channel mixing, resampling and PCM16 conversion
    # are checked against the training preparation, not the serving helper.
    expected, rate = sf.read(BytesIO(data), dtype="float32")
    if expected.ndim > 1:
        expected = expected.mean(axis=1)
    if rate != SAMPLE_RATE:
        expected = soxr.resample(expected, rate, SAMPLE_RATE, quality="HQ").astype(np.float32)
    trimmed = trim_silence(expected)
    expected = normalize_loudness(trimmed[:MAX_SECONDS * SAMPLE_RATE])
    stream = BytesIO()
    sf.write(stream, expected, SAMPLE_RATE, format="FLAC", subtype="PCM_16")
    stream.seek(0)
    expected, _ = sf.read(stream, dtype="float32")
    actual, metadata, warnings = decode_audio(data)
    assert actual.dtype == np.float32
    np.testing.assert_array_equal(actual, expected)
    assert metadata["truncated"] == (len(trimmed) > MAX_SECONDS * SAMPLE_RATE)
    assert bool(warnings) == metadata["truncated"]


@pytest.mark.parametrize("format,seconds", [("WAV", 8), ("FLAC", 10)])
def test_classifier_crops_while_transcription_keeps_full_clip(audio_bytes, format, seconds):
    data = audio_bytes(seconds, format=format)
    classified, metadata, warnings = decode_audio(data)
    first_six, _, _ = decode_audio(audio_bytes(6, format=format))
    np.testing.assert_array_equal(classified, first_six)
    assert metadata == {"duration_seconds": seconds, "classified_duration_seconds": 6, "truncated": True}
    assert warnings == ["Only the first six seconds after silence trimming were classified."]
    transcribed, metadata = prepare_transcription_audio(*read_audio(data))
    np.testing.assert_array_equal(transcribed, trim_silence(read_audio(data)[0]))
    assert transcribed.dtype == np.float32 and len(transcribed) > MAX_SECONDS * SAMPLE_RATE
    assert metadata == {"duration_seconds": seconds, "transcribed_duration_seconds": round(len(transcribed) / SAMPLE_RATE, 4)}


def test_only_classification_normalizes_loudness(audio_bytes):
    data = audio_bytes(4, amplitude=.02, pad=1)
    classified, classification, _ = decode_audio(data)
    transcribed, transcription = prepare_transcription_audio(*read_audio(data))
    assert 4 <= classification["classified_duration_seconds"] <= 4.25
    assert np.sqrt(np.mean(classified.astype(np.float64) ** 2)) == pytest.approx(.1, abs=2e-5)
    assert np.max(np.abs(classified)) <= .991
    assert .013 <= np.sqrt(np.mean(transcribed.astype(np.float64) ** 2)) <= .0142
    assert transcription == {"duration_seconds": 6, "transcribed_duration_seconds": classification["classified_duration_seconds"]}


def test_very_quiet_audio_is_rejected():
    stream = BytesIO()
    sf.write(stream, np.full(4 * SAMPLE_RATE, 1e-7), SAMPLE_RATE, format="WAV", subtype="PCM_32")
    with pytest.raises(AudioError, match="quiet"):
        decode_audio(stream.getvalue())


def test_trim_happens_before_crop(audio_bytes):
    processed, metadata, _ = decode_audio(audio_bytes(7, amplitude=.1, pad=1))
    assert len(processed) == 6 * SAMPLE_RATE
    # Retain ~100 ms boundary padding, not the full leading second of silence.
    active_start = np.flatnonzero(np.abs(processed) > 1e-3)[0] / SAMPLE_RATE
    assert .09 <= active_start <= .13
    assert metadata["duration_seconds"] == 9 and metadata["truncated"] is True
