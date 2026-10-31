"""
Vendored copy of the ASR project's shared Whisper decoding (temperature fallback).

Sources (sibling project, not deployed with the app):
  whisper_fine-tuned_transcription/scripts/whisper_decoding.py
    LANGUAGE, TASK, SAMPLE_RATE, FALLBACK_TEMPERATURES, DEFAULT_COMPRESSION_RATIO_THRESHOLD,
    DEFAULT_LOGPROB_THRESHOLD, DEFAULT_SEED, generate_kwargs, generated_sequences,
    select_device, decode_batch
  whisper_fine-tuned_transcription/scripts/transcribe.py
    CHUNK_SECONDS, MIN_TAIL_SECONDS, BATCH_SIZE, chunk_audio

KEEP IN SYNC: the functions are copied verbatim. mvp/tests/test_asr_decoding_sync.py
compares the constants, generate_kwargs() output and function source with the
sibling project when it exists (and skips otherwise). load_model is not copied:
it fails under transformers 4.57.6; mvp/transcription.py has the compatible loader.
"""

from typing import List, Tuple

import numpy as np
import torch

# ============================================================
# CONFIGURATION (whisper_decoding.py)
# ============================================================

LANGUAGE = "en"
TASK = "transcribe"
SAMPLE_RATE = 16000  # Whisper input rate

# Greedy first (0.0); a clip is re-decoded at the next temperature only if
# its output fails the compression-ratio or average-logprob check.
FALLBACK_TEMPERATURES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
DEFAULT_COMPRESSION_RATIO_THRESHOLD = 1.35  # HF: zlib ratio on token IDs (OpenAI's text-based 2.4 is a different scale)
DEFAULT_LOGPROB_THRESHOLD = -1.0
DEFAULT_SEED = 42

# ============================================================
# CHUNKING (transcribe.py)
# ============================================================

CHUNK_SECONDS = 30.0     # Whisper's input window
MIN_TAIL_SECONDS = 0.1   # a shorter final window is dropped (unless it is the only one)
BATCH_SIZE = 8           # windows per generate() call (as in the evaluation)


# ============================================================
# GENERATE() OPTIONS (whisper_decoding.py)
# ============================================================

def generate_kwargs(no_repeat_ngram_size: int = 0, max_new_tokens: int = 0,
                    temperature_fallback: bool = True,
                    compression_ratio_threshold: float = DEFAULT_COMPRESSION_RATIO_THRESHOLD,
                    logprob_threshold: float = DEFAULT_LOGPROB_THRESHOLD) -> dict:
    """Extra generate() kwargs ({} = plain greedy, identical to the 4k baseline decoding)."""
    kwargs = {}
    if no_repeat_ngram_size > 0:
        kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size
    if max_new_tokens > 0:
        kwargs["max_new_tokens"] = max_new_tokens
    if temperature_fallback:
        # No do_sample here: WhisperGenerationMixin sets do_sample per pass
        # (greedy at 0.0, sampling above) and drops a user-supplied one.
        kwargs["temperature"] = FALLBACK_TEMPERATURES
        kwargs["compression_ratio_threshold"] = compression_ratio_threshold
        kwargs["logprob_threshold"] = logprob_threshold
    return kwargs


def generated_sequences(outputs):
    """Token-ID tensor from generate(); transformers returns a dict/ModelOutput
    instead of a tensor when return_dict_in_generate is active."""
    if isinstance(outputs, torch.Tensor):
        return outputs
    return outputs["sequences"]


# ============================================================
# MODEL + BATCH DECODING (whisper_decoding.py)
# ============================================================

def select_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def decode_batch(model, processor, audios: List, sampling_rate: int, device: str,
                 dtype, gen_kwargs: dict) -> Tuple[List[str], "torch.Tensor"]:
    """(texts, token-ID sequences) for a batch of clips (each <= 30 s)."""

    # Whisper input (pads/trims every clip to 30 s, so the
    # batch is always a uniform tensor)
    inputs = processor(
        audios,
        sampling_rate=sampling_rate,
        return_tensors="pt"
    )

    # Match the model's dtype (fp16 on GPU) to avoid a dtype mismatch
    input_features = inputs.input_features.to(device, dtype=dtype)

    with torch.no_grad():

        predicted_ids = generated_sequences(model.generate(
            input_features,
            language=LANGUAGE,
            task=TASK,
            **gen_kwargs
        ))

    texts = processor.batch_decode(
        predicted_ids,
        skip_special_tokens=True
    )
    return texts, predicted_ids


# ============================================================
# AUDIO WINDOWS (transcribe.py)
# ============================================================

def chunk_audio(audio: np.ndarray, sr: int, chunk_seconds: float = CHUNK_SECONDS,
                min_tail_seconds: float = MIN_TAIL_SECONDS) -> List[np.ndarray]:
    """Consecutive non-overlapping windows of <= chunk_seconds."""
    if len(audio) == 0:
        return []
    size = int(chunk_seconds * sr)
    chunks = [audio[i:i + size] for i in range(0, len(audio), size)]
    if len(chunks) > 1 and len(chunks[-1]) < min_tail_seconds * sr:
        chunks.pop()
    return chunks
