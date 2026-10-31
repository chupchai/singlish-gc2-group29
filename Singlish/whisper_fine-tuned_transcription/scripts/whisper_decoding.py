"""
Shared Whisper decoding for our scripts (reference implementation).

Used by evaluate_whisper_speaker_split.py and transcribe.py, so evaluation numbers and
app transcripts come from exactly the same code: generate(language="en", task="transcribe"),
fp16 on GPU, greedy first pass.

Temperature fallback is the DEFAULT (chosen by the validation sweeps):
OpenAI-style greedy first pass, a clip is re-decoded at temperature 0.2..1.0 only when its
output fails the compression-ratio / average-logprob check. --no_temperature_fallback gives
plain greedy decoding. Nothing here is written to the saved models: their
generation_config.json is unchanged, so a raw model.generate() / HF pipeline call is still
plain greedy unless the caller passes these options.
"""

import argparse
import math
import zlib
from typing import List, Optional, Tuple

import torch

from transformers import WhisperProcessor, WhisperForConditionalGeneration

# ============================================================
# CONFIGURATION
# ============================================================

LANGUAGE = "en"
TASK = "transcribe"
SAMPLE_RATE = 16000  # Whisper input rate

# Temperature fallback (default ON; --no_temperature_fallback turns it off).
# Greedy first (0.0); a clip is re-decoded at the next temperature only if
# its output fails the compression-ratio or average-logprob check.
FALLBACK_TEMPERATURES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
DEFAULT_COMPRESSION_RATIO_THRESHOLD = 1.35  # HF: zlib ratio on token IDs (OpenAI's text-based 2.4 is a different scale)
DEFAULT_LOGPROB_THRESHOLD = -1.0
DEFAULT_SEED = 42


# ============================================================
# GENERATE() OPTIONS
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


def decoding_kwargs(args) -> dict:
    """generate_kwargs() from a namespace parsed with add_decoding_args/resolve_decoding_args."""
    return generate_kwargs(args.no_repeat_ngram_size, args.max_new_tokens,
                           args.temperature_fallback, args.compression_ratio_threshold,
                           args.logprob_threshold)


def generated_sequences(outputs):
    """Token-ID tensor from generate(); transformers returns a dict/ModelOutput
    instead of a tensor when return_dict_in_generate is active."""
    if isinstance(outputs, torch.Tensor):
        return outputs
    return outputs["sequences"]


def token_compression_ratio(token_ids, vocab_size: int, first_special_id: int) -> float:
    """zlib compression ratio of the predicted token IDs, as in
    WhisperGenerationMixin._retrieve_compression_ratio (the value compared with
    --compression_ratio_threshold). HF checks the text tokens plus the final EOS;
    the returned batch tensor may have lost that EOS (stripped for the longest
    item), so special tokens (ids >= first_special_id, i.e. <|endoftext|> and
    above: prompt, EOS, padding) are dropped and one EOS is always appended.
    Exact for outputs that end with EOS; one token off for outputs cut at the
    length limit (those loops are far above any threshold anyway)."""
    ids = [int(t) for t in token_ids if int(t) < first_special_id]
    if not ids:
        return 0.0
    ids.append(first_special_id)
    length = int(math.log2(vocab_size) / 8) + 1
    raw = b"".join(t.to_bytes(length, "little") for t in ids)
    return len(raw) / len(zlib.compress(raw))


# ============================================================
# COMMAND-LINE OPTIONS (same flags in every script)
# ============================================================

def add_decoding_args(parser: argparse.ArgumentParser) -> None:
    """Decoding flags; call resolve_decoding_args() after parse_args()."""
    parser.add_argument("--no_repeat_ngram_size", type=int, default=0,
                        help="OFF by default (0). Passed to generate() only if > 0, "
                             "to measure the repetition-loop effect separately.")
    parser.add_argument("--max_new_tokens", type=int, default=0,
                        help="OFF by default (0). Passed to generate() only if > 0.")
    parser.add_argument("--temperature_fallback", action="store_true",
                        help="Default ON (since 2026-10-11); accepted for old commands, no effect. "
                             "OpenAI-style fallback: greedy first, re-decode at temperature 0.2..1.0 "
                             "when the compression ratio / avg logprob check fails. "
                             "Decoding-only, not saved to the model.")
    parser.add_argument("--no_temperature_fallback", action="store_true",
                        help="Plain greedy decoding (as in the plain results and a raw "
                             "model.generate() call); eval output files keep the untagged names.")
    parser.add_argument("--compression_ratio_threshold", type=float, default=None,
                        help=f"Fallback threshold (default {DEFAULT_COMPRESSION_RATIO_THRESHOLD}): "
                             "zlib ratio of the generated token IDs (HF scale, not OpenAI's text-based 2.4).")
    parser.add_argument("--logprob_threshold", type=float, default=None,
                        help=f"Fallback threshold (default {DEFAULT_LOGPROB_THRESHOLD}): "
                             "re-decode if the average token logprob is below this.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"Random seed (default {DEFAULT_SEED}); makes sampled fallback passes reproducible.")


def resolve_decoding_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Check the decoding flags and set args.temperature_fallback + threshold defaults."""
    if args.temperature_fallback and args.no_temperature_fallback:
        parser.error("--temperature_fallback and --no_temperature_fallback cannot be combined")
    if args.no_temperature_fallback and (args.compression_ratio_threshold is not None
                                         or args.logprob_threshold is not None):
        parser.error("--compression_ratio_threshold/--logprob_threshold cannot be used with "
                     "--no_temperature_fallback")
    args.temperature_fallback = not args.no_temperature_fallback
    if args.compression_ratio_threshold is None:
        args.compression_ratio_threshold = DEFAULT_COMPRESSION_RATIO_THRESHOLD
    if args.logprob_threshold is None:
        args.logprob_threshold = DEFAULT_LOGPROB_THRESHOLD


def decoding_mode_label(temperature_fallback: bool, no_repeat_ngram_size: int = 0) -> str:
    """Human-readable decoding mode printed at startup."""
    label = ("temperature fallback (default)" if temperature_fallback
             else "plain greedy (--no_temperature_fallback)")
    if no_repeat_ngram_size > 0:
        label += f" + no_repeat_ngram_size {no_repeat_ngram_size}"
    return label


# ============================================================
# MODEL + BATCH DECODING
# ============================================================

def select_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model(model_path, device: Optional[str] = None):
    """(processor, model, device, dtype) for a saved model folder. Settings are applied
    in memory only; nothing is written to the model folder."""
    device = device or select_device()

    processor = WhisperProcessor.from_pretrained(
        str(model_path)
    )

    model = WhisperForConditionalGeneration.from_pretrained(
        str(model_path)
    )

    # fp16 on GPU roughly halves memory and speeds up decoding.
    # Cast after loading so it works on every transformers version.
    model_dtype = torch.float16 if device == "cuda" else torch.float32
    model.to(device, dtype=model_dtype)
    model.eval()

    # Force English transcription
    model.generation_config.language = LANGUAGE
    model.generation_config.task = TASK
    return processor, model, device, model_dtype


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
