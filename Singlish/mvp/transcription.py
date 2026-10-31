"""Speech-to-text with the fine-tuned Whisper small model and temperature-fallback decoding."""
import json
import os
from pathlib import Path
from threading import Lock
import numpy as np
import torch
import transformers
from transformers import (WhisperFeatureExtractor, WhisperForConditionalGeneration,
                          WhisperProcessor, WhisperTokenizerFast, set_seed)
from .asr_decoding import (BATCH_SIZE, DEFAULT_SEED, LANGUAGE, SAMPLE_RATE, TASK, chunk_audio,
                           decode_batch, generate_kwargs, select_device)

ROOT = Path(__file__).resolve().parent
MODEL_NAME = "whisper-small-mnsc-sg-15k-heldout"
DECODING = "temperature fallback"


def resolve_model_dir():
    """ASR_MODEL_PATH (relative paths resolve from mvp/), else mvp/models/<model>,
    else the sibling ASR project's models/<model> for local development."""
    override = os.environ.get("ASR_MODEL_PATH")
    if override:
        path = Path(override)
        if not path.is_absolute():
            path = ROOT / path
        if not (path / "config.json").is_file():
            raise FileNotFoundError("ASR_MODEL_PATH does not contain a saved model (config.json)")
        return path
    for path in (ROOT / "models" / MODEL_NAME,
                 ROOT.parent / "whisper_fine-tuned_transcription" / "models" / MODEL_NAME):
        if (path / "config.json").is_file():
            return path
    raise FileNotFoundError(f"ASR model not found; set ASR_MODEL_PATH or copy the model to mvp/models/{MODEL_NAME}")


def load_processor(model_dir):
    if int(transformers.__version__.split(".")[0]) < 5:
        # transformers 4.x cannot read the v5-saved tokenizer_config.json (extra_special_tokens
        # is a list there, 4.x expects a dict). Build the processor in memory instead.
        config = json.loads((Path(model_dir) / "tokenizer_config.json").read_text(encoding="utf-8"))
        tokenizer = WhisperTokenizerFast.from_pretrained(
            str(model_dir), extra_special_tokens={},
            additional_special_tokens=config.get("extra_special_tokens") or [],
            clean_up_tokenization_spaces=False)
        processor = WhisperProcessor(feature_extractor=WhisperFeatureExtractor.from_pretrained(str(model_dir)),
                                     tokenizer=tokenizer)
    else:
        processor = WhisperProcessor.from_pretrained(str(model_dir))
    # 4.57.6 would otherwise apply the saved clean_up_tokenization_spaces=True when decoding
    # ("oh!wow!" instead of "oh !wow!"); 5.x, which evaluated the model, does not.
    processor.tokenizer.clean_up_tokenization_spaces = False
    return processor


def load_model(model_dir, device):
    model = WhisperForConditionalGeneration.from_pretrained(str(model_dir))
    # fp16 on GPU; CPU uses float32.
    dtype = torch.float16 if device == "cuda" else torch.float32
    model.to(device, dtype=dtype)
    model.eval()
    # In memory only; nothing is written to the model folder.
    model.generation_config.language = LANGUAGE
    model.generation_config.task = TASK
    return model, dtype


class Transcriber:
    def __init__(self, model_dir=None, lock=None, device=None):
        self.model_dir = Path(model_dir) if model_dir else resolve_model_dir()
        self.device = device or select_device()
        self.processor = load_processor(self.model_dir)
        self.model, self.dtype = load_model(self.model_dir, self.device)
        self.gen_kwargs = generate_kwargs()
        self.version = os.environ.get("ASR_MODEL_VERSION") or self.model_dir.name
        self.decoding = DECODING
        # Share the classifier lock so ASR and classification never run concurrently.
        self.lock = lock or Lock()

    def transcribe(self, waveform):
        if (not isinstance(waveform, np.ndarray) or waveform.dtype != np.float32 or waveform.ndim != 1
                or not waveform.size or not np.isfinite(waveform).all()):
            raise ValueError("Expected a nonempty 1-D finite float32 waveform at 16 kHz")
        chunks = chunk_audio(waveform, SAMPLE_RATE)
        texts = []
        with self.lock:
            # Seed per request: the same upload gives the same transcript, and sampled
            # fallback passes are seeded as in the ASR project's transcribe/evaluation.
            set_seed(DEFAULT_SEED)
            for start in range(0, len(chunks), BATCH_SIZE):
                batch, _ = decode_batch(self.model, self.processor, chunks[start:start + BATCH_SIZE],
                                        SAMPLE_RATE, self.device, self.dtype, self.gen_kwargs)
                texts.extend(batch)
        return " ".join(t.strip() for t in texts if t.strip())
