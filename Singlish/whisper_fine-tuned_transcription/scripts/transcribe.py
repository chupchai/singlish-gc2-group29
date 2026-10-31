"""
Transcribe your own audio files with a fine-tuned Whisper model (app-style CLI).

Uses the same decoding code as evaluate_whisper_speaker_split.py (whisper_decoding.py):
temperature fallback by default (--no_temperature_fallback = plain greedy),
language="en", task="transcribe", fp16 on GPU if available, else CPU.

Audio: WAV / FLAC / OGG / MP3 (whatever libsndfile reads), mixed down to mono and
resampled to 16 kHz. Files longer than 30 s are cut into fixed 30 s windows that are
transcribed separately and joined (a word can be split at a window boundary).

Output: one line per file on stdout, "<path><TAB><transcript>"; progress goes to stderr.
--out file.csv also writes path,transcript,duration_sec,n_chunks,decoding (never overwrites
an existing file without --overwrite). Nothing is written to the model folder.

Usage (from the project root):
    python scripts/transcribe.py path/to/audio1.wav path/to/audio2.mp3
    python scripts/transcribe.py --model models/whisper-small-mnsc-sg-15k-heldout clip.wav --out transcripts.csv
    python scripts/transcribe.py --no_temperature_fallback clip.wav   # plain greedy
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import soundfile as sf

from transformers import set_seed

from whisper_decoding import (
    SAMPLE_RATE,
    add_decoding_args,
    decode_batch,
    decoding_kwargs,
    decoding_mode_label,
    load_model,
    resolve_decoding_args,
)

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent              # scripts/ holds code only

# ============================================================
# CONFIGURATION
# ============================================================

# Default model: the held-out retrain, validated on unseen PART3 speakers
# (closest to what an app faces). Relative paths resolve against the project root.
DEFAULT_MODEL = Path("models") / "whisper-small-mnsc-sg-15k-heldout"

CHUNK_SECONDS = 30.0     # Whisper's input window
MIN_TAIL_SECONDS = 0.1   # a shorter final window is dropped (unless it is the only one)
BATCH_SIZE = 8           # windows per generate() call (as in the evaluation)

CSV_COLUMNS = ["path", "transcript", "duration_sec", "n_chunks", "decoding"]


# ============================================================
# AUDIO
# ============================================================

def resolve_model_dir(model_arg) -> Path:
    """Absolute paths as given; relative paths against the project root."""
    p = Path(model_arg)
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_audio(path) -> Tuple[np.ndarray, int]:
    """(mono float32 audio, sample rate) of any file libsndfile can read."""
    try:
        audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    except (sf.LibsndfileError, RuntimeError) as exc:
        raise ValueError(f"cannot read {path}: {exc} (supported: WAV, FLAC, OGG, MP3; "
                         "convert other formats with ffmpeg)") from exc
    return audio.mean(axis=1).astype(np.float32), int(sr)


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """Band-limited FFT resampling (numpy only)."""
    audio = np.asarray(audio, dtype=np.float32)
    if orig_sr == target_sr:
        return audio
    if len(audio) == 0:
        return np.zeros(0, dtype=np.float32)
    n_out = int(round(len(audio) * target_sr / orig_sr))
    if n_out == 0:
        return np.zeros(0, dtype=np.float32)
    spectrum = np.fft.rfft(audio.astype(np.float64))
    out_spec = np.zeros(n_out // 2 + 1, dtype=np.complex128)
    n_keep = min(len(spectrum), len(out_spec))
    out_spec[:n_keep] = spectrum[:n_keep]
    return (np.fft.irfft(out_spec, n_out) * (n_out / len(audio))).astype(np.float32)


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


def transcribe_file(path, model, processor, device, dtype, gen_kwargs: dict,
                    batch_size: int = BATCH_SIZE) -> Tuple[str, float, int]:
    """(transcript, duration_sec, n_chunks) of one audio file."""
    audio, sr = load_audio(path)
    duration = len(audio) / sr if sr else 0.0
    chunks = [resample_audio(c, sr) for c in chunk_audio(audio, sr)]
    texts = []
    for start in range(0, len(chunks), batch_size):
        batch, _ = decode_batch(model, processor, chunks[start:start + batch_size], SAMPLE_RATE,
                                device, dtype, gen_kwargs)
        texts.extend(batch)
    return " ".join(t.strip() for t in texts if t.strip()), duration, len(chunks)


# ============================================================
# MAIN
# ============================================================

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Transcribe audio files with a fine-tuned Whisper model "
                                            "(temperature fallback by default).")
    p.add_argument("audio", nargs="+", help="Audio files (WAV, FLAC, OGG, MP3).")
    p.add_argument("--model", default=str(DEFAULT_MODEL),
                   help=f"Saved model folder; relative paths are resolved against the project root "
                        f"(default {DEFAULT_MODEL}).")
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                   help=f"30 s windows per generate() call (default {BATCH_SIZE}).")
    p.add_argument("--out", default=None,
                   help="Optional CSV with path,transcript,duration_sec,n_chunks,decoding.")
    p.add_argument("--overwrite", action="store_true",
                   help="Replace an existing --out file (never replaced by default).")
    add_decoding_args(p)
    args = p.parse_args(argv)
    if args.batch_size < 1:
        p.error("--batch_size must be at least 1")
    resolve_decoding_args(p, args)
    return args


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    model_dir = resolve_model_dir(args.model)
    if not (model_dir / "config.json").exists():
        print(f"ERROR: no saved model at {model_dir} (pass --model <folder with config.json>).",
              file=sys.stderr)
        return 2
    out_path = Path(args.out) if args.out else None
    if out_path is not None and out_path.exists() and not args.overwrite:
        print(f"ERROR: {out_path} already exists; refusing to overwrite. Pass --overwrite to replace.",
              file=sys.stderr)
        return 2

    set_seed(args.seed)  # reproducible sampled fallback passes
    mode = decoding_mode_label(args.temperature_fallback, args.no_repeat_ngram_size)
    gen_kwargs = decoding_kwargs(args)
    processor, model, device, dtype = load_model(model_dir)
    print(f"Model   : {model_dir}", file=sys.stderr)
    print(f"Device  : {device} ({dtype})", file=sys.stderr)
    print(f"decoding: {mode}", file=sys.stderr)
    print(f"generate() options: {gen_kwargs or 'none (plain greedy)'}", file=sys.stderr)

    rows = []
    failed = False
    for path in args.audio:
        try:
            text, duration, n_chunks = transcribe_file(path, model, processor, device, dtype,
                                                       gen_kwargs, args.batch_size)
        except Exception as exc:  # unreadable audio, CUDA OOM, ...: skip this file, keep the rest
            print(f"ERROR: {path}: {exc}", file=sys.stderr)
            failed = True
            continue
        print(f"{path}\t{text}", flush=True)
        rows.append({"path": path, "transcript": text, "duration_sec": round(duration, 3),
                     "n_chunks": n_chunks, "decoding": mode})

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved {len(rows)} rows to {out_path}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
