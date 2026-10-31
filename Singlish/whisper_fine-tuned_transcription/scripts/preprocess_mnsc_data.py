"""
preprocess_mnsc_data.py

Shared data-preprocessing script for the MNSC Extended dataset.

Purpose
-------
Download (or reuse cached) MNSC Extended data, extract speaker IDs,
perform a speaker-aware train/validation split, optionally filter by
audio duration, normalize transcripts, and save the results as parquet
files that any team script can load — transcription training, speaker-
classification training, and evaluation scripts alike.

Usage
-----
    # For transcription (full audio, no duration filter)
    python preprocess_mnsc_data.py

    # For classification (3-16 seconds as per group requirements)
    python preprocess_mnsc_data.py --min_duration 3.0 --max_duration 16.0

    # Custom configurations
    python preprocess_mnsc_data.py --max_samples 10000  # more data
    python preprocess_mnsc_data.py --max_samples 0      # all data
    python preprocess_mnsc_data.py --val_fraction 0.15 --seed 123

Outputs (written to <project root>/data/, one level above scripts/)
-----------------------------------------------------------
    data/preprocessed_train.parquet   -- training split
    data/preprocessed_val.parquet     -- validation split
    data/speaker_split_info.json      -- split metadata

Loading saved data from another script
--------------------------------------
    from preprocess_mnsc_data import load_preprocessed_data
    train_records, val_records, info = load_preprocessed_data()

Each record in the returned lists is a dict with keys:
    audio          : numpy float32 array, 16 kHz, mono
    sampling_rate  : int  (always 16000)
    text           : str  (normalised transcript)
    speaker_id     : str
    duration_sec   : float
"""

import argparse
import io
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
from datasets import Audio, load_dataset


# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent              # scripts/ holds code only
DATA_DIR     = PROJECT_ROOT / "data"

TRAIN_PARQUET = DATA_DIR / "preprocessed_train.parquet"
VAL_PARQUET   = DATA_DIR / "preprocessed_val.parquet"
SPLIT_INFO    = DATA_DIR / "speaker_split_info.json"


# ============================================================
# DATASET CONSTANTS
# ============================================================

DATASET_ID     = "AudioLLMs/Multitask-National-Speech-Corpus-v1-extend"
DATASET_CONFIG = "ASR-PART3-Train"
TARGET_SR      = 16_000


# ============================================================
# TRANSCRIPT NORMALIZATION
# ============================================================

def normalize_transcript(text: str) -> str:
    r"""
    Normalize a raw MNSC transcript.

    Rules (consistent with train_whisper_speaker_split.py and
    evaluate_whisper_speaker_split.py):

    - Remove <Speaker\d+>: labels
    - Unwrap [annotation] brackets  (keep inner text)
    - Unwrap {annotation} curly brackets  (keep inner text)
    - Unwrap (spoken content) parentheses  (keep inner text)
    - Unwrap #marker# hash markers  (keep inner text)
    - Collapse whitespace
    - Lowercase

    Singlish particles (lah, leh, loh, meh, sia, lor, hor) and local
    acronyms (HDB, MRT, NDP, ...) are preserved verbatim after lowercasing.
    """
    if not text:
        return ""

    # Remove speaker labels  e.g. <Speaker1>: hello
    text = re.sub(r"<Speaker\d+>:\s*", "", text, flags=re.IGNORECASE)

    # Unwrap square brackets  [lah] -> lah  (nested-safe loop)
    while re.search(r"\[[^\[\]]+\]", text):
        text = re.sub(r"\[([^\[\]]+)\]", r"\1", text)

    # Unwrap curly brackets  {term} -> term
    text = re.sub(r"\{([^{}\[\]]+)\}", r"\1", text)

    # Unwrap parentheses  (uh) -> uh
    text = re.sub(r"\(([^()]*)\)", r"\1", text)

    # Unwrap hash markers  #Sentosa# -> Sentosa
    text = re.sub(r"#([^#]+)#", r"\1", text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Lowercase
    return text.lower()


# ============================================================
# SPEAKER ID EXTRACTION
# ============================================================

def extract_speaker_id(row: dict) -> str:
    """
    Extract the primary speaker ID from an MNSC Extended row.

    Schema (from train_whisper_speaker_split.py investigation):
        row["other_attributes"]["speaker1"]["speaker_id"]
        row["other_attributes"]["speaker2"]["speaker_id"]

    Tries speaker1 first; falls back to speaker2; returns
    "unknown" if neither yields a non-empty string.
    """
    try:
        other = row.get("other_attributes") or {}
        if isinstance(other, str):
            other = json.loads(other)

        for key in ("speaker1", "speaker2"):
            block = other.get(key) or {}
            if isinstance(block, str):
                try:
                    block = json.loads(block)
                except Exception:
                    block = {}
            spk_id = block.get("speaker_id", None)
            if spk_id and str(spk_id).strip():
                return str(spk_id).strip()

        return "unknown"

    except Exception:
        return "unknown"


# ============================================================
# LOAD + COLLECT RECORDS FROM HF STREAMING DATASET
# ============================================================

def collect_records(
    max_samples:    Optional[int],
    min_duration:   Optional[float],
    max_duration:   Optional[float],
) -> Tuple[List[dict], dict]:
    """
    Stream the MNSC Extended dataset, decode audio, normalize
    transcripts, extract speaker IDs, optionally filter by
    duration, and return a list of record dicts.

    Parameters
    ----------
    max_samples  : Upper bound on collected records (None = unlimited).
    min_duration : Minimum audio duration in seconds (None = no minimum).
    max_duration : Maximum audio duration in seconds (None = no maximum).

    Returns
    -------
    records : list of dicts with keys
                  audio, sampling_rate, text, speaker_id, duration_sec
    stats   : dict with skipped-record counts for the metadata JSON
    """

    print("=" * 70)
    print("LOADING MNSC EXTENDED DATASET")
    print("=" * 70)
    print(f"Dataset : {DATASET_ID}")
    print(f"Config  : {DATASET_CONFIG}")
    print(f"Max samples : {max_samples if max_samples else 'all'}")
    print(f"Min duration: {min_duration} s")
    print(f"Max duration: {max_duration} s")
    print()
    print("Note: First run downloads the dataset from HuggingFace Hub.")
    print("      Subsequent runs reuse cached data.")
    print()

    dataset = load_dataset(
        DATASET_ID,
        DATASET_CONFIG,
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    # Keep audio as raw bytes -- we decode ourselves
    dataset = dataset.cast_column("context", Audio(decode=False))

    records:       List[dict] = []
    skipped_audio: int = 0
    skipped_text:  int = 0
    skipped_dur:   int = 0

    print("=" * 70)
    print("COLLECTING RECORDS")
    print("=" * 70)

    for row in dataset:

        # ---- audio ----
        context = row.get("context") or {}
        audio_bytes = (
            context.get("bytes") if isinstance(context, dict) else None
        )

        if not audio_bytes:
            skipped_audio += 1
            continue

        try:
            audio, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        except Exception as exc:
            print(f"  Audio decode error (skipping): {exc}")
            skipped_audio += 1
            continue

        # Stereo -> mono
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        if sr != TARGET_SR:
            print(f"  Unexpected sample rate {sr} Hz -- skipping")
            skipped_audio += 1
            continue

        duration_sec: float = len(audio) / sr

        # ---- duration filter ----
        if min_duration is not None and duration_sec < min_duration:
            skipped_dur += 1
            continue
        if max_duration is not None and duration_sec > max_duration:
            skipped_dur += 1
            continue

        # ---- transcript ----
        text = normalize_transcript(row.get("answer", ""))
        if not text:
            skipped_text += 1
            continue

        # ---- speaker ID ----
        speaker_id = extract_speaker_id(row)

        records.append({
            "audio":        audio,
            "sampling_rate": sr,
            "text":         text,
            "speaker_id":   speaker_id,
            "duration_sec": duration_sec,
        })

        if len(records) % 500 == 0:
            print(f"  Collected {len(records):,} records so far...")

        if max_samples and len(records) >= max_samples:
            break

    print()
    print(f"Total collected   : {len(records):,}")
    print(f"Skipped (audio)   : {skipped_audio:,}")
    print(f"Skipped (text)    : {skipped_text:,}")
    print(f"Skipped (duration): {skipped_dur:,}")

    stats = {
        "total_collected": len(records),
        "skipped_audio":   skipped_audio,
        "skipped_text":    skipped_text,
        "skipped_duration": skipped_dur,
    }
    return records, stats


# ============================================================
# SPEAKER-AWARE TRAIN / VALIDATION SPLIT
# ============================================================

def speaker_split(
    records:      List[dict],
    val_fraction: float,
    seed:         int,
) -> Tuple[List[dict], List[dict], dict]:
    """
    Split records so that speakers appear exclusively in either
    the training or the validation set (no leakage).

    Handles the edge case where all records have speaker_id == "unknown":
    in that case a random record-level split is used (with a warning),
    which avoids the crash seen in train_whisper_speaker_split.py.

    Parameters
    ----------
    records      : list of record dicts with a "speaker_id" key
    val_fraction : fraction of speakers to hold out (0 < val_fraction < 1)
    seed         : random seed

    Returns
    -------
    train_records : list[dict]
    val_records   : list[dict]
    split_info    : dict  (speaker assignments and sample counts)
    """

    print()
    print("=" * 70)
    print("SPEAKER-AWARE TRAIN / VALIDATION SPLIT")
    print("=" * 70)

    all_speakers = list({r["speaker_id"] for r in records})
    unique_speakers = [s for s in all_speakers if s != "unknown"]

    print(f"Total unique speaker IDs (including 'unknown'): {len(all_speakers)}")
    print(f"Named speakers (excluding 'unknown'): {len(unique_speakers)}")

    rng = random.Random(seed)

    # --- edge case: all records have no named speaker ---
    if len(unique_speakers) == 0:
        print()
        print(
            "WARNING: All speaker IDs are 'unknown'. "
            "Falling back to a random record-level split. "
            "Speaker leakage cannot be prevented without speaker labels."
        )
        shuffled = records[:]
        rng.shuffle(shuffled)
        n_val = max(1, int(len(shuffled) * val_fraction))
        val_records   = shuffled[:n_val]
        train_records = shuffled[n_val:]

        split_info = {
            "split_strategy":  "random_record_level_fallback",
            "reason":          "all speaker_ids were 'unknown'",
            "val_fraction":    val_fraction,
            "seed":            seed,
            "train_speakers":  [],
            "val_speakers":    [],
            "unknown_records": len(records),
            "train_samples":   len(train_records),
            "val_samples":     len(val_records),
        }

        print(f"Train samples (random split): {len(train_records):,}")
        print(f"Val   samples (random split): {len(val_records):,}")
        return train_records, val_records, split_info

    # --- normal path: speaker-level split ---
    rng.shuffle(unique_speakers)

    n_val_speakers = max(1, int(len(unique_speakers) * val_fraction))
    val_speakers   = set(unique_speakers[:n_val_speakers])
    train_speakers = set(unique_speakers[n_val_speakers:])

    # Records with unknown speaker go to train to avoid data loss
    train_records = [
        r for r in records
        if r["speaker_id"] in train_speakers or r["speaker_id"] == "unknown"
    ]
    val_records = [
        r for r in records
        if r["speaker_id"] in val_speakers
    ]

    print(f"Train speakers : {len(train_speakers):,}")
    print(f"Val   speakers : {len(val_speakers):,}")
    print(f"Train samples  : {len(train_records):,}")
    print(f"Val   samples  : {len(val_records):,}")

    if len(train_records) == 0:
        raise RuntimeError(
            "No training records after speaker split. "
            f"Named speakers found: {len(unique_speakers)}, "
            f"val fraction: {val_fraction}. "
            "Try reducing --val_fraction or increasing --max_samples."
        )
    if len(val_records) == 0:
        raise RuntimeError(
            "No validation records after speaker split. "
            f"Named speakers found: {len(unique_speakers)}, "
            f"val fraction: {val_fraction}. "
            "Try reducing --val_fraction or increasing --max_samples."
        )

    print()
    print("Sample val speakers (first 5):")
    for spk in sorted(val_speakers)[:5]:
        count = sum(1 for r in records if r["speaker_id"] == spk)
        print(f"  {spk}: {count} sample(s)")

    split_info = {
        "split_strategy": "speaker_level",
        "val_fraction":   val_fraction,
        "seed":           seed,
        "train_speakers": sorted(train_speakers),
        "val_speakers":   sorted(val_speakers),
        "unknown_records_in_train": sum(
            1 for r in train_records if r["speaker_id"] == "unknown"
        ),
        "train_samples":  len(train_records),
        "val_samples":    len(val_records),
    }
    return train_records, val_records, split_info


# ============================================================
# SAVE TO PARQUET
# ============================================================

def records_to_table(records: List[dict]) -> pa.Table:
    """
    Convert a list of record dicts to a PyArrow table.

    Audio arrays are stored as serialized bytes (float32 little-endian)
    alongside their length, so they survive a round-trip through parquet
    without requiring a special audio codec.
    """
    audio_bytes_list  = []
    sampling_rates    = []
    texts             = []
    speaker_ids       = []
    duration_secs     = []
    audio_lengths     = []

    for r in records:
        arr = r["audio"]
        # Store as raw float32 bytes
        audio_bytes_list.append(arr.astype(np.float32).tobytes())
        audio_lengths.append(len(arr))
        sampling_rates.append(r["sampling_rate"])
        texts.append(r["text"])
        speaker_ids.append(r["speaker_id"])
        duration_secs.append(r["duration_sec"])

    return pa.table({
        "audio_bytes":   pa.array(audio_bytes_list, type=pa.large_binary()),
        "audio_length":  pa.array(audio_lengths,    type=pa.int32()),
        "sampling_rate": pa.array(sampling_rates,   type=pa.int32()),
        "text":          pa.array(texts,             type=pa.string()),
        "speaker_id":    pa.array(speaker_ids,       type=pa.string()),
        "duration_sec":  pa.array(duration_secs,     type=pa.float32()),
    })


def save_parquet(records: List[dict], path: Path) -> None:
    table = records_to_table(records)
    pq.write_table(table, str(path))
    print(f"  Saved {len(records):,} records -> {path}")


# ============================================================
# LOAD PREPROCESSED DATA  (public API for other scripts)
# ============================================================

def load_preprocessed_data(
    train_path: Optional[str] = None,
    val_path:   Optional[str] = None,
    info_path:  Optional[str] = None,
) -> Tuple[List[dict], List[dict], dict]:
    """
    Load preprocessed train and validation records from parquet files.

    Parameters
    ----------
    train_path : path to preprocessed_train.parquet
                 (default: <project root>/data/preprocessed_train.parquet)
    val_path   : path to preprocessed_val.parquet
    info_path  : path to speaker_split_info.json

    Returns
    -------
    train_records : list[dict] with keys
                        audio (np.ndarray float32), sampling_rate (int),
                        text (str), speaker_id (str), duration_sec (float)
    val_records   : list[dict]  (same schema)
    info          : dict  (metadata from speaker_split_info.json)

    Example
    -------
    >>> from preprocess_mnsc_data import load_preprocessed_data
    >>> train, val, info = load_preprocessed_data()
    >>> print(f"Train: {len(train)}, Val: {len(val)}")
    """
    train_path = Path(train_path) if train_path else TRAIN_PARQUET
    val_path   = Path(val_path)   if val_path   else VAL_PARQUET
    info_path  = Path(info_path)  if info_path  else SPLIT_INFO

    def _load_parquet(p: Path) -> List[dict]:
        if not p.exists():
            raise FileNotFoundError(
                f"Preprocessed data not found at {p}.\n"
                "Run preprocess_mnsc_data.py first to generate it."
            )
        table  = pq.read_table(str(p))
        result = []
        rows   = table.to_pydict()
        n      = len(rows["text"])
        for i in range(n):
            length = rows["audio_length"][i]
            audio  = np.frombuffer(
                rows["audio_bytes"][i], dtype=np.float32
            )[:length].copy()
            result.append({
                "audio":         audio,
                "sampling_rate": int(rows["sampling_rate"][i]),
                "text":          rows["text"][i],
                "speaker_id":    rows["speaker_id"][i],
                "duration_sec":  float(rows["duration_sec"][i]),
            })
        return result

    train_records = _load_parquet(train_path)
    val_records   = _load_parquet(val_path)

    info: dict = {}
    if info_path.exists():
        with open(info_path, "r", encoding="utf-8") as fh:
            info = json.load(fh)

    return train_records, val_records, info


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Preprocess MNSC Extended dataset for shared team use."
    )
    p.add_argument(
        "--max_samples",
        type=int,
        default=5000,
        help=(
            "Maximum number of records to collect. "
            "Pass 0 to collect all available samples (may take a long time)."
        ),
    )
    p.add_argument(
        "--min_duration",
        type=float,
        default=None,
        help="Minimum audio duration in seconds (default: no minimum).",
    )
    p.add_argument(
        "--max_duration",
        type=float,
        default=None,
        help="Maximum audio duration in seconds (default: no maximum).",
    )
    p.add_argument(
        "--val_fraction",
        type=float,
        default=0.2,
        help="Fraction of unique speakers reserved for validation (default: 0.2).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    return p.parse_args()


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    args = parse_args()

    # Treat 0 as "no limit" for convenience
    max_samples = args.max_samples if args.max_samples > 0 else None

    # ---- collect ----
    records, collection_stats = collect_records(
        max_samples=max_samples,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
    )

    if not records:
        print("No records collected. Check dataset ID and internet connection.")
        sys.exit(1)

    # ---- split ----
    train_records, val_records, split_info = speaker_split(
        records,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )

    # ---- save ----
    print()
    print("=" * 70)
    print("SAVING PREPROCESSED DATA")
    print("=" * 70)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    save_parquet(train_records, TRAIN_PARQUET)
    save_parquet(val_records,   VAL_PARQUET)

    metadata = {
        "dataset_id":     DATASET_ID,
        "dataset_config": DATASET_CONFIG,
        "max_samples":    max_samples,
        "min_duration":   args.min_duration,
        "max_duration":   args.max_duration,
        "collection":     collection_stats,
        "split":          split_info,
    }

    with open(SPLIT_INFO, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)
    print(f"  Saved metadata  -> {SPLIT_INFO}")

    # ---- summary ----
    print()
    print("=" * 70)
    print("PREPROCESSING COMPLETE")
    print("=" * 70)
    print(f"Train : {len(train_records):,} samples")
    print(f"Val   : {len(val_records):,} samples")
    print()
    print("Load in other scripts with:")
    print("  from preprocess_mnsc_data import load_preprocessed_data")
    print("  train_records, val_records, info = load_preprocessed_data()")


if __name__ == "__main__":
    main()