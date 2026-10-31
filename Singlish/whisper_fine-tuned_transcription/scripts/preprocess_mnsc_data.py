"""
preprocess_mnsc_data.py

Shared data-preprocessing script for the MNSC Extended dataset.

Purpose
-------
Download (or reuse cached) MNSC Extended training shards, extract speaker
IDs, cap clips per speaker, perform a speaker-aware train/validation split,
optionally filter by audio duration, normalize transcripts, and save the
result as SHARDED parquet with FLAC-compressed audio under
<project root>/data/<DATA_NAME>/.

Only *-Train configs are ever read. ASR-PART1-Test / ASR-PART3-Test are the
fixed evaluation set and are refused by the code.

Usage (run from the project root)
-----
    # Default: ~15,000 PART3-Train clips, 10% of speakers held out for
    # validation, max 50 clips per speaker, seed 42 -> data/part3_15k/
    python scripts/preprocess_mnsc_data.py

    # Small real build for smoke tests (1 source shard, ~0.5 GB download)
    python scripts/preprocess_mnsc_data.py --data_name part3_smoke500 --max_samples 500

    # Custom configurations
    python scripts/preprocess_mnsc_data.py --data_name part3_30k --max_samples 30000
    python scripts/preprocess_mnsc_data.py --data_name part3_cls --min_duration 3 --max_duration 16
    python scripts/preprocess_mnsc_data.py --data_name mix10 --part1_fraction 0.1  # OFF by default

    # Held-out build (OFF by default): 40 PART3-Test conversations (and all their
    # voices, plus PART1-Test voices) kept out of train AND val, train/val split by
    # conversation, only the 30 cached source shards -> data/part3_15k_heldout/
    # (+ heldout_test_manifest.json, read by the eval/audit scripts)
    python scripts/preprocess_mnsc_data.py --data_name part3_15k_heldout --heldout_test_conversations 40 --max_source_shards 30

Outputs (one self-contained folder per dataset; nothing else is touched)
-------
    data/<DATA_NAME>/train/shard-00000.parquet ...
    data/<DATA_NAME>/val/shard-00000.parquet ...
    data/<DATA_NAME>/speaker_split_info.json   -- metadata + split
                                                  (split.train_speakers lists
                                                  the training speakers)

The legacy 4k files data/preprocessed_{train,val}.parquet and
data/speaker_split_info.json are only ever READ by this module
(load_preprocessed_data() with no data_name) and never written.

Loading saved data from another script
--------------------------------------
    from preprocess_mnsc_data import load_preprocessed_index, read_clip_audio
    train_items, val_items, info = load_preprocessed_index("part3_15k")
    audio = read_clip_audio(train_items[0])        # lazy, one clip at a time

    # Eager (decodes everything into RAM, ~1.7 MB per 28 s clip):
    from preprocess_mnsc_data import load_preprocessed_data
    train_records, val_records, info = load_preprocessed_data(data_name="part3_15k")

    # One legacy 4k file only, eager (e.g. the 990-clip legacy validation set):
    from preprocess_mnsc_data import load_legacy_parquet, VAL_PARQUET
    val_records = load_legacy_parquet(VAL_PARQUET)

Each eager record is a dict with keys:
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
import shutil
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf


# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent              # scripts/ holds code only
DATA_DIR     = PROJECT_ROOT / "data"

# Legacy (4k baseline) files: read-only, never written by this module.
TRAIN_PARQUET = DATA_DIR / "preprocessed_train.parquet"
VAL_PARQUET   = DATA_DIR / "preprocessed_val.parquet"
SPLIT_INFO    = DATA_DIR / "speaker_split_info.json"

INFO_NAME = "speaker_split_info.json"


# ============================================================
# DATASET CONSTANTS
# ============================================================

DATASET_ID     = "AudioLLMs/Multitask-National-Speech-Corpus-v1-extend"
DATASET_CONFIG = "ASR-PART3-Train"
PART1_CONFIG   = "ASR-PART1-Train"
TEST_CONFIG_RE = re.compile(r"-Test$")
TARGET_SR      = 16_000
FORMAT_VERSION = 2

DATA_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
DEFAULT_DATA_NAME = "part3_15k"

SOURCE_COLUMNS = ["context", "answer", "other_attributes"]
META_COLUMNS = [
    "clip_id", "source_config", "audio_length", "sampling_rate",
    "text", "speaker_id", "duration_sec",
]


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


def extract_speaker_id_any(row: dict) -> str:
    """
    extract_speaker_id() (unchanged, PART3 layout) plus a fallback for the
    ASR-PART1-Train layout, row["other_attributes"]["speaker"]["speaker_id"].
    For PART3 rows the result is identical to extract_speaker_id().
    """
    spk = extract_speaker_id(row)
    if spk != "unknown":
        return spk
    try:
        other = row.get("other_attributes") or {}
        if isinstance(other, str):
            other = json.loads(other)
        block = other.get("speaker") or {}
        if isinstance(block, str):
            block = json.loads(block)
        spk_id = block.get("speaker_id", None)
        if spk_id and str(spk_id).strip():
            return str(spk_id).strip()
    except Exception:
        pass
    return "unknown"


# ============================================================
# HELD-OUT TEST SELECTION (opt-in: --heldout_test_conversations N)
# ============================================================
#
# ASR-PART3-Test is cut from the same conversations as ASR-PART3-Train. A held-out
# build picks N PART3-Test conversations at runtime and removes every training row
# of those conversations, and every row with one of their voices (either role), plus
# any row with a PART1-Test voice. Only the other_attributes column of the test
# configs is read (never audio), and only to build this exclusion set.

HELDOUT_MANIFEST_NAME = "heldout_test_manifest.json"
PART3_TEST_CONFIG = "ASR-PART3-Test"
PART1_TEST_CONFIG = "ASR-PART1-Test"
HELDOUT_SELECTION_RULE = (
    "PART3-Test conversations whose voices occur in no other conversation (train or test); "
    "sorted, shuffled with random.Random(seed), first N"
)


def _as_dict(value) -> dict:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except Exception:
            return {}
    return value if isinstance(value, dict) else {}


def row_voices(attrs) -> List[str]:
    """Sorted unique voices of a row: speaker1, speaker2 (PART3) and speaker (PART1)."""
    attrs = _as_dict(attrs)
    out = set()
    for key in ("speaker1", "speaker2", "speaker"):
        spk = str(_as_dict(attrs.get(key)).get("speaker_id") or "").strip()
        if spk and spk != "unknown":
            out.add(spk)
    return sorted(out)


def row_conversation(attrs) -> str:
    """Stripped conversation_id, or '' (PART1 rows have none)."""
    return str(_as_dict(attrs).get("conversation_id") or "").strip()


def list_test_shards(config: str, source_dir: Optional[Path] = None) -> List[str]:
    """Like list_source_shards, but ONLY for *-Test configs (metadata reads)."""
    if not TEST_CONFIG_RE.search(config):
        raise ValueError(f"list_test_shards only reads *-Test configs, got {config}")
    if source_dir is not None:
        return [str(p) for p in sorted((Path(source_dir) / config).glob("*.parquet"))]
    from huggingface_hub import HfApi
    entries = HfApi().list_repo_tree(
        DATASET_ID, path_in_repo=config, repo_type="dataset"
    )
    return sorted(e.path for e in entries if e.path.endswith(".parquet"))


def read_attrs_column(paths: List[str], local: bool) -> List[dict]:
    """other_attributes of every row (no audio), in order across `paths`.

    Remote shards are read from the HF cache when present; otherwise only the
    other_attributes column is fetched over HfFileSystem (parquet column
    projection), so a metadata scan never downloads the ~0.5 GB audio shards."""
    out: List[dict] = []
    for shard in paths:
        if local:
            col = pq.read_table(shard, columns=["other_attributes"]).column(0)
        else:
            from huggingface_hub import try_to_load_from_cache
            cached = try_to_load_from_cache(DATASET_ID, shard, repo_type="dataset")
            if isinstance(cached, str):
                col = pq.read_table(cached, columns=["other_attributes"]).column(0)
            else:
                from huggingface_hub import HfFileSystem
                with HfFileSystem().open(f"datasets/{DATASET_ID}/{shard}", "rb") as fh:
                    col = pq.ParquetFile(fh).read(columns=["other_attributes"]).column(0)
        out.extend(_as_dict(a) for a in col.to_pylist())
    return out


def iter_train_attrs(config: str, source_dir: Optional[Path] = None,
                     max_shards: Optional[int] = None) -> Iterator[dict]:
    """other_attributes of a *-Train config (list_source_shards refuses *-Test)."""
    shards = list_source_shards(config, source_dir)
    if max_shards:
        shards = shards[:max_shards]
    if not shards:
        raise FileNotFoundError(f"No parquet shards found for {config}")
    for shard in shards:
        yield from read_attrs_column([shard], source_dir is not None)


def select_heldout_conversations(n: int, seed: int, source_dir: Optional[Path] = None,
                                 max_shards: Optional[int] = None) -> dict:
    """
    Choose n PART3-Test conversations to hold out (runtime, never hardcoded).

    Returns the manifest dict written to heldout_test_manifest.json, plus the
    in-memory sets '_conv_set', '_voice_set', '_part1_voices' (not serialised).
    """
    local = source_dir is not None
    p3_paths = list_test_shards(PART3_TEST_CONFIG, source_dir)
    p1_paths = list_test_shards(PART1_TEST_CONFIG, source_dir)
    if not p3_paths or not p1_paths:
        raise ValueError(f"No parquet shards found for {PART3_TEST_CONFIG} / {PART1_TEST_CONFIG}")
    p3_rows = read_attrs_column(p3_paths, local)
    part1_voices = set()
    for a in read_attrs_column(p1_paths, local):
        part1_voices.update(row_voices(a))

    conv_voices: Dict[str, set] = {}
    voice_convs: Dict[str, set] = {}

    def add(attrs):
        conv = row_conversation(attrs)
        if not conv:
            return
        for v in row_voices(attrs):
            conv_voices.setdefault(conv, set()).add(v)
            voice_convs.setdefault(v, set()).add(conv)
        conv_voices.setdefault(conv, set())

    n_shards = len(list_source_shards(DATASET_CONFIG, source_dir))
    if max_shards:
        n_shards = min(n_shards, max_shards)
    for attrs in iter_train_attrs(DATASET_CONFIG, source_dir, max_shards):
        add(attrs)
    test_convs = set()
    for attrs in p3_rows:
        add(attrs)
        if row_conversation(attrs):
            test_convs.add(row_conversation(attrs))

    candidates = sorted(c for c in test_convs
                        if all(voice_convs[v] == {c} for v in conv_voices[c]))
    if n > len(candidates):
        raise ValueError(f"--heldout_test_conversations {n} > {len(candidates)} candidate "
                         f"PART3-Test conversations")
    shuffled = candidates[:]
    random.Random(seed).shuffle(shuffled)
    chosen = set(shuffled[:n])
    held_voices = set()
    for c in chosen:
        held_voices |= conv_voices[c]
    rows = [i for i, a in enumerate(p3_rows) if row_conversation(a) in chosen]
    return {
        "format_version": 1,
        "dataset_id": DATASET_ID,
        "seed": seed,
        "n_requested": n,
        "n_candidates": len(candidates),
        "n_part3_test_conversations": len(test_convs),
        "selection_rule": HELDOUT_SELECTION_RULE,
        "train_shards_scanned": n_shards,
        "conversations": sorted(chosen),
        "voices": sorted(held_voices),
        "part3_test_row_indices": rows,
        "n_part3_test_rows": len(rows),
        "part1_test_voices_count": len(part1_voices),
        "_conv_set": chosen,
        "_voice_set": held_voices,
        "_part1_voices": part1_voices,
    }


# ============================================================
# AUDIO CODEC (FLAC, 16 kHz mono, PCM_16)
# ============================================================

def encode_flac(audio: np.ndarray, sr: int = TARGET_SR) -> bytes:
    """Encode a mono int16 or float32 waveform as FLAC PCM_16 bytes."""
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="FLAC", subtype="PCM_16")
    return buf.getvalue()


def decode_flac(data: bytes) -> Tuple[np.ndarray, int]:
    """Decode FLAC bytes to (float32 mono array, sampling rate)."""
    audio, sr = sf.read(io.BytesIO(data), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1).astype(np.float32)
    return audio, sr


# ============================================================
# OUTPUT LOCATION
# ============================================================

def dataset_dir(data_name: str, data_root: Path = DATA_DIR) -> Path:
    """
    Validated output folder data_root/<data_name>. Always a direct child of
    data_root; 'legacy' is reserved for the old 4k files.
    """
    if not data_name or not DATA_NAME_RE.match(data_name) or data_name.lower() == "legacy":
        raise ValueError(
            f"Invalid --data_name {data_name!r}: use letters, digits, '_', '-', '.' "
            "(must not start with '.' or be 'legacy')."
        )
    root = Path(data_root).resolve()
    out = (root / data_name).resolve()
    if out.parent != root:
        raise ValueError(f"--data_name {data_name!r} escapes {root}")
    return out


# ============================================================
# SHARDED PARQUET WRITER
# ============================================================

def _peak_rss_mb() -> Optional[float]:
    try:
        import psutil
        mi = psutil.Process().memory_info()
        peak = getattr(mi, "peak_wset", None) or mi.rss
        return round(peak / 2**20, 1)
    except Exception:
        return None


class ShardWriter:
    """
    Buffers at most one shard of rows and writes
    <out_dir>/<prefix>-NNNNN.parquet files with small row groups so a
    single clip can be read without loading the whole shard.
    """

    def __init__(self, out_dir: Path, prefix: str,
                 rows_per_shard: int = 512, row_group_size: int = 8):
        self.out_dir = Path(out_dir)
        self.prefix = prefix
        self.rows_per_shard = int(rows_per_shard)
        self.row_group_size = int(row_group_size)
        self.files: List[str] = []
        self.n_rows = 0
        self._buf: Dict[str, list] = self._empty()
        self.out_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _empty() -> Dict[str, list]:
        return {c: [] for c in META_COLUMNS + ["audio_flac"]}

    def _current_name(self) -> str:
        return f"{self.prefix}-{len(self.files):05d}.parquet"

    def add(self, record: dict) -> Tuple[str, int]:
        name = self._current_name()
        idx = len(self._buf["clip_id"])
        for c in self._buf:
            self._buf[c].append(record[c])
        self.n_rows += 1
        if idx + 1 >= self.rows_per_shard:
            self._flush()
        return name, idx

    def _flush(self) -> None:
        if not self._buf["clip_id"]:
            return
        b = self._buf
        table = pa.table({
            "clip_id":       pa.array(b["clip_id"],       type=pa.string()),
            "source_config": pa.array(b["source_config"], type=pa.string()),
            "audio_flac":    pa.array(b["audio_flac"],    type=pa.large_binary()),
            "audio_length":  pa.array(b["audio_length"],  type=pa.int32()),
            "sampling_rate": pa.array(b["sampling_rate"], type=pa.int32()),
            "text":          pa.array(b["text"],          type=pa.string()),
            "speaker_id":    pa.array(b["speaker_id"],    type=pa.string()),
            "duration_sec":  pa.array(b["duration_sec"],  type=pa.float32()),
        })
        compression = {c: "snappy" for c in META_COLUMNS}
        compression["audio_flac"] = "NONE"
        name = self._current_name()
        pq.write_table(
            table,
            str(self.out_dir / name),
            row_group_size=self.row_group_size,
            compression=compression,
            use_dictionary=list(META_COLUMNS),
        )
        self.files.append(name)
        self._buf = self._empty()

    def close(self) -> List[str]:
        self._flush()
        return list(self.files)


# ============================================================
# SOURCE SHARDS (HF Hub or a local mirror)
# ============================================================

def _refuse_test(config: str) -> None:
    if TEST_CONFIG_RE.search(config):
        raise ValueError(
            f"{config} is part of the fixed evaluation set and must never be "
            "used for training/validation."
        )


def list_source_shards(config: str, source_dir: Optional[Path] = None) -> List[str]:
    """
    Sorted parquet shard names (relative to the repo, e.g.
    'ASR-PART3-Train/train-00000-of-00171.parquet') or, with source_dir,
    absolute local paths.
    """
    _refuse_test(config)
    if source_dir is not None:
        return [str(p) for p in sorted((Path(source_dir) / config).glob("*.parquet"))]
    from huggingface_hub import HfApi
    entries = HfApi().list_repo_tree(
        DATASET_ID, path_in_repo=config, repo_type="dataset"
    )
    return sorted(e.path for e in entries if e.path.endswith(".parquet"))


def iter_source_rows(
    config: str,
    source_dir: Optional[Path] = None,
    timing: Optional[dict] = None,
    max_shards: Optional[int] = None,
) -> Iterator[Tuple[int, int, dict]]:
    """
    Yield (shard_idx, row_idx_in_shard, row) in shard order. Hub shards are
    downloaded lazily (only when the next shard is needed) into the HF cache,
    so re-runs reuse them. max_shards limits reading to the first K shards.
    """
    timing = timing if timing is not None else {}
    timing.setdefault("download_s", 0.0)
    timing.setdefault("shards_used", {})
    shards = list_source_shards(config, source_dir)
    if max_shards:
        shards = shards[:max_shards]
    if not shards:
        raise FileNotFoundError(f"No parquet shards found for {config}")

    for shard_idx, shard in enumerate(shards):
        if source_dir is not None:
            path = shard
        else:
            from huggingface_hub import hf_hub_download
            t0 = time.time()
            path = hf_hub_download(DATASET_ID, shard, repo_type="dataset")
            dt = time.time() - t0
            timing["download_s"] += dt
            size_mb = os.path.getsize(path) / 1e6
            print(f"  [{config}] shard {shard_idx + 1}/{len(shards)} "
                  f"{os.path.basename(shard)}: {size_mb:,.0f} MB ready in {dt:.1f} s")
        timing["shards_used"][config] = shard_idx + 1

        pf = pq.ParquetFile(path)
        row_idx = 0
        for batch in pf.iter_batches(batch_size=32, columns=SOURCE_COLUMNS):
            for row in batch.to_pylist():
                yield shard_idx, row_idx, row
                row_idx += 1


# ============================================================
# COLLECT RECORDS (pass 1: FLAC to staging, metadata in RAM)
# ============================================================

def collect_records(
    max_samples:           Optional[int],
    min_duration:          Optional[float],
    max_duration:          Optional[float],
    max_clips_per_speaker: int,
    part1_fraction:        float,
    staging:               ShardWriter,
    source_dir:            Optional[Path] = None,
    exclusion:             Optional[dict] = None,
    keep_group_keys:       bool = False,
    max_source_shards:     Optional[int] = None,
) -> Tuple[List[dict], dict]:
    """
    Read source rows, decode audio, filter by duration, normalize
    transcripts, extract speaker IDs, apply the per-speaker cap, encode FLAC
    and append to `staging`. Only light metadata is kept in RAM.

    Returns
    -------
    meta  : list of dicts with keys clip_id, source_config, speaker_id,
            text, duration_sec, sampling_rate, audio_length,
            stage_shard, stage_row   (no audio)
    stats : collection statistics for the metadata JSON

    exclusion (held-out builds only): {"conversations", "voices", "part1_voices"}
    sets; a row is dropped BEFORE the speaker cap if its conversation is held
    out, or any of its voices (either role) is a held-out or PART1-Test voice.
    keep_group_keys adds conversation_id / voices to the in-RAM metadata (not
    to the parquet columns) for conversation_split().
    """
    if part1_fraction > 0:
        if not max_samples:
            raise ValueError("--part1_fraction requires --max_samples > 0")
        n1 = int(round(max_samples * part1_fraction))
        quotas = [(DATASET_CONFIG, max_samples - n1), (PART1_CONFIG, n1)]
    else:
        quotas = [(DATASET_CONFIG, max_samples)]
    quotas = [(c, q) for c, q in quotas if q is None or q > 0]

    print("=" * 70)
    print("COLLECTING RECORDS")
    print("=" * 70)
    print(f"Dataset              : {DATASET_ID}")
    print(f"Configs / quotas     : {quotas}")
    print(f"Max clips per speaker: {max_clips_per_speaker or 'no cap'}")
    print(f"Min / max duration   : {min_duration} / {max_duration} s")
    print(f"Source               : {source_dir or 'HuggingFace Hub (cached after first download)'}")
    print()

    meta: List[dict] = []
    per_speaker: Dict[str, int] = {}
    skipped = {"audio": 0, "text": 0, "duration": 0, "speaker_cap": 0}
    excl_convs: set = set()
    excl_voices: set = set()
    if exclusion is not None:
        skipped.update({"heldout_conversation": 0, "heldout_voice": 0, "part1_test_voice": 0})
    per_config: Dict[str, int] = {}
    timing: dict = {"download_s": 0.0, "shards_used": {}}
    t_start = time.time()

    for config, quota in quotas:
        _refuse_test(config)
        got = 0
        if max_source_shards:
            rows = iter_source_rows(config, source_dir, timing, max_source_shards)
        else:
            rows = iter_source_rows(config, source_dir, timing)
        try:
            for shard_idx, row_idx, row in rows:
                # ---- speaker ID + cap (before audio decode) ----
                speaker_id = extract_speaker_id_any(row)
                if exclusion is not None or keep_group_keys:
                    row_attrs = row.get("other_attributes")
                    conv = row_conversation(row_attrs)
                    row_vs = row_voices(row_attrs)
                if exclusion is not None:
                    reason = None
                    if conv and conv in exclusion["conversations"]:
                        reason = "heldout_conversation"
                    elif any(v in exclusion["voices"] for v in row_vs):
                        reason = "heldout_voice"
                    elif any(v in exclusion["part1_voices"] for v in row_vs):
                        reason = "part1_test_voice"
                    if reason:
                        skipped[reason] += 1
                        if conv:
                            excl_convs.add(conv)
                        excl_voices.update(row_vs)
                        continue
                if max_clips_per_speaker and per_speaker.get(speaker_id, 0) >= max_clips_per_speaker:
                    skipped["speaker_cap"] += 1
                    continue

                # ---- audio ----
                context = row.get("context") or {}
                audio_bytes = context.get("bytes") if isinstance(context, dict) else None
                if not audio_bytes:
                    skipped["audio"] += 1
                    continue
                try:
                    audio, sr = sf.read(io.BytesIO(audio_bytes), dtype="int16")
                except Exception as exc:
                    print(f"  Audio decode error (skipping): {exc}")
                    skipped["audio"] += 1
                    continue
                if audio.ndim > 1:  # stereo -> mono
                    audio = np.round(audio.astype(np.float32).mean(axis=1)).astype(np.int16)
                if sr != TARGET_SR:
                    print(f"  Unexpected sample rate {sr} Hz -- skipping")
                    skipped["audio"] += 1
                    continue

                duration_sec = len(audio) / sr

                # ---- duration filter ----
                if min_duration is not None and duration_sec < min_duration:
                    skipped["duration"] += 1
                    continue
                if max_duration is not None and duration_sec > max_duration:
                    skipped["duration"] += 1
                    continue

                # ---- transcript ----
                text = normalize_transcript(row.get("answer", ""))
                if not text:
                    skipped["text"] += 1
                    continue

                clip_id = f"{config}/{shard_idx:05d}:{row_idx}"
                rec = {
                    "clip_id":       clip_id,
                    "source_config": config,
                    "audio_flac":    encode_flac(audio, sr),
                    "audio_length":  int(len(audio)),
                    "sampling_rate": int(sr),
                    "text":          text,
                    "speaker_id":    speaker_id,
                    "duration_sec":  float(duration_sec),
                }
                stage_shard, stage_row = staging.add(rec)
                del rec["audio_flac"]
                rec["stage_shard"] = stage_shard
                rec["stage_row"] = stage_row
                if keep_group_keys:
                    rec["conversation_id"] = conv
                    rec["voices"] = row_vs
                meta.append(rec)
                per_speaker[speaker_id] = per_speaker.get(speaker_id, 0) + 1
                got += 1

                if len(meta) % 500 == 0:
                    el = time.time() - t_start
                    print(f"  Collected {len(meta):,} records ({el:,.0f} s elapsed, "
                          f"peak RSS {_peak_rss_mb()} MB)")

                if quota and got >= quota:
                    break
        finally:
            rows.close()
        per_config[config] = got
        if quota and got < quota:
            print(f"  WARNING: {config} exhausted with {got:,} < {quota:,} clips")

    staging.close()
    total_s = time.time() - t_start
    hours = sum(r["duration_sec"] for r in meta) / 3600

    print()
    print(f"Total collected       : {len(meta):,}  ({hours:.2f} h)")
    print(f"Skipped (audio)       : {skipped['audio']:,}")
    print(f"Skipped (text)        : {skipped['text']:,}")
    print(f"Skipped (duration)    : {skipped['duration']:,}")
    print(f"Skipped (speaker cap) : {skipped['speaker_cap']:,}")
    if exclusion is not None:
        print(f"Excluded (held-out conversation): {skipped['heldout_conversation']:,}")
        print(f"Excluded (held-out voice)       : {skipped['heldout_voice']:,}")
        print(f"Excluded (PART1-Test voice)     : {skipped['part1_test_voice']:,}")
        print(f"Excluded train conversations    : {len(excl_convs):,}")
        print(f"Excluded train voices           : {len(excl_voices):,}")
    print(f"Download time         : {timing['download_s']:,.1f} s")

    stats = {
        "total_collected":     len(meta),
        "total_hours":         round(hours, 3),
        "skipped_audio":       skipped["audio"],
        "skipped_text":        skipped["text"],
        "skipped_duration":    skipped["duration"],
        "skipped_speaker_cap": skipped["speaker_cap"],
        "per_config":          per_config,
        "source_shards_used":  timing["shards_used"],
        "download_s":          round(timing["download_s"], 1),
        "process_s":           round(total_s - timing["download_s"], 1),
    }
    if exclusion is not None:
        stats.update({
            "skipped_heldout_conversation": skipped["heldout_conversation"],
            "skipped_heldout_voice":        skipped["heldout_voice"],
            "skipped_part1_test_voice":     skipped["part1_test_voice"],
            "excluded_train_conversations": len(excl_convs),
            "excluded_train_voices":        len(excl_voices),
        })
    return meta, stats


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

    # sorted(): set order depends on PYTHONHASHSEED; without it the split
    # would differ between runs despite the fixed seed.
    all_speakers = sorted({r["speaker_id"] for r in records})
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


def conversation_split(
    records:      List[dict],
    val_fraction: float,
    seed:         int,
) -> Tuple[List[dict], List[dict], dict]:
    """
    Split by connected components of the conversation/voice graph, so no
    conversation and no voice (speaker1 or speaker2, either role) is in both
    train and val. Records need "conversation_id" and "voices" keys
    (collect_records(..., keep_group_keys=True)).

    Components are sorted by their smallest node, shuffled with
    random.Random(seed), and the first max(1, int(n * val_fraction)) go to val.
    """
    print()
    print("=" * 70)
    print("CONVERSATION-LEVEL TRAIN / VALIDATION SPLIT")
    print("=" * 70)

    parent: Dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    rec_nodes: List[List[str]] = []
    for r in records:
        nodes = [f"v:{v}" for v in r.get("voices") or []]
        if r.get("conversation_id"):
            nodes.insert(0, f"c:{r['conversation_id']}")
        if not nodes:
            nodes = [f"r:{r['clip_id']}"]
        for n in nodes:
            find(n)
        for n in nodes[1:]:
            union(nodes[0], n)
        rec_nodes.append(nodes)

    members: Dict[str, List[str]] = {}
    for n in parent:
        members.setdefault(find(n), []).append(n)
    comps = sorted(members.values(), key=min)
    comp_of = {n: i for i, nodes in enumerate(comps) for n in nodes}
    order = list(range(len(comps)))
    random.Random(seed).shuffle(order)
    n_val = max(1, int(len(comps) * val_fraction))
    val_comps = set(order[:n_val])
    if len(comps) < 2:
        raise RuntimeError(
            f"Only {len(comps)} conversation component(s); cannot split train/val. "
            "Try increasing --max_samples."
        )

    train_records, val_records = [], []
    comp_clips: Dict[int, int] = {}
    for r, nodes in zip(records, rec_nodes):
        c = comp_of[nodes[0]]
        comp_clips[c] = comp_clips.get(c, 0) + 1
        (val_records if c in val_comps else train_records).append(r)
    if not train_records or not val_records:
        raise RuntimeError(
            "Empty train or validation split after conversation split. "
            f"Components: {len(comps)}, val fraction: {val_fraction}. "
            "Try changing --val_fraction or increasing --max_samples."
        )

    def collect(recs, key):
        out = set()
        for r in recs:
            if key == "voices":
                out.update(r.get("voices") or [])
            elif r.get(key):
                out.add(r[key])
        return out

    train_voices, val_voices = collect(train_records, "voices"), collect(val_records, "voices")
    train_convs = collect(train_records, "conversation_id")
    val_convs = collect(val_records, "conversation_id")
    n_multi = sum(1 for nodes in comps if sum(n.startswith("c:") for n in nodes) > 1)

    print(f"Components             : {len(comps):,} "
          f"({n_multi} with more than one conversation)")
    print(f"Train components       : {len(comps) - n_val:,}")
    print(f"Val   components       : {n_val:,}")
    print(f"Train conversations    : {len(train_convs):,}   voices: {len(train_voices):,}")
    print(f"Val   conversations    : {len(val_convs):,}   voices: {len(val_voices):,}")
    print(f"Train samples          : {len(train_records):,}")
    print(f"Val   samples          : {len(val_records):,}")

    split_info = {
        "split_strategy": "conversation_component",
        "val_fraction":   val_fraction,
        "seed":           seed,
        "train_speakers": sorted({r["speaker_id"] for r in train_records} - {"unknown"}),
        "val_speakers":   sorted({r["speaker_id"] for r in val_records}),
        "train_voices":   sorted(train_voices),
        "val_voices":     sorted(val_voices),
        "train_conversations": sorted(train_convs),
        "val_conversations":   sorted(val_convs),
        "n_components":       len(comps),
        "n_train_components": len(comps) - n_val,
        "n_val_components":   n_val,
        "n_multi_conversation_components": n_multi,
        "largest_component_clips": max(comp_clips.values()),
        "train_samples":  len(train_records),
        "val_samples":    len(val_records),
    }
    return train_records, val_records, split_info


# ============================================================
# PASS 2: ROUTE STAGED CLIPS INTO train/ AND val/ SHARDS
# ============================================================

def write_split(
    meta_train:     List[dict],
    meta_val:       List[dict],
    staging_dir:    Path,
    out_dir:        Path,
    rows_per_shard: int,
    row_group_size: int,
) -> Dict[str, List[str]]:
    """Read staging shards one row group at a time and route rows by clip_id."""
    route = {r["clip_id"]: "train" for r in meta_train}
    route.update({r["clip_id"]: "val" for r in meta_val})
    writers = {
        "train": ShardWriter(out_dir / "train", "shard", rows_per_shard, row_group_size),
        "val":   ShardWriter(out_dir / "val",   "shard", rows_per_shard, row_group_size),
    }
    for stage_file in sorted(Path(staging_dir).glob("*.parquet")):
        pf = pq.ParquetFile(str(stage_file))
        for rg in range(pf.metadata.num_row_groups):
            for row in pf.read_row_group(rg).to_pylist():
                dest = route.get(row["clip_id"])
                if dest is not None:
                    writers[dest].add(row)
        del pf
    layout = {"train": writers["train"].close(), "val": writers["val"].close()}
    shutil.rmtree(staging_dir)
    return layout


# ============================================================
# SPEAKER DISTRIBUTION
# ============================================================

def speaker_distribution(meta: List[dict], cap: int) -> dict:
    counts: Dict[str, int] = {}
    for r in meta:
        counts[r["speaker_id"]] = counts.get(r["speaker_id"], 0) + 1
    if not counts:
        return {"n_speakers": 0}
    vals = sorted(counts.values())
    p90 = vals[min(len(vals) - 1, int(round(0.9 * (len(vals) - 1))))]
    top = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:10]
    return {
        "n_speakers": len(vals),
        "min":    vals[0],
        "median": statistics.median(vals),
        "mean":   round(statistics.mean(vals), 2),
        "p90":    p90,
        "max":    vals[-1],
        "n_at_cap": sum(1 for v in vals if cap and v >= cap),
        "top10":  [{"speaker_id": k, "clips": v} for k, v in top],
    }


# ============================================================
# LOAD PREPROCESSED DATA  (public API for other scripts)
# ============================================================

def _read_info(out_dir: Path) -> dict:
    info_path = out_dir / INFO_NAME
    if not info_path.exists():
        raise FileNotFoundError(
            f"Preprocessed dataset not found at {out_dir}.\n"
            f"Run: python scripts/preprocess_mnsc_data.py --data_name {out_dir.name}"
        )
    with open(info_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_preprocessed_index(
    data_name: str = DEFAULT_DATA_NAME,
    data_root: Path = DATA_DIR,
) -> Tuple[List[dict], List[dict], dict]:
    """
    Lightweight per-clip index (NO audio) for lazy loading.

    Each item: shard (absolute path str), row_group, row_in_group, clip_id,
    text, speaker_id, duration_sec, sampling_rate, audio_length.
    Use read_clip_audio(item) to decode one clip.
    """
    out_dir = dataset_dir(data_name, data_root)
    info = _read_info(out_dir)
    layout = info["layout"]
    result: Dict[str, List[dict]] = {}
    for split in ("train", "val"):
        items: List[dict] = []
        for name in layout[f"{split}_shards"]:
            path = out_dir / split / name
            pf = pq.ParquetFile(str(path))
            cols = pf.read(columns=["clip_id", "text", "speaker_id", "duration_sec",
                                    "sampling_rate", "audio_length"]).to_pydict()
            positions = []
            for rg in range(pf.metadata.num_row_groups):
                positions += [(rg, j) for j in range(pf.metadata.row_group(rg).num_rows)]
            for i, (rg, j) in enumerate(positions):
                items.append({
                    "shard":         str(path),
                    "row_group":     rg,
                    "row_in_group":  j,
                    "clip_id":       cols["clip_id"][i],
                    "text":          cols["text"][i],
                    "speaker_id":    cols["speaker_id"][i],
                    "duration_sec":  float(cols["duration_sec"][i]),
                    "sampling_rate": int(cols["sampling_rate"][i]),
                    "audio_length":  int(cols["audio_length"][i]),
                })
        result[split] = items
    return result["train"], result["val"], info


def read_clip_audio(item: dict, cache: Optional[dict] = None) -> np.ndarray:
    """Decode one clip: reads a single row group (audio column only)."""
    path = item["shard"]
    pf = cache.get(path) if cache is not None else None
    if pf is None:
        pf = pq.ParquetFile(path)
        if cache is not None:
            if len(cache) >= 8:
                cache.pop(next(iter(cache)))
            cache[path] = pf
    col = pf.read_row_group(item["row_group"], columns=["audio_flac"]).column("audio_flac")
    audio, _ = decode_flac(col[item["row_in_group"]].as_py())
    return audio


def iter_preprocessed_records(
    data_name: str,
    split: str,
    data_root: Path = DATA_DIR,
) -> Iterator[dict]:
    """Stream decoded records (legacy schema) one row group at a time."""
    out_dir = dataset_dir(data_name, data_root)
    info = _read_info(out_dir)
    for name in info["layout"][f"{split}_shards"]:
        pf = pq.ParquetFile(str(out_dir / split / name))
        for rg in range(pf.metadata.num_row_groups):
            for row in pf.read_row_group(rg).to_pylist():
                audio, sr = decode_flac(row["audio_flac"])
                yield {
                    "audio":         audio,
                    "sampling_rate": int(sr),
                    "text":          row["text"],
                    "speaker_id":    row["speaker_id"],
                    "duration_sec":  float(row["duration_sec"]),
                }


def load_legacy_parquet(path) -> List[dict]:
    """Eagerly load ONE legacy float32 parquet file (e.g. VAL_PARQUET) into records."""
    p = Path(path)
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


def load_preprocessed_data(
    train_path: Optional[str] = None,
    val_path:   Optional[str] = None,
    info_path:  Optional[str] = None,
    data_name:  Optional[str] = None,
    data_root:  Path = DATA_DIR,
) -> Tuple[List[dict], List[dict], dict]:
    """
    Load preprocessed train and validation records EAGERLY (all audio in RAM).

    data_name=None -> legacy float32 files (data/preprocessed_*.parquet,
    the 4k baseline), exactly as before. data_name='part3_15k' etc. -> the
    sharded FLAC dataset, fully decoded (~1.7 MB RAM per 28 s clip; prefer
    load_preprocessed_index() + read_clip_audio() for training).

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
    if data_name is not None:
        out_dir = dataset_dir(data_name, data_root)
        info = _read_info(out_dir)
        n = info["split"]["train_samples"] + info["split"]["val_samples"]
        print(f"WARNING: eager load of {n:,} clips decodes all audio into RAM "
              f"(~{n * 1.7 / 1024:.1f} GB for ~28 s clips).")
        train_records = list(iter_preprocessed_records(data_name, "train", data_root))
        val_records = list(iter_preprocessed_records(data_name, "val", data_root))
        return train_records, val_records, info

    train_path = Path(train_path) if train_path else TRAIN_PARQUET
    val_path   = Path(val_path)   if val_path   else VAL_PARQUET
    info_path  = Path(info_path)  if info_path  else SPLIT_INFO

    train_records = load_legacy_parquet(train_path)
    val_records   = load_legacy_parquet(val_path)

    info: dict = {}
    if info_path.exists():
        with open(info_path, "r", encoding="utf-8") as fh:
            info = json.load(fh)

    return train_records, val_records, info


# ============================================================
# CLI
# ============================================================

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Preprocess MNSC Extended (ASR-PART3-Train) into data/<DATA_NAME>/."
    )
    p.add_argument("--data_name", default=DEFAULT_DATA_NAME,
                   help="Output folder name under data/ (default: part3_15k).")
    p.add_argument("--data_root", default=str(DATA_DIR),
                   help="Parent folder for datasets (default: <project>/data).")
    p.add_argument("--source_dir", default=None,
                   help="Local mirror with <config>/*.parquet instead of the HF Hub (tests).")
    p.add_argument("--max_samples", type=int, default=15000,
                   help="Clips to collect (default 15000). 0 = all available.")
    p.add_argument("--min_duration", type=float, default=None,
                   help="Minimum audio duration in seconds (default: no minimum).")
    p.add_argument("--max_duration", type=float, default=None,
                   help="Maximum audio duration in seconds (default: no maximum).")
    p.add_argument("--val_fraction", type=float, default=0.1,
                   help="Fraction of unique speakers held out for validation (default 0.1).")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for the speaker split (default 42).")
    p.add_argument("--max_clips_per_speaker", type=int, default=50,
                   help="Cap clips per speaker so a few speakers don't dominate "
                        "(default 50, ~2.3x the mean at 15k clips; 0 = no cap).")
    p.add_argument("--part1_fraction", type=float, default=0.0,
                   help="OFF by default. Share of --max_samples taken from ASR-PART1-Train "
                        "(e.g. 0.1). Keep 0 for comparability with the 4k baseline.")
    p.add_argument("--rows_per_shard", type=int, default=512,
                   help="Clips per output parquet shard (default 512).")
    p.add_argument("--row_group_size", type=int, default=8,
                   help="Clips per parquet row group, the lazy-read unit (default 8).")
    p.add_argument("--overwrite", action="store_true",
                   help="Replace an existing data/<DATA_NAME>/ built by this script.")
    p.add_argument("--heldout_test_conversations", type=int, default=0,
                   help="OFF by default (0). Hold out N ASR-PART3-Test conversations: drop every "
                        "training row of those conversations or with one of their voices (either "
                        "role) or a PART1-Test voice, split train/val by conversation, and write "
                        f"{HELDOUT_MANIFEST_NAME}. Chosen at runtime with --seed.")
    p.add_argument("--split_by_conversation", action="store_true",
                   help="Split train/val by conversation/voice components instead of speaker1 id "
                        "(implied by --heldout_test_conversations).")
    p.add_argument("--max_source_shards", type=int, default=0,
                   help="Read only the first K source shards per config (default 0 = all).")
    args = p.parse_args(argv)
    if not 0.0 <= args.part1_fraction < 1.0:
        p.error("--part1_fraction must be in [0, 1)")
    if not 0.0 < args.val_fraction < 1.0:
        p.error("--val_fraction must be in (0, 1)")
    if args.heldout_test_conversations < 0 or args.max_source_shards < 0:
        p.error("--heldout_test_conversations/--max_source_shards must be >= 0")
    if (args.heldout_test_conversations or args.split_by_conversation) and args.part1_fraction > 0:
        p.error("--heldout_test_conversations/--split_by_conversation cannot be combined with "
                "--part1_fraction (PART1-Train rows have no conversation ids)")
    return args


def _prepare_out_dir(out_dir: Path, overwrite: bool) -> Optional[str]:
    """Return an error message if out_dir cannot be (re)used."""
    if not out_dir.exists() or not any(out_dir.iterdir()):
        return None
    if not overwrite:
        return (f"{out_dir} already exists and is not empty. "
                "Pick another --data_name or pass --overwrite.")
    info_path = out_dir / INFO_NAME
    contents = {p.name for p in out_dir.iterdir()}
    ours = False
    if info_path.exists():
        try:
            ours = json.loads(info_path.read_text(encoding="utf-8")).get("format_version") == FORMAT_VERSION
        except Exception:
            ours = False
    elif contents == {"_staging"}:
        ours = True  # leftover of an interrupted run
    if not ours:
        return (f"Refusing to delete {out_dir}: it was not created by this script "
                f"(no {INFO_NAME} with format_version {FORMAT_VERSION}).")
    shutil.rmtree(out_dir)
    return None


# ============================================================
# MAIN
# ============================================================

def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    t0 = time.time()

    # Treat 0 as "no limit" for convenience
    max_samples = args.max_samples if args.max_samples > 0 else None

    try:
        out_dir = dataset_dir(args.data_name, Path(args.data_root))
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 2
    err = _prepare_out_dir(out_dir, args.overwrite)
    if err:
        print(f"ERROR: {err}")
        return 2

    source_dir = Path(args.source_dir) if args.source_dir else None
    n_heldout = args.heldout_test_conversations
    by_conversation = n_heldout > 0 or args.split_by_conversation
    extra_opts = n_heldout > 0 or args.split_by_conversation or args.max_source_shards > 0

    heldout = None
    exclusion = None
    if n_heldout > 0:
        print("=" * 70)
        print("HELD-OUT TEST SELECTION")
        print("=" * 70)
        try:
            heldout = select_heldout_conversations(n_heldout, args.seed, source_dir,
                                                   args.max_source_shards or None)
        except ValueError as exc:
            print(f"ERROR: {exc}")
            return 2
        exclusion = {"conversations": heldout["_conv_set"], "voices": heldout["_voice_set"],
                     "part1_voices": heldout["_part1_voices"]}
        print(f"PART3-Test conversations : {heldout['n_part3_test_conversations']:,}")
        print(f"Candidates               : {heldout['n_candidates']:,} "
              "(voices in no other conversation)")
        print(f"Held out                 : {len(heldout['conversations']):,} conversations, "
              f"{heldout['n_part3_test_rows']:,} PART3-Test rows, {len(heldout['voices']):,} voices")
        print(f"PART1-Test voices        : {heldout['part1_test_voices_count']:,} (also excluded)")
        print(f"Train shards scanned     : {heldout['train_shards_scanned']}")
        print()

    staging_dir = out_dir / "_staging"
    staging = ShardWriter(staging_dir, "stage", args.rows_per_shard, args.row_group_size)

    # ---- pass 1: collect ----
    collect_kwargs = {}
    if extra_opts:
        collect_kwargs = {"exclusion": exclusion, "keep_group_keys": by_conversation,
                          "max_source_shards": args.max_source_shards or None}
    meta, collection_stats = collect_records(
        max_samples=max_samples,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        max_clips_per_speaker=args.max_clips_per_speaker,
        part1_fraction=args.part1_fraction,
        staging=staging,
        source_dir=source_dir,
        **collect_kwargs,
    )
    if not meta:
        print("No records collected. Check dataset ID and internet connection.")
        shutil.rmtree(out_dir, ignore_errors=True)
        return 1
    if n_heldout > 0 and max_samples and len(meta) < max_samples:
        print(f"WARNING: only {len(meta):,} of {max_samples:,} clips available after the "
              "held-out exclusion (cap / shard limit); continuing with all of them.")

    dist = speaker_distribution(meta, args.max_clips_per_speaker)

    # ---- split (metadata only) ----
    if by_conversation:
        meta_train, meta_val, split_info = conversation_split(
            meta, val_fraction=args.val_fraction, seed=args.seed,
        )
    else:
        meta_train, meta_val, split_info = speaker_split(
            meta, val_fraction=args.val_fraction, seed=args.seed,
        )
    split_info["train_hours"] = round(sum(r["duration_sec"] for r in meta_train) / 3600, 3)
    split_info["val_hours"] = round(sum(r["duration_sec"] for r in meta_val) / 3600, 3)

    # ---- pass 2: write ----
    print()
    print("=" * 70)
    print("SAVING PREPROCESSED DATA")
    print("=" * 70)
    t_write = time.time()
    layout = write_split(meta_train, meta_val, staging_dir, out_dir,
                         args.rows_per_shard, args.row_group_size)
    write_s = time.time() - t_write

    configs = list(collection_stats["per_config"].keys())
    metadata = {
        "format_version": FORMAT_VERSION,
        "data_name":      args.data_name,
        "dataset_id":     DATASET_ID,
        "dataset_config": DATASET_CONFIG,
        "configs":        configs,
        "max_samples":    max_samples,
        "min_duration":   args.min_duration,
        "max_duration":   args.max_duration,
        "val_fraction":   args.val_fraction,
        "seed":           args.seed,
        "max_clips_per_speaker": args.max_clips_per_speaker,
        "part1_fraction": args.part1_fraction,
        "audio_format":   {"codec": "flac", "subtype": "PCM_16",
                           "sampling_rate": TARGET_SR, "channels": 1},
        "layout": {
            "rows_per_shard": args.rows_per_shard,
            "row_group_size": args.row_group_size,
            "train_shards":   layout["train"],
            "val_shards":     layout["val"],
        },
        "collection": collection_stats,
        "timing": {
            "download_s": collection_stats["download_s"],
            "process_s":  collection_stats["process_s"],
            "write_s":    round(write_s, 1),
            "total_s":    round(time.time() - t0, 1),
        },
        "peak_rss_mb": _peak_rss_mb(),
        "speaker_distribution": dist,
        "split": split_info,
    }
    if extra_opts:
        metadata["heldout_test_conversations"] = n_heldout
        metadata["split_by_conversation"] = by_conversation
        metadata["max_source_shards"] = args.max_source_shards
    if heldout is not None:
        public = {k: v for k, v in heldout.items() if not k.startswith("_")}
        with open(out_dir / HELDOUT_MANIFEST_NAME, "w", encoding="utf-8") as fh:
            json.dump(public, fh, indent=2)
        metadata["heldout_test"] = {
            "manifest":          HELDOUT_MANIFEST_NAME,
            "n_conversations":   len(heldout["conversations"]),
            "n_part3_test_rows": heldout["n_part3_test_rows"],
            "n_voices":          len(heldout["voices"]),
            "n_candidates":      heldout["n_candidates"],
            "excluded_train_rows": (collection_stats["skipped_heldout_conversation"]
                                    + collection_stats["skipped_heldout_voice"]
                                    + collection_stats["skipped_part1_test_voice"]),
            "excluded_train_conversations": collection_stats["excluded_train_conversations"],
            "excluded_train_voices":        collection_stats["excluded_train_voices"],
        }
    info_path = out_dir / INFO_NAME
    with open(info_path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    size_gb = sum(p.stat().st_size for p in out_dir.rglob("*") if p.is_file()) / 1e9

    # ---- summary ----
    print()
    print("=" * 70)
    print("PREPROCESSING COMPLETE")
    print("=" * 70)
    print(f"Output dir : {out_dir}  ({size_gb:.2f} GB)")
    print(f"Train      : {len(meta_train):,} clips, {split_info['train_hours']:.2f} h, "
          f"{len(split_info['train_speakers']):,} speakers, {len(layout['train'])} shards")
    print(f"Val        : {len(meta_val):,} clips, {split_info['val_hours']:.2f} h, "
          f"{len(split_info['val_speakers']):,} speakers, {len(layout['val'])} shards")
    if "train_voices" in split_info:
        print(f"Voices     : train {len(split_info['train_voices']):,}, "
              f"val {len(split_info['val_voices']):,}; conversations: train "
              f"{len(split_info['train_conversations']):,}, val {len(split_info['val_conversations']):,}")
    if heldout is not None:
        print(f"Held-out   : {out_dir / HELDOUT_MANIFEST_NAME} "
              f"({heldout['n_part3_test_rows']:,} PART3-Test rows)")
    print(f"Clips/speaker: min {dist.get('min')}, median {dist.get('median')}, "
          f"mean {dist.get('mean')}, p90 {dist.get('p90')}, max {dist.get('max')}, "
          f"at cap {dist.get('n_at_cap')}")
    print(f"Timing     : {metadata['timing']}")
    print(f"Peak RSS   : {metadata['peak_rss_mb']} MB")
    print(f"Metadata   : {info_path}")
    print(f"Training speakers are listed in {info_path} (key split.train_speakers).")
    print()
    print("Next:")
    print(f"  python scripts/verify_mnsc_dataset.py --data_name {args.data_name}")
    print(f"  python scripts/train_whisper_speaker_split.py --data_name {args.data_name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
