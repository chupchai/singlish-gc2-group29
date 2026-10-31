"""
Evaluation script for speaker-split fine-tuned Whisper models.

Purpose: evaluate a fine-tuned Whisper model (models/<RUN_NAME>/) for comparison with previous runs.
Test set: 4000 samples (3000 MNSC PART1 + 1000 MNSC PART3), same fixed manifest as all prior experiments
or, with --split val, the model's own validation split (data/part3_15k/val or the legacy 4k val
file); val outputs get a _val tag, so they never replace test files.
Output: results/eval_<RUN_NAME>.csv (per sample) and results/summary_<RUN_NAME>.csv
        (a suffix such as _rows16 / _nrng3 / _mnt225 is added for partial or non-default runs,
        so the default full-run files are never overwritten by experiments). Existing
        eval/summary files are never overwritten unless --overwrite is given.
Decoding code is shared with transcribe.py (whisper_decoding.py): generate(language="en",
task="transcribe"), fp16 on GPU, batch 8.
Temperature fallback is ON by default (chosen by the validation sweeps) and
decoding-only (nothing is saved to the model): OpenAI-style greedy first pass, re-decoded at
temperature 0.2..1.0 when the compression ratio / average logprob check fails. Fallback runs get
an _fb suffix (plus _cr<x> / _lp<x> / _seed<N> for non-default values), so a default run writes
_fb files. --no_temperature_fallback gives plain greedy decoding (the 4k baseline decoding) with
the untagged file names; --temperature_fallback is still accepted and has no effect.
--no_repeat_ngram_size / --max_new_tokens are OFF by default and only used to measure the
repetition-loop effect separately.

Usage (from the project root):
    python scripts/evaluate_whisper_speaker_split.py --run_name whisper-small-mnsc-sg-15k --no_temperature_fallback   # plain
    python scripts/evaluate_whisper_speaker_split.py --run_name whisper-small-mnsc-sg-15k --no_repeat_ngram_size 3 --no_temperature_fallback
    python scripts/evaluate_whisper_speaker_split.py --run_name whisper-small-mnsc-sg-15k --temperature_fallback   # = default
    python scripts/evaluate_whisper_speaker_split.py --run_name whisper-small-mnsc-sg-15k --split val --no_temperature_fallback
    python scripts/evaluate_whisper_speaker_split.py --run_name whisper-small-mnsc-sg-15k --split val --no_repeat_ngram_size 5 --temperature_fallback
    python scripts/evaluate_whisper_speaker_split.py --run_name whisper-small-mnsc-sg-4k --max_rows 16 --no_temperature_fallback   # smoke
    # Held-out retrain (own val = data/part3_15k_heldout/val) and the held-out PART3 rows
    # (PART1 3000 + the PART3-Test rows of its 40 held-out conversations; any --run_name):
    python scripts/evaluate_whisper_speaker_split.py --run_name whisper-small-mnsc-sg-15k-heldout --split val --no_temperature_fallback
    python scripts/evaluate_whisper_speaker_split.py --run_name whisper-small-mnsc-sg-15k-heldout --heldout_test part3_15k_heldout --temperature_fallback
    # Extra set (build_heldout_extra_test.py): PART3-Train clips of the held-out conversations,
    # reported as the separate group PART3-heldout-train-clips (never pooled with PART3-Test):
    python scripts/evaluate_whisper_speaker_split.py --run_name whisper-small-mnsc-sg-15k-heldout --heldout_extra_test part3_15k_heldout_extra_test --temperature_fallback
    python scripts/evaluate_whisper_speaker_split.py --run_name whisper-small-mnsc-sg-15k --heldout_extra_test part3_15k_heldout_extra_test --temperature_fallback
    # Zero-shot baseline: un-fine-tuned Hub model from the HF cache (nothing copied to models/);
    # outputs are named whisper-small-zeroshot unless --run_name is given; test split only:
    python scripts/evaluate_whisper_speaker_split.py --model_id openai/whisper-small --no_temperature_fallback
    python scripts/evaluate_whisper_speaker_split.py --model_id openai/whisper-small --heldout_test part3_15k_heldout
    python scripts/evaluate_whisper_speaker_split.py --model_id openai/whisper-small --model_revision <commit> --heldout_extra_test part3_15k_heldout_extra_test
Per-row CSVs carry conversation_id (PART3) and speaker_id (PART1 speaker / PART3 speaker1)
columns for bootstrap_wer_ci.py.
"""

import argparse
import io
import json
import os
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import snapshot_download

from transformers import set_seed
from jiwer import wer, cer

# Shared decoding (same implementation as transcribe.py); re-exported here
from whisper_decoding import (
    DEFAULT_COMPRESSION_RATIO_THRESHOLD,
    DEFAULT_LOGPROB_THRESHOLD,
    DEFAULT_SEED,
    FALLBACK_TEMPERATURES,
    add_decoding_args,
    decode_batch,
    decoding_kwargs,
    decoding_mode_label,
    generated_sequences,
    load_model,
    resolve_decoding_args,
    token_compression_ratio,
)

# Shared loaders for --split val (same scripts/ folder, as in the training script)
from preprocess_mnsc_data import (
    DATA_DIR,
    DEFAULT_DATA_NAME,
    HELDOUT_MANIFEST_NAME,
    SPLIT_INFO,
    VAL_PARQUET,
    dataset_dir,
    extract_speaker_id_any,
    load_legacy_parquet,
    load_preprocessed_index,
    read_clip_audio,
    row_conversation,
)
from build_heldout_extra_test import EXTRA_TEST_CONFIG, load_extra_test_index

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent              # scripts/ holds code only

# ============================================================
# CONFIGURATION
# ============================================================

# Default model to evaluate (output of train_whisper_speaker_split.py).
# Override with --run_name (e.g. whisper-small-mnsc-sg-4k for the baseline).
DEFAULT_RUN_NAME = "whisper-small-mnsc-sg-15k"
MODELS_DIR = PROJECT_ROOT / "models"

# --model_id: un-fine-tuned Hub model (zero-shot baseline), loaded from the HF cache.
# Only these files are fetched (skips the TF/Flax/pytorch_model.bin weights).
HUB_MODEL_FILES = ["*.json", "*.txt", "model.safetensors"]
HUB_REVISION_DEFAULT = "main"
ZEROSHOT_SUFFIX = "-zeroshot"
HUB_ID_RE = re.compile(r"^[A-Za-z0-9][\w.-]*/[\w.-]+$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")

# Manifest (input) lives in data/; evaluation outputs go to results/
MANIFEST_PATH = PROJECT_ROOT / "data" / "mnsc_extended_evaluation_manifest.csv"
RESULTS_DIR   = PROJECT_ROOT / "results"

# --split val: each model is tuned on the validation split it was trained with.
LEGACY_RUN_NAME = "whisper-small-mnsc-sg-4k"  # = train_whisper_speaker_split.LEGACY_RUN_NAME
# Held-out retrain: 40 PART3-Test conversations never seen in its train/val data
HELDOUT_RUN_NAME = "whisper-small-mnsc-sg-15k-heldout"
HELDOUT_DATA_NAME = "part3_15k_heldout"
TEST_CONFIGS = ["ASR-PART1-Test", "ASR-PART3-Test"]
DATASET_LABEL = "MNSC-Extended"  # 'dataset' column, same value as the test manifest

# Where the MNSC test parquet files are located
HF_DATASET_ID = "AudioLLMs/Multitask-National-Speech-Corpus-v1-extend"

# Samples decoded together per generate() call.
# 8 fits comfortably in 8 GB VRAM for whisper-small in fp16;
# lower this if you hit CUDA out-of-memory errors.
EVAL_BATCH_SIZE = 8

# Temperature fallback constants (FALLBACK_TEMPERATURES, DEFAULT_*_THRESHOLD,
# DEFAULT_SEED) live in whisper_decoding.py; fallback is ON by default.


# ============================================================
# TRANSCRIPT NORMALIZATION
# ============================================================

def normalize_transcript(text):
    if not text:
        return ""

    # Remove speaker labels
    text = re.sub(
        r"<Speaker\d+>:\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    # Remove square-bracket annotation markers
    # while preserving their contents
    while re.search(r"\[[^\[\]]+\]", text):
        text = re.sub(
            r"\[([^\[\]]+)\]",
            r"\1",
            text
        )

    # Remove curly-bracket markers
    # while preserving their contents
    text = re.sub(
        r"\{([^{}\[\]]+)\}",
        r"\1",
        text
    )

    # Remove parentheses while preserving
    # spoken content
    text = re.sub(
        r"\(([^()]*)\)",
        r"\1",
        text
    )

    # Remove hash markers while preserving content
    text = re.sub(
        r"#([^#]+)#",
        r"\1",
        text
    )

    # Normalize whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    # Lowercase
    text = text.lower()

    return text


def normalize_for_evaluation(text):
    """
    Additional normalization used specifically for WER/CER.

    Keeps the same transcript content but removes punctuation,
    except apostrophes.
    """

    text = normalize_transcript(text)

    # Remove punctuation except apostrophes
    text = re.sub(
        r"[^\w\s']",
        "",
        text
    )

    # Normalize whitespace again
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ============================================================
# HELPERS
# ============================================================

def output_suffix(max_rows: int, no_repeat_ngram_size: int, max_new_tokens: int,
                  temperature_fallback: bool = False,
                  compression_ratio_threshold: float = DEFAULT_COMPRESSION_RATIO_THRESHOLD,
                  logprob_threshold: float = DEFAULT_LOGPROB_THRESHOLD,
                  seed: int = DEFAULT_SEED, split_tag: str = "") -> str:
    """'' for a full default test run, else e.g. '_val_rows16_nrng3_mnt225_fb_cr1.5'."""
    s = split_tag
    if max_rows > 0:
        s += f"_rows{max_rows}"
    if no_repeat_ngram_size > 0:
        s += f"_nrng{no_repeat_ngram_size}"
    if max_new_tokens > 0:
        s += f"_mnt{max_new_tokens}"
    if temperature_fallback:
        s += "_fb"
        if compression_ratio_threshold != DEFAULT_COMPRESSION_RATIO_THRESHOLD:
            s += f"_cr{compression_ratio_threshold:g}"
        if logprob_threshold != DEFAULT_LOGPROB_THRESHOLD:
            s += f"_lp{logprob_threshold:g}"
        if seed != DEFAULT_SEED:
            s += f"_seed{seed}"
    return s


def output_paths(run_name: str, suffix: str):
    return (RESULTS_DIR / f"eval_{run_name}{suffix}.csv",
            RESULTS_DIR / f"summary_{run_name}{suffix}.csv")


def run_suffix(args) -> str:
    """Output suffix of a parsed run, e.g. '_fb' for a default full test run,
    '' for --no_temperature_fallback, '_val_nrng5_fb', '_ho-part3_15k_heldout_fb'."""
    return output_suffix(args.max_rows, args.no_repeat_ngram_size, args.max_new_tokens,
                         args.temperature_fallback, args.compression_ratio_threshold,
                         args.logprob_threshold, args.seed,
                         split_tag=split_tag(args.split, args.run_name, args.data_name)
                         + heldout_tag(args.heldout_test)
                         + heldout_extra_tag(args.heldout_extra_test))


def existing_outputs(paths) -> List[Path]:
    """The output paths that already exist (never overwritten without --overwrite)."""
    return [Path(p) for p in paths if Path(p).exists()]


def hub_run_name(model_id: str) -> str:
    """Output run name of a --model_id run: 'openai/whisper-small' -> 'whisper-small-zeroshot'."""
    return model_id.split("/")[-1] + ZEROSHOT_SUFFIX


def resolve_hub_model(model_id: str, revision: str) -> Tuple[Path, str]:
    """(local snapshot folder, resolved 40-hex commit) of a Hub model at a revision. Uses the
    HF cache (offline: the cached snapshot); the snapshot folder name is the commit. A full
    commit hash as revision must match the resolved one."""
    local_dir = Path(snapshot_download(repo_id=model_id, revision=revision,
                                       allow_patterns=HUB_MODEL_FILES))
    commit = local_dir.name
    if not COMMIT_RE.match(commit):
        raise ValueError(f"could not resolve a commit hash for {model_id}@{revision} "
                         f"(snapshot folder {local_dir})")
    if COMMIT_RE.match(revision) and commit != revision:
        raise ValueError(f"{model_id}: requested commit {revision} but got {commit}")
    return local_dir, commit


def default_val_data_name(run_name: str) -> str:
    """Data the model was trained on: 'legacy' for the 4k baseline, part3_15k_heldout for
    the held-out retrain and its learning-rate / seed sweep runs (names starting with
    '<held-out name>-', optionally prefixed 'smoke-'), else the training default."""
    base = run_name[len("smoke-"):] if run_name.startswith("smoke-") else run_name
    if base == HELDOUT_RUN_NAME or base.startswith(HELDOUT_RUN_NAME + "-"):
        return HELDOUT_DATA_NAME
    return "legacy" if run_name == LEGACY_RUN_NAME else DEFAULT_DATA_NAME


def heldout_tag(data_name: Optional[str]) -> str:
    """Output tag of a --heldout_test run, e.g. '_ho-part3_15k_heldout' ('' when off)."""
    return f"_ho-{data_name}" if data_name else ""


def load_heldout_rows(data_name: str, data_root: Optional[Path] = None) -> List[int]:
    """PART3-Test row_index values held out by data/<data_name>/heldout_test_manifest.json."""
    path = dataset_dir(data_name, Path(data_root) if data_root else DATA_DIR) / HELDOUT_MANIFEST_NAME
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Build it with: python scripts/preprocess_mnsc_data.py "
            f"--data_name {data_name} --heldout_test_conversations 40 --max_source_shards 30")
    with open(path, "r", encoding="utf-8") as fh:
        return [int(i) for i in json.load(fh)["part3_test_row_indices"]]


def filter_heldout(manifest: pd.DataFrame, rows: List[int]) -> pd.DataFrame:
    """All ASR-PART1-Test rows + the listed ASR-PART3-Test rows, manifest order."""
    wanted = set(rows)
    is_p3 = manifest["config"] == "ASR-PART3-Test"
    keep = (manifest["config"] == "ASR-PART1-Test") | (is_p3 & manifest["row_index"].isin(wanted))
    found = set(manifest.loc[is_p3 & keep, "row_index"].astype(int))
    missing = wanted - found
    assert not missing, f"held-out PART3-Test rows not in the manifest: {sorted(missing)[:10]}"
    return manifest[keep].reset_index(drop=True)


def val_config_name(data_name: str) -> str:
    """'config' column / summary group of validation rows, e.g. 'part3_15k-val'."""
    return f"{data_name}-val"


def split_tag(split: str, run_name: str, data_name: Optional[str]) -> str:
    """'' for test, '_val' for the model's own val split, '_val-<data_name>' for another one."""
    if split != "val":
        return ""
    if data_name == default_val_data_name(run_name):
        return "_val"
    return f"_val-{data_name}"


def heldout_extra_tag(data_name: Optional[str]) -> str:
    """Output tag of a --heldout_extra_test run, e.g. '_hx-part3_15k_heldout_extra_test' ('' when off)."""
    return f"_hx-{data_name}" if data_name else ""


def summary_groups(split: str, data_name: Optional[str], extra_test: Optional[str] = None) -> List[str]:
    """Per-configuration summary rows (after Overall)."""
    if extra_test:
        return [EXTRA_TEST_CONFIG]
    if split != "val":
        return list(TEST_CONFIGS)
    return [val_config_name(data_name)]


def load_extra_test_manifest(data_name: str, data_root: Optional[str] = None,
                             max_rows: int = 0) -> Tuple[pd.DataFrame, List[dict]]:
    """Manifest-like frame for the extra held-out test set (row_index = position in the set)
    plus the full lazy item list (no audio)."""
    items, _ = load_extra_test_index(data_name, Path(data_root) if data_root else DATA_DIR)
    manifest = pd.DataFrame({
        "sample_id": [it["clip_id"] for it in items],
        "dataset": DATASET_LABEL,
        "config": EXTRA_TEST_CONFIG,
        "row_index": list(range(len(items))),
    })
    assert manifest["sample_id"].duplicated().sum() == 0
    if max_rows > 0:
        manifest = manifest.head(max_rows)
    return manifest, items


def test_cluster_ids(lookup: dict, manifest_row: dict) -> Tuple[str, str]:
    """(conversation_id, speaker_id) of a test row; PART1 rows have conversation_id ''."""
    attrs = lookup[(manifest_row["config"], int(manifest_row["row_index"]))].get("other_attributes")
    return row_conversation(attrs), extract_speaker_id_any({"other_attributes": attrs})


def data_source_label(split: str, data_name: Optional[str], data_root: Optional[str],
                      heldout_test: Optional[str] = None,
                      heldout_rows: Optional[List[int]] = None,
                      extra_test: Optional[str] = None) -> str:
    """Human-readable data source for the DECODING CONFIG block."""
    if split != "val" and extra_test:
        d = dataset_dir(extra_test, Path(data_root) if data_root else DATA_DIR)
        return (f"{d / 'test'} (lazy FLAC): PART3-Train clips of the held-out conversations "
                f"(separate group, not PART3-Test)")
    if split != "val" and heldout_test:
        return (f"MNSC test manifest, held-out subset ({MANIFEST_PATH.name}: ASR-PART1-Test 3000 + "
                f"{len(heldout_rows or [])} ASR-PART3-Test rows from {heldout_test}/{HELDOUT_MANIFEST_NAME})")
    if split != "val":
        return f"MNSC test manifest ({MANIFEST_PATH.name}: ASR-PART1-Test 3000 + ASR-PART3-Test 1000)"
    if data_name == "legacy":
        return f"legacy 4k validation split ({VAL_PARQUET.name}, 990 clips, lazy float32)"
    return f"{dataset_dir(data_name, Path(data_root) if data_root else DATA_DIR) / 'val'} (lazy FLAC)"


# ============================================================
# LEGACY VALIDATION AUDIO (eval-only, lazy)
# ============================================================

# The legacy val file stores all 990 audio_bytes values in ONE snappy-compressed
# Parquet dictionary page (one row group, one write batch). pyarrow decodes that page
# into a full BinaryArray (measured 4.6-9.6 GB peak), and load_legacy_parquet also
# converts every clip to float32 up front. Here the page is read and decompressed once
# (~1.8 GB) and each clip is converted to float32 only when it is decoded.
# load_legacy_parquet (shared with training) remains the fallback.
LEGACY_META_COLUMNS = ["audio_length", "sampling_rate", "text", "speaker_id", "duration_sec"]
LEGACY_AUDIO_COLUMN = "audio_bytes"


def _thrift_varint(buf: bytes, pos: int) -> Tuple[int, int]:
    """Unsigned LEB128 varint at pos -> (value, next pos)."""
    value = shift = 0
    while True:
        b = buf[pos]
        pos += 1
        value |= (b & 0x7F) << shift
        if not b & 0x80:
            return value, pos
        shift += 7


def _thrift_zigzag(buf: bytes, pos: int) -> Tuple[int, int]:
    n, pos = _thrift_varint(buf, pos)
    return (n >> 1) ^ -(n & 1), pos


def _thrift_skip(buf: bytes, pos: int, ftype: int) -> int:
    """Skip one thrift-compact value of type ftype (bools in lists take one byte)."""
    if ftype in (1, 2):
        return pos + 1
    if ftype == 3:
        return pos + 1
    if ftype in (4, 5, 6):
        return _thrift_varint(buf, pos)[1]
    if ftype == 7:
        return pos + 8
    if ftype == 8:
        n, pos = _thrift_varint(buf, pos)
        return pos + n
    if ftype in (9, 10):
        head = buf[pos]
        pos += 1
        size, etype = head >> 4, head & 0x0F
        if size == 15:
            size, pos = _thrift_varint(buf, pos)
        for _ in range(size):
            pos = _thrift_skip(buf, pos, etype)
        return pos
    if ftype == 11:
        size, pos = _thrift_varint(buf, pos)
        if size:
            kv = buf[pos]
            pos += 1
            for _ in range(size):
                pos = _thrift_skip(buf, pos, kv >> 4)
                pos = _thrift_skip(buf, pos, kv & 0x0F)
        return pos
    if ftype == 12:
        return _thrift_struct(buf, pos)[1]
    raise ValueError(f"unknown thrift compact type {ftype}")


def _thrift_struct(buf: bytes, pos: int) -> Tuple[dict, int]:
    """Thrift-compact struct -> ({field_id: value} for int/bool/struct fields, next pos)."""
    fields = {}
    field_id = 0
    while True:
        head = buf[pos]
        pos += 1
        ftype = head & 0x0F
        if ftype == 0:
            return fields, pos
        delta = head >> 4
        if delta:
            field_id += delta
        else:
            field_id, pos = _thrift_zigzag(buf, pos)
        if ftype in (1, 2):
            fields[field_id] = ftype == 1
        elif ftype in (4, 5, 6):
            fields[field_id], pos = _thrift_zigzag(buf, pos)
        elif ftype == 12:
            fields[field_id], pos = _thrift_struct(buf, pos)
        else:
            pos = _thrift_skip(buf, pos, ftype)


class LegacyValAudio:
    """Decompressed audio_bytes values of a legacy file; float32 conversion per clip."""

    def __init__(self, raw: "pa.Buffer", spans: List[Tuple[int, int]], lengths: List[int]):
        self.raw = raw          # all values back to back, in row order
        self.spans = spans      # (start, nbytes) of row i in raw
        self.lengths = lengths  # audio_length column

    def clip(self, i: int) -> np.ndarray:
        """Same expression as load_legacy_parquet: frombuffer(bytes)[:length].copy()."""
        if i >= len(self.spans):
            raise IndexError(f"legacy val row {i} was not kept (only {len(self.spans)} rows loaded)")
        start, nbytes = self.spans[i]
        return np.frombuffer(self.raw.slice(start, nbytes), dtype=np.float32)[:self.lengths[i]].copy()

    def keep_first(self, n: int) -> None:
        """--max_rows: keep only the bytes of rows 0..n-1 (frees the rest)."""
        if 0 < n < len(self.spans):
            start, nbytes = self.spans[n - 1]
            self.raw = pa.py_buffer(self.raw.slice(0, start + nbytes).to_pybytes())
            self.spans = self.spans[:n]


def _legacy_audio_page(path: Path, lengths: List[int]) -> Tuple[Optional[LegacyValAudio], str]:
    """Read the audio_bytes dictionary page directly -> (LegacyValAudio, '') or
    (None, reason) if the file does not have the expected single-page layout."""
    pf = pq.ParquetFile(str(path))
    try:
        meta = pf.metadata
        if meta.num_row_groups != 1:
            return None, f"{meta.num_row_groups} row groups"
        atype = pf.schema_arrow.field(LEGACY_AUDIO_COLUMN).type
        if not (pa.types.is_binary(atype) or pa.types.is_large_binary(atype)):
            return None, f"{LEGACY_AUDIO_COLUMN} type {atype}"
        rg = meta.row_group(0)
        col = next((rg.column(j) for j in range(rg.num_columns)
                    if rg.column(j).path_in_schema == LEGACY_AUDIO_COLUMN), None)
        if col is None:
            return None, f"no {LEGACY_AUDIO_COLUMN} column chunk"
        if not col.has_dictionary_page or col.dictionary_page_offset is None:
            return None, "no dictionary page"
        if col.compression not in ("SNAPPY", "UNCOMPRESSED"):
            return None, f"codec {col.compression}"
        if col.num_values != rg.num_rows or len(lengths) != rg.num_rows:
            return None, "value count != row count"
        offset = col.dictionary_page_offset
        region = col.data_page_offset - offset
        codec = col.compression
    finally:
        pf.close()

    with open(path, "rb") as fh:
        fh.seek(offset)
        head = fh.read(64 * 1024)
        try:
            header, header_len = _thrift_struct(head, 0)
        except (IndexError, ValueError):
            return None, "unreadable page header"
        dict_header = header.get(7)
        if header.get(1) != 2 or not isinstance(dict_header, dict):
            return None, f"page type {header.get(1)} is not a dictionary page"
        if dict_header.get(1) != len(lengths):
            return None, f"dictionary has {dict_header.get(1)} values for {len(lengths)} rows"
        if dict_header.get(2) not in (0, 2):
            return None, f"dictionary encoding {dict_header.get(2)}"
        size, comp_size = header.get(2), header.get(3)
        if not isinstance(size, int) or not isinstance(comp_size, int):
            return None, "page header without sizes"
        if header_len + comp_size != region:
            return None, "dictionary page does not end at the data page"
        fh.seek(offset + header_len)
        comp = fh.read(comp_size)
    if len(comp) != comp_size:
        return None, "truncated dictionary page"
    if codec == "SNAPPY":
        raw = pa.decompress(comp, decompressed_size=size, codec="snappy",
                            memory_pool=pa.system_memory_pool())
    else:
        raw = pa.py_buffer(comp)
    del comp
    if raw.size != size:
        return None, "decompressed size != page header"

    # PLAIN BYTE_ARRAY values: 4-byte little-endian length + bytes
    spans = []
    view = memoryview(raw)
    pos = 0
    try:
        for length in lengths:
            if pos + 4 > size:
                return None, "dictionary page ends early"
            nbytes = int.from_bytes(view[pos:pos + 4], "little")
            pos += 4
            if nbytes != 4 * length:
                # also guards the row <-> dictionary index identity
                return None, f"value {len(spans)} has {nbytes} bytes, audio_length {length}"
            spans.append((pos, nbytes))
            pos += nbytes
    finally:
        view.release()
    if pos != size:
        return None, "extra bytes after the last value"
    return LegacyValAudio(raw, spans, lengths), ""


def load_legacy_val_lazy(path, keep_rows: int = 0) -> List[dict]:
    """Legacy val records like load_legacy_parquet, but the audio stays in one decompressed
    buffer and val_clip() converts it to float32 per clip. keep_rows > 0 keeps only the
    first rows' audio (--max_rows). Falls back to load_legacy_parquet on another layout."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Preprocessed data not found at {p}.\n"
            "Run preprocess_mnsc_data.py first to generate it."
        )
    rows = pq.read_table(str(p), columns=LEGACY_META_COLUMNS).to_pydict()
    audio, reason = _legacy_audio_page(p, rows["audio_length"])
    if audio is None:
        print(f"WARNING: {p.name}: {reason}; falling back to the eager loader (~9-10 GB RAM).")
        return load_legacy_parquet(p)
    if keep_rows > 0:
        audio.keep_first(keep_rows)
    return [{
        "legacy_audio":  audio,
        "legacy_row":    i,
        "sampling_rate": int(rows["sampling_rate"][i]),
        "text":          rows["text"][i],
        "speaker_id":    rows["speaker_id"][i],
        "duration_sec":  float(rows["duration_sec"][i]),
    } for i in range(len(rows["text"]))]


def load_val_manifest(data_name: str, data_root: Optional[str] = None,
                      max_rows: int = 0) -> Tuple[pd.DataFrame, List[dict]]:
    """Manifest-like frame for the validation split (row_index = position in load order)
    plus the full item list. Legacy: lazy items (audio decompressed once, float32 per clip);
    else lazy index items (no audio)."""
    expected = None
    if data_name == "legacy":
        items = load_legacy_val_lazy(VAL_PARQUET, keep_rows=max_rows)
        if SPLIT_INFO.exists():
            with open(SPLIT_INFO, "r", encoding="utf-8") as fh:
                expected = json.load(fh)["split"]["val_samples"]
    else:
        _, items, info = load_preprocessed_index(data_name, Path(data_root) if data_root else DATA_DIR)
        expected = info["split"]["val_samples"]
    if expected is not None:
        assert len(items) == expected, f"val clips {len(items)} != split info {expected}"

    config = val_config_name(data_name)
    manifest = pd.DataFrame({
        "sample_id": [it.get("clip_id") or f"{config}_{i:06d}" for i, it in enumerate(items)],
        "dataset": DATASET_LABEL,
        "config": config,
        "row_index": list(range(len(items))),
    })
    assert manifest["sample_id"].duplicated().sum() == 0
    if max_rows > 0:
        manifest = manifest.head(max_rows)
    return manifest, items


def val_clip(item: dict, cache: dict):
    """(audio, sampling_rate, transcript) of one val item; legacy items are converted to
    float32 here (LegacyValAudio.clip), index items are decoded with read_clip_audio,
    exactly like WhisperMNSCDataset in the training script."""
    audio = item.get("audio")
    if audio is None:
        if "legacy_audio" in item:
            audio = item["legacy_audio"].clip(item["legacy_row"])
        else:
            audio = read_clip_audio(item, cache)
    return audio, item["sampling_rate"], item["text"]


def test_clip(lookup: dict, manifest_row: dict):
    """(audio, sampling_rate, answer) of one test manifest row."""
    row = lookup[(manifest_row["config"], int(manifest_row["row_index"]))]

    audio, sample_rate = sf.read(
        io.BytesIO(row["context"]["bytes"]),
        dtype="float32"
    )
    return audio, sample_rate, row["answer"]


def subset_manifest(manifest: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    """First ceil(N/2) PART1-Test + floor(N/2) PART3-Test rows, manifest order."""
    if max_rows <= 0:
        return manifest
    n1 = (max_rows + 1) // 2
    n3 = max_rows // 2
    keep = pd.concat([
        manifest[manifest["config"] == "ASR-PART1-Test"].head(n1),
        manifest[manifest["config"] == "ASR-PART3-Test"].head(n3),
    ])
    return manifest.loc[sorted(keep.index)]


def load_parquet_rows(parquet_path):
    """
    Load all rows from a MNSC parquet file.
    """

    table = pq.read_table(parquet_path)

    rows = table.to_pylist()

    return rows


def load_test_manifest(max_rows: int, heldout_rows: Optional[List[int]] = None) -> Tuple[pd.DataFrame, dict]:
    """Fixed 4000-row test manifest (held-out subset for --heldout_test, then subset for --max_rows)
    + (config, row_index) -> parquet row lookup."""

    # ============================================================
    # LOAD MANIFEST
    # ============================================================

    print("=" * 70)
    print("LOADING FIXED EVALUATION MANIFEST")
    print("=" * 70)

    manifest = pd.read_csv(MANIFEST_PATH)

    print("Rows:", len(manifest))
    print()
    print(manifest["config"].value_counts())

    # Important sanity checks (always on the full manifest)
    assert len(manifest) == 4000
    assert manifest["sample_id"].duplicated().sum() == 0
    assert manifest["config"].isin(
        ["ASR-PART1-Test", "ASR-PART3-Test"]
    ).all()

    print("\nManifest verification: PASS")

    # ============================================================
    # LOAD TEST PARQUET DATA
    # ============================================================

    print("\n" + "=" * 70)
    print("LOADING MNSC TEST DATA")
    print("=" * 70)

    # Downloads only the two test splits on first run; afterwards reuses the
    # local HuggingFace cache (~/.cache/huggingface/hub) without re-downloading.
    print("Resolving MNSC test splits (downloads on first run, then cached)...")
    dataset_snapshot_dir = Path(
        snapshot_download(
            repo_id=HF_DATASET_ID,
            repo_type="dataset",
            allow_patterns=["ASR-PART1-Test/*", "ASR-PART3-Test/*"],
        )
    )
    print("Snapshot dir:", dataset_snapshot_dir)

    parquet_files = {
        "ASR-PART1-Test": sorted(
            (dataset_snapshot_dir / "ASR-PART1-Test").glob("*.parquet")
        ),
        "ASR-PART3-Test": sorted(
            (dataset_snapshot_dir / "ASR-PART3-Test").glob("*.parquet")
        ),
    }

    # ============================================================
    # BUILD LOOKUP
    # ============================================================

    print("\nBuilding row lookup...")

    lookup = {}

    for config, paths in parquet_files.items():

        global_row_index = 0

        for parquet_path in paths:

            print(
                f"Reading {config}: "
                f"{os.path.basename(parquet_path)}"
            )

            rows = load_parquet_rows(parquet_path)

            print("Rows:", len(rows))

            for row in rows:

                lookup[(config, global_row_index)] = row

                global_row_index += 1

    print("\nTotal lookup rows:", len(lookup))

    # ============================================================
    # VERIFY MANIFEST AGAINST PARQUET
    # ============================================================

    print("\nVerifying manifest rows exist...")

    for _, manifest_row in manifest.iterrows():

        key = (
            manifest_row["config"],
            int(manifest_row["row_index"])
        )

        assert key in lookup, (
            f"Missing row: {key}"
        )

    print("Manifest parquet verification: PASS")

    if heldout_rows is not None:
        manifest = filter_heldout(manifest, heldout_rows)
        n1 = int((manifest["config"] == "ASR-PART1-Test").sum())
        print(f"\nHELD-OUT MODE: PART1 {n1} + PART3 {len(manifest) - n1}")

    n_rows = len(manifest)
    manifest = subset_manifest(manifest, max_rows)
    if max_rows > 0:
        print(f"\nSMOKE MODE: evaluating {len(manifest)} of {n_rows} rows")

    return manifest, lookup


def corpus_metric(group_df, metric):

    references = group_df["reference"].tolist()
    predictions = group_df["prediction"].tolist()

    if metric == "WER":
        return wer(
            references,
            predictions
        )

    elif metric == "CER":
        return cer(
            references,
            predictions
        )


def n_runaway(group_df) -> int:
    """Predictions with > 3x as many words as the reference (repetition loops)."""
    ref_len = group_df["reference"].fillna("").str.split().str.len()
    pred_len = group_df["prediction"].fillna("").str.split().str.len()
    return int((pred_len > 3 * ref_len).sum())


def n_cr_gt_threshold(group_df, threshold: float) -> int:
    """Predictions still above the compression-ratio threshold after fallback."""
    return int((group_df["compression_ratio"] > threshold).sum())


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a fine-tuned Whisper model on the fixed MNSC test manifest, "
                                            "or with --split val on the model's validation split.")
    p.add_argument("--run_name", default=None,
                   help=f"Model folder under models/ (default {DEFAULT_RUN_NAME}). With --model_id: "
                        f"only the output name (default <repo name>{ZEROSHOT_SUFFIX}, e.g. "
                        f"whisper-small{ZEROSHOT_SUFFIX}); it must not name a folder under models/.")
    p.add_argument("--model_id", default=None, metavar="REPO_ID",
                   help="OFF by default. Evaluate an un-fine-tuned Hugging Face Hub model (zero-shot "
                        "baseline, e.g. openai/whisper-small) loaded through the HF cache; nothing is "
                        "copied to models/. Same language/task forcing and decoding as the fine-tuned "
                        "models. Test split only (with --heldout_test / --heldout_extra_test).")
    p.add_argument("--model_revision", default=None, metavar="REV",
                   help=f"With --model_id: branch, tag or commit hash (default {HUB_REVISION_DEFAULT}). "
                        "The resolved commit is printed and written to the summary CSV "
                        "(columns model_id, model_revision, model_commit).")
    p.add_argument("--split", choices=("test", "val"), default="test",
                   help="test (default): fixed 4000-row MNSC test manifest. val: the model's validation "
                        "split; choose decoding settings on val, then run test once.")
    p.add_argument("--data_name", default=None,
                   help="With --split val: dataset under data/ whose val/ clips are used. Default = what the "
                        f"model was trained on: legacy for {LEGACY_RUN_NAME}, {HELDOUT_DATA_NAME} for "
                        f"{HELDOUT_RUN_NAME} and its sweep runs ({HELDOUT_RUN_NAME}-*), "
                        f"else {DEFAULT_DATA_NAME}.")
    p.add_argument("--data_root", default=None,
                   help="With --split val: parent folder of data/<data_name>/ (default <project>/data). "
                        "Not for legacy.")
    p.add_argument("--heldout_test", default=None, metavar="DATA_NAME",
                   help="OFF by default. Test split only: score all ASR-PART1-Test rows plus only the "
                        f"ASR-PART3-Test rows listed in data/<DATA_NAME>/{HELDOUT_MANIFEST_NAME} "
                        f"(e.g. {HELDOUT_DATA_NAME}); works for any --run_name. Output tag _ho-<DATA_NAME>.")
    p.add_argument("--heldout_extra_test", default=None, metavar="DATA_NAME",
                   help="OFF by default. Test split only, not with --heldout_test: score the extra set "
                        "data/<DATA_NAME>/ from build_heldout_extra_test.py (PART3-Train clips of the "
                        "held-out conversations, e.g. part3_15k_heldout_extra_test) as the separate group "
                        f"{EXTRA_TEST_CONFIG}; works for any --run_name. Output tag _hx-<DATA_NAME>. "
                        "--data_root may be given with it.")
    p.add_argument("--batch_size", type=int, default=EVAL_BATCH_SIZE,
                   help=f"Clips per generate() call (default {EVAL_BATCH_SIZE}).")
    p.add_argument("--max_rows", type=int, default=0,
                   help="Smoke test: evaluate only N rows (test: half PART1, half PART3; "
                        "val: first N clips). 0 = all.")
    add_decoding_args(p)  # fallback ON by default; --no_temperature_fallback = plain greedy
    p.add_argument("--overwrite", action="store_true",
                   help="Replace existing eval/summary files; by default an existing file is never "
                        "overwritten (the run stops before loading anything).")
    args = p.parse_args(argv)
    resolve_decoding_args(p, args)
    if args.model_id is None:
        if args.model_revision is not None:
            p.error("--model_revision requires --model_id")
        args.run_name = args.run_name or DEFAULT_RUN_NAME
    else:
        if not HUB_ID_RE.match(args.model_id) or ".." in args.model_id:
            p.error(f"--model_id must be a Hub repo id like openai/whisper-small, got {args.model_id!r} "
                    "(use --run_name for a model under models/)")
        if args.split != "test":
            p.error("--model_id requires --split test (a Hub base model has no own validation split)")
        args.model_revision = args.model_revision or HUB_REVISION_DEFAULT
        if args.run_name is None:
            args.run_name = hub_run_name(args.model_id)
        elif (MODELS_DIR / args.run_name).exists():
            p.error(f"--run_name {args.run_name} is a model folder under models/; choose another "
                    "output name for a --model_id run")
    if args.split != "val" and (args.data_name is not None
                                or (args.data_root is not None and not args.heldout_extra_test)):
        p.error("--data_name/--data_root require --split val")
    if args.heldout_extra_test is not None:
        if args.split != "test":
            p.error("--heldout_extra_test requires --split test")
        if args.heldout_test is not None:
            p.error("--heldout_extra_test cannot be combined with --heldout_test (separate runs)")
        try:
            dataset_dir(args.heldout_extra_test, Path(args.data_root) if args.data_root else DATA_DIR)
        except ValueError as exc:
            p.error(str(exc))
    if args.heldout_test is not None:
        if args.split != "test":
            p.error("--heldout_test requires --split test")
        try:
            dataset_dir(args.heldout_test, DATA_DIR)
        except ValueError as exc:
            p.error(str(exc))
    if args.split == "val":
        args.data_name = args.data_name or default_val_data_name(args.run_name)
        if args.data_name == "legacy":
            if args.data_root is not None:
                p.error("--data_root cannot be used with --data_name legacy")
        else:
            try:
                dataset_dir(args.data_name, Path(args.data_root) if args.data_root else DATA_DIR)
            except ValueError as exc:
                p.error(str(exc))
    return args


# ============================================================
# MAIN
# ============================================================

def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    set_seed(args.seed)  # random/numpy/torch (+CUDA): reproducible sampled fallback passes

    model_path = None if args.model_id else MODELS_DIR / args.run_name
    if model_path is not None and not (model_path / "config.json").exists():
        print(f"ERROR: no trained model at {model_path}.\n"
              "Train it first (train_whisper_speaker_split.py) or pass the right --run_name.")
        return 2

    suffix = run_suffix(args)
    heldout_rows = load_heldout_rows(args.heldout_test) if args.heldout_test else None
    output_path, summary_path = output_paths(args.run_name, suffix)

    print("=" * 70)
    print("DECODING CONFIG")
    print("=" * 70)
    print("decoding mode        :", decoding_mode_label(args.temperature_fallback,
                                                        args.no_repeat_ngram_size))
    print("split                :", args.split)
    print("data source          :", data_source_label(args.split, args.data_name, args.data_root,
                                                      args.heldout_test, heldout_rows,
                                                      args.heldout_extra_test))
    if heldout_rows is not None:
        print("ASR-PART3-Test held-out rows:", len(heldout_rows))
    print("no_repeat_ngram_size :", args.no_repeat_ngram_size or "off")
    print("max_new_tokens       :", args.max_new_tokens or "off")
    print("temperature fallback :", "on" if args.temperature_fallback else "off")
    if args.temperature_fallback:
        print("  temperatures       :", FALLBACK_TEMPERATURES)
        print("  compression_ratio_threshold:", args.compression_ratio_threshold)
        print("  logprob_threshold  :", args.logprob_threshold)
    print("seed                 :", args.seed)
    print("output file          :", output_path.name)
    print()

    # Never replace existing results (a default run targets _fb names that may already exist)
    existing = existing_outputs([output_path, summary_path])
    if existing and not args.overwrite:
        names = ", ".join(p.name for p in existing)
        verb = "exists" if len(existing) == 1 else "exist"
        print(f"ERROR: {names} already {verb} in {RESULTS_DIR.name}/; refusing to overwrite. "
              "Pass --overwrite to replace, or change the decoding flags "
              "(e.g. --no_temperature_fallback for the plain run).")
        return 2

    model_commit = None
    if args.model_id:
        # Resolve before the data load so a missing/unreachable model fails fast
        try:
            model_path, model_commit = resolve_hub_model(args.model_id, args.model_revision)
        except Exception as exc:  # ValueError, OSError and huggingface_hub errors
            print(f"ERROR: cannot load {args.model_id}@{args.model_revision} from the Hub or the "
                  f"HF cache: {exc}")
            return 2
        print(f"model source         : {args.model_id} @ {model_commit} "
              f"(requested revision {args.model_revision})")
        print()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.split == "val" and args.data_name != default_val_data_name(args.run_name):
        print(f"WARNING: {args.run_name} is evaluated on the {args.data_name} validation split, "
              "not the split it was trained with; its training speakers may overlap these "
              "validation speakers (see README).")
        print()

    if args.heldout_extra_test:
        print("=" * 70)
        print("LOADING EXTRA HELD-OUT TEST SET")
        print("=" * 70)

        manifest, extra_items = load_extra_test_manifest(args.heldout_extra_test, args.data_root,
                                                         args.max_rows)
        n_conv = len({it["conversation_id"] for it in extra_items})
        print(f"EXTRA TEST MODE: {len(extra_items)} clips, {n_conv} conversations (separate group "
              f"'{EXTRA_TEST_CONFIG}'; never pooled with PART3-Test)")
        if args.max_rows > 0:
            print(f"\nSMOKE MODE: evaluating {len(manifest)} of {len(extra_items)} clips")

        audio_cache = {}

        def load_clip(manifest_row):
            return val_clip(extra_items[int(manifest_row["row_index"])], audio_cache)

        def cluster_of(manifest_row):
            item = extra_items[int(manifest_row["row_index"])]
            return item["conversation_id"], item["speaker_id"]
    elif args.split == "test":
        manifest, lookup = load_test_manifest(args.max_rows, heldout_rows)

        def load_clip(manifest_row):
            return test_clip(lookup, manifest_row)

        def cluster_of(manifest_row):
            return test_cluster_ids(lookup, manifest_row)
    else:
        print("=" * 70)
        print("LOADING VALIDATION SPLIT")
        print("=" * 70)

        manifest, val_items = load_val_manifest(args.data_name, args.data_root, args.max_rows)
        print("Clips   :", len(val_items))
        print("Speakers:", len({it["speaker_id"] for it in val_items}))
        print("Config  :", val_config_name(args.data_name))
        if args.max_rows > 0:
            print(f"\nSMOKE MODE: evaluating {len(manifest)} of {len(val_items)} validation clips")

        audio_cache = {}  # <= 8 open parquet handles (read_clip_audio), as in training

        def load_clip(manifest_row):
            return val_clip(val_items[int(manifest_row["row_index"])], audio_cache)

        def cluster_of(manifest_row):
            return "", val_items[int(manifest_row["row_index"])]["speaker_id"]

    # ============================================================
    # LOAD MODEL
    # ============================================================

    print("\n" + "=" * 70)
    print(f"LOADING BASE MODEL (zero-shot, {args.model_id})" if args.model_id
          else "LOADING FINE-TUNED MODEL")
    print("=" * 70)

    # fp16 on GPU, eval mode, English transcription (whisper_decoding.load_model, in memory only)
    processor, model, device, model_dtype = load_model(model_path)

    print("Device:", device)
    print("Model :", model_path)

    # Decoding options: temperature fallback by default; {} with --no_temperature_fallback
    # (= baseline decoding); nrng / mnt only when given
    extra_generate_kwargs = decoding_kwargs(args)

    print("Model loaded successfully.")
    print("Extra generate() options:", extra_generate_kwargs or "none (baseline decoding)")

    # ============================================================
    # EVALUATION
    # ============================================================

    results = []

    print("\n" + "=" * 70)
    print("STARTING EVALUATION")
    print("=" * 70)
    print(f"Batch size: {args.batch_size} | dtype: {model_dtype}")

    manifest_rows = manifest.to_dict("records")
    total = len(manifest_rows)
    next_progress = 50

    for start in range(0, total, args.batch_size):

        batch_rows = manifest_rows[start:start + args.batch_size]

        # --------------------------------------------------------
        # --------------------------------------------------------

        audios = []
        references = []

        for manifest_row in batch_rows:

            audio, sample_rate, answer = load_clip(manifest_row)

            audios.append(audio)
            references.append(normalize_for_evaluation(answer))

        # --------------------------------------------------------
        # Generate predictions for the whole batch (shared code:
        # processor -> fp16 features -> generate -> batch_decode)
        # --------------------------------------------------------

        predictions, predicted_ids = decode_batch(
            model, processor, audios, sample_rate, device, model_dtype,
            extra_generate_kwargs
        )

        # --------------------------------------------------------
        # Per-sample WER / CER + store results
        # --------------------------------------------------------

        for manifest_row, reference, prediction, ids in zip(batch_rows, references, predictions,
                                                             predicted_ids):

            prediction_eval = normalize_for_evaluation(prediction)
            conversation_id, speaker_id = cluster_of(manifest_row)

            row_result = {
                "sample_id": manifest_row["sample_id"],
                "dataset": manifest_row["dataset"],
                "config": manifest_row["config"],
                "row_index": int(manifest_row["row_index"]),
                "reference": reference,
                "prediction": prediction_eval,
                "WER": wer(reference, prediction_eval),
                "CER": cer(reference, prediction_eval),
                "conversation_id": conversation_id,
                "speaker_id": speaker_id,
            }
            if args.temperature_fallback:
                row_result["compression_ratio"] = token_compression_ratio(
                    ids.tolist(), model.config.vocab_size, processor.tokenizer.eos_token_id)
            results.append(row_result)

        done = start + len(batch_rows)
        if done >= next_progress or done == total:
            print(f"Processed {done}/{total}")
            next_progress = (done // 50 + 1) * 50

    # ============================================================
    # SAVE RESULTS
    # ============================================================

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        output_path,
        index=False
    )

    print("\nEvaluation results saved to:")
    print(output_path)

    # ============================================================
    # CALCULATE SUMMARY
    # ============================================================

    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)

    summary = []

    summary.append({
        "group": "Overall",
        "samples": len(results_df),
        "mean_WER": results_df["WER"].mean(),
        "mean_CER": results_df["CER"].mean(),
        "corpus_WER": corpus_metric(results_df, "WER"),
        "corpus_CER": corpus_metric(results_df, "CER"),
        "n_pred_gt3x_ref": n_runaway(results_df),
    })
    if args.temperature_fallback:
        summary[-1]["n_cr_gt_threshold"] = n_cr_gt_threshold(results_df, args.compression_ratio_threshold)

    for config in summary_groups(args.split, args.data_name, args.heldout_extra_test):

        group = results_df[
            results_df["config"] == config
        ]

        summary.append({
            "group": config,
            "samples": len(group),
            "mean_WER": group["WER"].mean(),
            "mean_CER": group["CER"].mean(),
            "corpus_WER": corpus_metric(group, "WER"),
            "corpus_CER": corpus_metric(group, "CER"),
            "n_pred_gt3x_ref": n_runaway(group),
        })
        if args.temperature_fallback:
            summary[-1]["n_cr_gt_threshold"] = n_cr_gt_threshold(group, args.compression_ratio_threshold)

    if args.model_id:
        # Pin the Hub model in the summary (fine-tuned summaries keep their columns)
        for row in summary:
            row.update(model_id=args.model_id, model_revision=args.model_revision,
                       model_commit=model_commit)

    summary_df = pd.DataFrame(summary)

    print(
        summary_df.to_string(
            index=False
        )
    )

    if args.temperature_fallback:
        print(f"\nPredictions still above compression ratio {args.compression_ratio_threshold} "
              f"after fallback: {n_cr_gt_threshold(results_df, args.compression_ratio_threshold)}"
              f" / {len(results_df)}")

    summary_df.to_csv(
        summary_path,
        index=False
    )

    print("\nSummary saved to:")
    print(summary_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
