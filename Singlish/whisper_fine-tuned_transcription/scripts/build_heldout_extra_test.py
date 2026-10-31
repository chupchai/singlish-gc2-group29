"""
build_heldout_extra_test.py

Extra unseen-speaker TEST set for a held-out build: every ASR-PART3-Train clip of the
held-out conversations (the clips preprocess_mnsc_data.py --heldout_test_conversations N
excluded from train/val), with exact duplicates of the held-out ASR-PART3-Test rows
dropped. No per-speaker cap, no duration filter, no train/val split.

The set shares its conversations and voices with the held-out PART3-Test rows, so it is
NOT independent of them. It is reported as its own group 'PART3-heldout-train-clips' and
never pooled with the official ASR-PART3-Test number. It is unseen-speaker only for the
model trained on the held-out build; older models (15k, 4k) trained on many of these clips.

Usage (from the project root):
    python scripts/build_heldout_extra_test.py --data_name part3_15k_heldout
    # -> data/part3_15k_heldout_extra_test/

Reads (never writes): data/<data_name>/heldout_test_manifest.json and
speaker_split_info.json, the first `train_shards_scanned` cached ASR-PART3-Train shards
and the ASR-PART3-Test shards (cached).

Outputs:
    data/<out_name>/test/shard-00000.parquet ...   same columns as train/val shards
                                                    (META_COLUMNS + audio_flac, FLAC PCM_16)
    data/<out_name>/extra_test_info.json            counts, duplicates, per-clip
                                                    conversation_id / voices / start / end

Loading from another script:
    from build_heldout_extra_test import load_extra_test_index
    items, info = load_extra_test_index("part3_15k_heldout_extra_test")
    audio = preprocess_mnsc_data.read_clip_audio(items[0])
"""

import argparse
import hashlib
import io
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

from preprocess_mnsc_data import (
    DATA_DIR,
    DATASET_CONFIG,
    DATASET_ID,
    HELDOUT_MANIFEST_NAME,
    INFO_NAME,
    PART3_TEST_CONFIG,
    TARGET_SR,
    ShardWriter,
    _as_dict,
    _peak_rss_mb,
    dataset_dir,
    encode_flac,
    extract_speaker_id_any,
    iter_source_rows,
    list_test_shards,
    normalize_transcript,
    row_conversation,
    row_voices,
)

EXTRA_INFO_NAME = "extra_test_info.json"
EXTRA_TEST_CONFIG = "PART3-heldout-train-clips"
EXTRA_FORMAT_VERSION = 1
EXTRA_KIND = "heldout_extra_test"
EXTRA_SPLIT_DIR = "test"
DUPLICATE_METHOD = ("segment_id (conversation_id,start,end) OR int16 PCM sha256 OR "
                    "normalize_for_evaluation(text)+round(duration,2)")


# ============================================================
# HELPERS
# ============================================================

def load_manifest(data_dir: Path) -> dict:
    path = Path(data_dir) / HELDOUT_MANIFEST_NAME
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Build the held-out set first: python scripts/preprocess_mnsc_data.py "
            f"--data_name {Path(data_dir).name} --heldout_test_conversations 40 --max_source_shards 30")
    return json.loads(path.read_text(encoding="utf-8"))


def segment_key(attrs) -> Optional[Tuple[str, str, str]]:
    """(conversation_id, start, end) as stripped strings, or None if start/end is empty."""
    a = _as_dict(attrs)
    start = str(a.get("start") or "").strip()
    end = str(a.get("end") or "").strip()
    if not start or not end:
        return None
    return row_conversation(a), start, end


def pcm_sha256(audio_int16: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(audio_int16, dtype="<i2").tobytes()).hexdigest()


def decode_int16(audio_bytes: bytes) -> Tuple[np.ndarray, int]:
    """int16 mono samples + sampling rate (same decode as preprocess collect_records)."""
    audio, sr = sf.read(io.BytesIO(audio_bytes), dtype="int16")
    if audio.ndim > 1:
        audio = np.round(audio.astype(np.float32).mean(axis=1)).astype(np.int16)
    return audio, int(sr)


def _normalize_for_evaluation(text: str) -> str:
    # Lazy import: the eval script is the single source of truth for WER normalization.
    from evaluate_whisper_speaker_split import normalize_for_evaluation
    return normalize_for_evaluation(text)


def text_key(answer: str, duration_sec: float) -> Tuple[str, float]:
    return _normalize_for_evaluation(answer or ""), round(float(duration_sec), 2)


def _row_text_keys(row: dict, duration_sec: float) -> List[Tuple[str, float]]:
    """Text keys of a source row: its answer and, if present, other_attributes.Transcription
    (PART3-Train stores a punctuated answer next to the raw transcription)."""
    keys = [text_key(row.get("answer") or "", duration_sec)]
    raw = _as_dict(row.get("other_attributes")).get("Transcription")
    if raw:
        keys.append(text_key(raw, duration_sec))
    return [k for k in keys if k[0]]


def _interval(attrs) -> Optional[Tuple[float, float]]:
    a = _as_dict(attrs)
    try:
        return float(a.get("start")), float(a.get("end"))
    except (TypeError, ValueError):
        return None


def _test_shard_paths(source_dir: Optional[Path]) -> List[str]:
    shards = list_test_shards(PART3_TEST_CONFIG, source_dir)
    if source_dir is not None:
        return shards
    from huggingface_hub import hf_hub_download
    return [hf_hub_download(DATASET_ID, s, repo_type="dataset") for s in shards]


def heldout_test_keys(manifest: dict, source_dir: Optional[Path] = None) -> dict:
    """Duplicate keys of the held-out PART3-Test rows listed in the manifest."""
    wanted = set(int(i) for i in manifest["part3_test_row_indices"])
    convs = set(manifest["conversations"])
    keys = {"segments": {}, "audio": {}, "text": {}, "intervals": {}, "n_rows": 0}
    recomputed = []
    gi = 0
    for path in _test_shard_paths(source_dir):
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=32, columns=["context", "answer", "other_attributes"]):
            for row in batch.to_pylist():
                attrs = row.get("other_attributes")
                if row_conversation(attrs) in convs:
                    recomputed.append(gi)
                if gi in wanted:
                    seg = segment_key(attrs)
                    if seg is not None:
                        keys["segments"].setdefault(seg, gi)
                    ctx = row.get("context") or {}
                    data = ctx.get("bytes") if isinstance(ctx, dict) else None
                    if data:
                        audio, sr = decode_int16(data)
                        keys["audio"].setdefault(pcm_sha256(audio), gi)
                        for k in _row_text_keys(row, len(audio) / sr):
                            keys["text"].setdefault(k, gi)
                    iv = _interval(attrs)
                    if iv is not None:
                        keys["intervals"].setdefault(row_conversation(attrs), []).append((iv[0], iv[1], gi))
                    keys["n_rows"] += 1
                gi += 1
        del pf
    if recomputed != sorted(wanted):
        raise ValueError("held-out PART3-Test rows recomputed from the conversations do not match "
                         f"{HELDOUT_MANIFEST_NAME} ({len(recomputed)} vs {len(wanted)})")
    return keys


def duplicate_reasons(seg_key, sha: Optional[str], txt_keys: Iterable, keys: dict) -> List[str]:
    """Subset of ['segment_id', 'audio_sha256', 'text_duration'] that match a held-out test row."""
    out = []
    if seg_key is not None and seg_key in keys["segments"]:
        out.append("segment_id")
    if sha is not None and sha in keys["audio"]:
        out.append("audio_sha256")
    if any(k in keys["text"] for k in txt_keys):
        out.append("text_duration")
    return out


def _first_match(seg_key, sha, txt_keys, keys) -> Optional[int]:
    if seg_key is not None and seg_key in keys["segments"]:
        return keys["segments"][seg_key]
    if sha is not None and sha in keys["audio"]:
        return keys["audio"][sha]
    for k in txt_keys:
        if k in keys["text"]:
            return keys["text"][k]
    return None


def overlapping_rows(attrs, keys: dict) -> List[int]:
    """Held-out test rows of the same conversation whose [start, end) overlaps this row."""
    iv = _interval(attrs)
    if iv is None:
        return []
    s1, e1 = iv
    return [gi for s2, e2, gi in keys["intervals"].get(row_conversation(attrs), [])
            if s1 < e2 and s2 < e1]


# ============================================================
# BUILD
# ============================================================

def build_extra_test(manifest: dict, writer: ShardWriter, source_dir: Optional[Path],
                     max_shards: int, keys: dict) -> Tuple[List[dict], dict]:
    """Stream ASR-PART3-Train, write every held-out-conversation clip that is not a duplicate."""
    convs = set(manifest["conversations"])
    timing: dict = {"download_s": 0.0, "shards_used": {}}
    clips: List[dict] = []
    stats = {"n_heldout_conversation_rows_seen": 0, "skipped_audio": 0, "skipped_text": 0,
             "duplicates": [], "n_near_duplicate_clips": 0}
    t0 = time.time()
    rows = iter_source_rows(DATASET_CONFIG, source_dir, timing, max_shards)
    try:
        for shard_idx, row_idx, row in rows:
            attrs = row.get("other_attributes")
            conv = row_conversation(attrs)
            if conv not in convs:
                continue
            stats["n_heldout_conversation_rows_seen"] += 1
            clip_id = f"{DATASET_CONFIG}/{shard_idx:05d}:{row_idx}"

            ctx = row.get("context") or {}
            data = ctx.get("bytes") if isinstance(ctx, dict) else None
            if not data:
                stats["skipped_audio"] += 1
                continue
            try:
                audio, sr = decode_int16(data)
            except Exception as exc:
                print(f"  Audio decode error {clip_id} (skipping): {exc}")
                stats["skipped_audio"] += 1
                continue
            if sr != TARGET_SR:
                print(f"  Unexpected sample rate {sr} Hz {clip_id} -- skipping")
                stats["skipped_audio"] += 1
                continue
            duration = len(audio) / sr

            text = normalize_transcript(row.get("answer") or "")
            if _normalize_for_evaluation(text) == "":
                stats["skipped_text"] += 1
                continue

            a = _as_dict(attrs)
            seg = segment_key(a)
            sha = pcm_sha256(audio)
            txt = _row_text_keys(row, duration)
            reasons = duplicate_reasons(seg, sha, txt, keys)
            if reasons:
                stats["duplicates"].append({
                    "clip_id": clip_id, "conversation_id": conv,
                    "start": str(a.get("start") or ""), "end": str(a.get("end") or ""),
                    "duration_sec": round(duration, 3), "reasons": reasons,
                    "test_row_index": _first_match(seg, sha, txt, keys),
                })
                continue

            speaker_id = extract_speaker_id_any(row)
            writer.add({
                "clip_id": clip_id, "source_config": DATASET_CONFIG,
                "audio_flac": encode_flac(audio, sr), "audio_length": int(len(audio)),
                "sampling_rate": int(sr), "text": text, "speaker_id": speaker_id,
                "duration_sec": float(duration),
            })
            overlaps = overlapping_rows(a, keys)
            if overlaps:
                stats["n_near_duplicate_clips"] += 1
            clips.append({
                "row_index": len(clips), "clip_id": clip_id, "conversation_id": conv,
                "speaker_id": speaker_id, "voices": row_voices(a),
                "start": str(a.get("start") or ""), "end": str(a.get("end") or ""),
                "duration_sec": float(duration), "overlaps_test_rows": overlaps,
            })
            if len(clips) % 250 == 0:
                print(f"  Kept {len(clips):,} clips ({time.time() - t0:,.0f} s, "
                      f"peak RSS {_peak_rss_mb()} MB)")
    finally:
        rows.close()
    stats["source_shards_used"] = timing["shards_used"].get(DATASET_CONFIG, 0)
    stats["download_s"] = round(timing["download_s"], 1)
    stats["process_s"] = round(time.time() - t0 - timing["download_s"], 1)
    return clips, stats


def _prepare_extra_dir(out_dir: Path, source_dir_path: Path, overwrite: bool) -> Optional[str]:
    """Error message if out_dir cannot be (re)used; deletes it only if it is ours."""
    out_dir, source_dir_path = Path(out_dir).resolve(), Path(source_dir_path).resolve()
    if out_dir == source_dir_path:
        return f"--out_name must differ from the held-out build {source_dir_path.name}"
    if not out_dir.exists() or not any(out_dir.iterdir()):
        return None
    if (out_dir / INFO_NAME).exists():
        return (f"Refusing to use {out_dir}: it contains {INFO_NAME} (a train/val build). "
                "Pick another --out_name.")
    if not overwrite:
        return f"{out_dir} already exists and is not empty. Pick another --out_name or pass --overwrite."
    contents = {p.name for p in out_dir.iterdir()}
    ours = False
    info_path = out_dir / EXTRA_INFO_NAME
    if info_path.exists():
        try:
            ours = json.loads(info_path.read_text(encoding="utf-8")).get("kind") == EXTRA_KIND
        except Exception:
            ours = False
    elif contents == {EXTRA_SPLIT_DIR}:
        ours = True  # leftover of an interrupted run
    if not ours:
        return (f"Refusing to delete {out_dir}: it was not created by this script "
                f"(no {EXTRA_INFO_NAME} with kind {EXTRA_KIND}).")
    shutil.rmtree(out_dir)
    return None


# ============================================================
# LOADER (public API for the eval script)
# ============================================================

def load_extra_test_index(data_name: str, data_root: Path = DATA_DIR) -> Tuple[List[dict], dict]:
    """Lazy per-clip index (no audio), like preprocess_mnsc_data.load_preprocessed_index,
    plus conversation_id and voices. Use read_clip_audio(item) to decode one clip."""
    out_dir = dataset_dir(data_name, Path(data_root))
    info_path = out_dir / EXTRA_INFO_NAME
    if not info_path.exists():
        raise FileNotFoundError(
            f"{info_path} not found. Build it with: python scripts/build_heldout_extra_test.py "
            f"--data_name <held-out build> --out_name {data_name}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    items: List[dict] = []
    for name in info["layout"]["shards"]:
        path = out_dir / EXTRA_SPLIT_DIR / name
        pf = pq.ParquetFile(str(path))
        cols = pf.read(columns=["clip_id", "text", "speaker_id", "duration_sec",
                                "sampling_rate", "audio_length"]).to_pydict()
        positions = []
        for rg in range(pf.metadata.num_row_groups):
            positions += [(rg, j) for j in range(pf.metadata.row_group(rg).num_rows)]
        for i, (rg, j) in enumerate(positions):
            items.append({
                "shard": str(path), "row_group": rg, "row_in_group": j,
                "clip_id": cols["clip_id"][i], "text": cols["text"][i],
                "speaker_id": cols["speaker_id"][i],
                "duration_sec": float(cols["duration_sec"][i]),
                "sampling_rate": int(cols["sampling_rate"][i]),
                "audio_length": int(cols["audio_length"][i]),
            })
        del pf
    clips = info["clips"]
    assert len(items) == len(clips) == int(info["n_clips"]), \
        f"extra test clips {len(items)} != info {len(clips)} / {info['n_clips']}"
    for item, clip in zip(items, clips):
        assert item["clip_id"] == clip["clip_id"], f"clip order mismatch at {clip['row_index']}"
        item["conversation_id"] = clip["conversation_id"]
        item["voices"] = list(clip["voices"])
    return items, info


# ============================================================
# CLI
# ============================================================

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build the extra unseen-speaker test set (all PART3-Train clips of the held-out "
                    "conversations) from a held-out build.")
    p.add_argument("--data_name", default="part3_15k_heldout",
                   help=f"Held-out build under --data_root with {HELDOUT_MANIFEST_NAME} "
                        "(default part3_15k_heldout). Read only.")
    p.add_argument("--data_root", default=str(DATA_DIR),
                   help="Parent folder of the datasets (default <project>/data).")
    p.add_argument("--out_name", default=None,
                   help="Output folder under --data_root (default <data_name>_extra_test).")
    p.add_argument("--max_source_shards", type=int, default=0,
                   help="Source shards to read (default 0 = the manifest's train_shards_scanned). "
                        "Must not exceed train_shards_scanned.")
    p.add_argument("--source_dir", default=None,
                   help="Local mirror with <config>/*.parquet instead of the HF Hub (tests).")
    p.add_argument("--rows_per_shard", type=int, default=512,
                   help="Clips per output parquet shard (default 512).")
    p.add_argument("--row_group_size", type=int, default=8,
                   help="Clips per parquet row group (default 8).")
    p.add_argument("--overwrite", action="store_true",
                   help="Replace an existing extra test folder created by this script.")
    args = p.parse_args(argv)
    if args.out_name is None:
        args.out_name = f"{args.data_name}_extra_test"
    return args


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    t0 = time.time()
    data_root = Path(args.data_root)
    try:
        data_dir = dataset_dir(args.data_name, data_root)
        out_dir = dataset_dir(args.out_name, data_root)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 2
    try:
        manifest = load_manifest(data_dir)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        return 2
    scanned = int(manifest.get("train_shards_scanned") or 0)
    if args.max_source_shards < 0 or (scanned and args.max_source_shards > scanned):
        print(f"ERROR: --max_source_shards {args.max_source_shards} must be in [0, {scanned}] "
              "(conversations from unscanned shards were never excluded from training).")
        return 2
    max_shards = args.max_source_shards or scanned or None
    err = _prepare_extra_dir(out_dir, data_dir, args.overwrite)
    if err:
        print(f"ERROR: {err}")
        return 2

    src_info_path = data_dir / INFO_NAME
    src_info = json.loads(src_info_path.read_text(encoding="utf-8")) if src_info_path.exists() else {}
    expected = (src_info.get("collection") or {}).get("skipped_heldout_conversation")
    source_dir = Path(args.source_dir) if args.source_dir else None

    print("=" * 70)
    print("EXTRA HELD-OUT TEST SET")
    print("=" * 70)
    print(f"Held-out build   : {data_dir}")
    print(f"Output           : {out_dir}")
    print(f"Conversations    : {len(manifest['conversations'])}  voices: {len(manifest['voices'])}")
    print(f"Source shards    : {max_shards}  ({source_dir or 'HF cache'})")
    print()
    print("Reading held-out PART3-Test rows (duplicate keys)...")
    keys = heldout_test_keys(manifest, source_dir)
    print(f"  {keys['n_rows']} rows: {len(keys['segments'])} segment ids, {len(keys['audio'])} "
          f"audio hashes, {len(keys['text'])} text keys")

    writer = ShardWriter(out_dir / EXTRA_SPLIT_DIR, "shard", args.rows_per_shard, args.row_group_size)
    clips, stats = build_extra_test(manifest, writer, source_dir, max_shards, keys)
    shards = writer.close()
    if not clips:
        print("ERROR: 0 clips kept; nothing written.")
        shutil.rmtree(out_dir, ignore_errors=True)
        return 1

    conv_counts: Dict[str, int] = {}
    voices = set()
    for c in clips:
        conv_counts[c["conversation_id"]] = conv_counts.get(c["conversation_id"], 0) + 1
        voices.update(c["voices"])
    hours = sum(c["duration_sec"] for c in clips) / 3600
    dups = stats["duplicates"]
    info = {
        "format_version": EXTRA_FORMAT_VERSION,
        "kind": EXTRA_KIND,
        "config": EXTRA_TEST_CONFIG,
        "source_data_name": args.data_name,
        "dataset_id": DATASET_ID,
        "dataset_config": DATASET_CONFIG,
        "source_manifest": HELDOUT_MANIFEST_NAME,
        "selection_seed": manifest.get("seed"),
        "max_source_shards": max_shards,
        "source_shards_used": stats["source_shards_used"],
        "no_speaker_cap": True,
        "n_heldout_conversation_rows_seen": stats["n_heldout_conversation_rows_seen"],
        "n_clips": len(clips),
        "total_hours": round(hours, 3),
        "n_conversations": len(conv_counts),
        "n_voices": len(voices),
        "conversations": sorted(conv_counts),
        "voices": sorted(voices),
        "clips_per_conversation": dict(sorted(conv_counts.items())),
        "duplicates": {"n_dropped": len(dups), "method": DUPLICATE_METHOD, "dropped": dups},
        "n_near_duplicate_clips": stats["n_near_duplicate_clips"],
        "skipped_audio": stats["skipped_audio"],
        "skipped_text": stats["skipped_text"],
        "expected_excluded_by_build": expected,
        "heldout_test_rows": keys["n_rows"],
        "audio_format": {"codec": "flac", "subtype": "PCM_16", "sampling_rate": TARGET_SR, "channels": 1},
        "layout": {"rows_per_shard": args.rows_per_shard, "row_group_size": args.row_group_size,
                   "shards": shards},
        "clips": clips,
        "timing": {"download_s": stats["download_s"], "process_s": stats["process_s"],
                   "total_s": round(time.time() - t0, 1)},
        "peak_rss_mb": _peak_rss_mb(),
    }
    info_path = out_dir / EXTRA_INFO_NAME
    with open(info_path, "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=2)
    size_gb = sum(p.stat().st_size for p in out_dir.rglob("*") if p.is_file()) / 1e9

    reasons: Dict[str, int] = {}
    for d in dups:
        key = "+".join(d["reasons"])
        reasons[key] = reasons.get(key, 0) + 1
    print()
    print("=" * 70)
    print("EXTRA TEST SET COMPLETE")
    print("=" * 70)
    print(f"Output        : {out_dir}  ({size_gb:.2f} GB, {len(shards)} shards)")
    print(f"Clips         : {len(clips):,}  ({hours:.2f} h)")
    print(f"Conversations : {len(conv_counts)}   voices: {len(voices)}")
    print(f"Rows seen     : {stats['n_heldout_conversation_rows_seen']:,} "
          f"(build excluded {expected if expected is not None else 'n/a'})")
    print(f"Duplicates    : {len(dups)} dropped {reasons or ''}")
    print(f"Near-dup clips: {stats['n_near_duplicate_clips']} (time overlap with a held-out test row; kept)")
    print(f"Skipped       : audio {stats['skipped_audio']}, empty text {stats['skipped_text']}")
    print(f"Timing        : {info['timing']}")
    print(f"Peak RSS      : {info['peak_rss_mb']} MB")
    print(f"Info          : {info_path}")
    if expected is not None and stats["n_heldout_conversation_rows_seen"] != expected:
        print("WARNING: rows seen != rows the held-out build excluded; check --max_source_shards.")
    print()
    print("Next:")
    print(f"  python scripts/audit_test_speaker_overlap.py --data_name {args.data_name} "
          f"--extra_test {args.out_name}")
    print(f"  python scripts/evaluate_whisper_speaker_split.py --run_name <RUN> "
          f"--heldout_extra_test {args.out_name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
