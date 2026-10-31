"""Prepare Singaporean clips from MNSC PART1-Train and PART2-Train.

Downloads only needed audio; saves 16 kHz, 3–6 s FLAC-in-parquet shards.
"""

import argparse
import hashlib
import json
import math
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from cv_utils import (
    AUDIO_PROCESSING,
    DATA_DIR,
    SHARD_ROWS,
    SUBSET_SCHEMA,
    TARGET_SR,
    ascii_safe,
    decode_audio_bytes,
    encode_flac,
    ensure_free_space,
    preprocess_waveform,
    subset_paths,
    write_subset_shards,
)
from extract_common_voice_subset import (
    assign_splits,
    print_plan,
    select_clips,
    summarise,
)


SOURCE = "mnsc"
SUBSET_NAME = "mnsc_subset"
ACCENT_GROUP = "Singaporean"
ID_PREFIX = "mnsc_"
HF_REPO_DEFAULT = "AudioLLMs/Multitask-National-Speech-Corpus-v1-extend"
DEFAULT_CONFIGS = ["ASR-PART1-Train", "ASR-PART2-Train"]
EXTRA_MANIFEST = Path(__file__).with_name("mnsc_part1_extra.csv")
MIN_SPEAKERS_WARNING = 300
RANGE_READ_BLOCK_SIZE = 256 * 1024
SPEAKER_KEYS = ("speaker1", "speaker2", "speaker")
EMPTY_IDS = ("", "none", "null", "nan")
LICENSE_NOTE = (
    "MNSC is derived from IMDA Singapore's National Speech Corpus "
    "(https://www.imda.gov.sg/how-we-can-help/national-speech-corpus) and "
    f"redistributed on Hugging Face by AudioLLMs ({HF_REPO_DEFAULT}). The HF "
    "card states no license; check IMDA's NSC terms. Used here for a "
    "non-commercial school project."
)
TEST_CONFIGS_POLICY = (
    "ASR-*-Test configs are excluded by default to keep evaluation data separate "
    "(pass --allow_test_configs to override)."
)


LAST_STATS: Dict = {}


class SourceError(Exception):
    """The requested MNSC source cannot be used."""


@dataclass
class ShardRef:
    config: str
    name: str                     # path in repo, e.g. ASR-PART1-Train/train-00000-of-00772.parquet
    shard_idx: int
    size: int
    local_path: Optional[Path]


# ============================================================
# PURE HELPERS
# ============================================================

def normalize_transcript(text) -> str:
    """Remove speaker labels and annotation brackets; lowercase and collapse whitespace."""
    if not text:
        return ""
    text = str(text)
    text = re.sub(r"<Speaker\d+>:\s*", "", text, flags=re.IGNORECASE)
    while re.search(r"\[[^\[\]]+\]", text):
        text = re.sub(r"\[([^\[\]]+)\]", r"\1", text)
    text = re.sub(r"\{([^{}\[\]]+)\}", r"\1", text)
    text = re.sub(r"\(([^()]*)\)", r"\1", text)
    text = re.sub(r"#([^#]+)#", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


def transcript_key(text):
    return " ".join(re.findall(r"\w+(?:'\w+)?", normalize_transcript(text)))


def _as_dict(value) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, (str, bytes)):
        try:
            parsed = json.loads(value)
        except (ValueError, TypeError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def extract_speaker_id(other_attributes) -> str:
    """speaker1 -> speaker2 -> speaker (PART1); "" when none is usable."""
    attrs = _as_dict(other_attributes)
    for key in SPEAKER_KEYS:
        sid = _as_dict(attrs.get(key)).get("speaker_id")
        if sid is None:
            continue
        sid = str(sid).strip()
        if sid.lower() not in EMPTY_IDS:
            return sid
    return ""


def duration_from_attrs(other_attributes) -> float:
    """end - start (PART3) when both parse and are positive, else NaN."""
    attrs = _as_dict(other_attributes)
    try:
        d = float(attrs.get("end")) - float(attrs.get("start"))
    except (TypeError, ValueError):
        return float("nan")
    return d if d > 0 else float("nan")


def metadata_columns(schema: pa.Schema) -> List[str]:
    """Small columns needed to plan the selection (never the audio)."""
    cols = ["answer"] if "answer" in schema.names else []
    if "other_attributes" not in schema.names:
        return cols
    t = schema.field("other_attributes").type
    if not pa.types.is_struct(t):
        return cols + ["other_attributes"]          # JSON string column
    names = [t.field(i).name for i in range(t.num_fields)]
    for key in SPEAKER_KEYS:
        if key not in names:
            continue
        sub = t.field(key).type
        if pa.types.is_struct(sub):
            if sub.get_field_index("speaker_id") >= 0:
                cols.append(f"other_attributes.{key}.speaker_id")
        else:
            cols.append(f"other_attributes.{key}")  # JSON string leaf
    for key in ("start", "end"):
        if key in names:
            cols.append(f"other_attributes.{key}")
    return cols


def accent_raw_for(config: str) -> str:
    m = re.search(r"PART\d+", config, flags=re.IGNORECASE)
    return f"Singaporean (MNSC {m.group(0).upper() if m else config})"


def config_slug(config: str) -> str:
    return config.lower().removeprefix("asr-")


def clip_id_for(config: str, shard_idx: int, row: int) -> str:
    return f"{ID_PREFIX}{config_slug(config)}_{shard_idx:05d}_{row:05d}"


def shard_index_from_name(name: str, fallback: int) -> int:
    m = re.search(r"-(\d+)-of-\d+\.parquet$", name)
    return int(m.group(1)) if m else fallback


def is_test_config(name: str) -> bool:
    return name.lower().endswith("-test")


def pool_capacity(counts: Dict[str, int], cap: int) -> int:
    return sum(min(n, cap) for n in counts.values())


def scan_should_stop(capacity: int, target: int, new_speakers: int,
                     total_speakers: int, saturation: float,
                     scanned: int, max_shards: int) -> Tuple[bool, str]:
    saturated = total_speakers > 0 and (new_speakers / total_speakers) < saturation
    if capacity >= target and saturated:
        return True, "target+saturated"
    if scanned >= max_shards:
        return True, "max_shards"
    return False, ""


# ============================================================
# SHARD LISTING
# ============================================================

def _interleave(per_config: List[List[ShardRef]]) -> List[ShardRef]:
    out: List[ShardRef] = []
    for i in range(max((len(c) for c in per_config), default=0)):
        out.extend(c[i] for c in per_config if i < len(c))
    return out


def list_local_shards(source_dir: Path, configs: List[str]) -> List[ShardRef]:
    per_config = []
    for config in configs:
        files = sorted((Path(source_dir) / config).glob("*.parquet"))
        if not files:
            raise SourceError(f"No parquet files in {Path(source_dir) / config}.")
        refs = [ShardRef(config, f"{config}/{f.name}", shard_index_from_name(f.name, i),
                         f.stat().st_size, f) for i, f in enumerate(files)]
        print(f"  {config}: {len(refs)} shard(s), {sum(r.size for r in refs) / 1e9:.3f} GB (local)")
        per_config.append(refs)
    return _interleave(per_config)


def hf_dataset_info(repo: str, revision: str) -> Tuple[str, str]:
    """(resolved sha, card license or 'not stated on the card')."""
    from huggingface_hub import HfApi
    info = HfApi().dataset_info(repo, revision=revision)
    card = info.card_data
    lic = None
    if card is not None:
        try:
            lic = card.to_dict().get("license")
        except Exception:
            lic = getattr(card, "license", None)
    return info.sha, (str(lic) if lic else "not stated on the card")


def list_hf_shards(repo: str, sha: str, configs: List[str],
                   cache_dir: Optional[Path]) -> Tuple[List[ShardRef], Dict[str, dict]]:
    from huggingface_hub import HfApi, try_to_load_from_cache
    from huggingface_hub.hf_api import RepoFile

    api = HfApi()
    per_config, sizes = [], {}
    for config in configs:
        try:
            files = sorted(
                (f for f in api.list_repo_tree(repo, path_in_repo=config, repo_type="dataset",
                                               revision=sha)
                 if isinstance(f, RepoFile) and f.path.endswith(".parquet")),
                key=lambda f: f.path)
        except Exception:
            files = []
        if not files:
            top = api.list_repo_tree(repo, repo_type="dataset", revision=sha)
            asr = sorted(f.path for f in top if f.path.startswith("ASR-"))
            raise SourceError(f"Config {config!r} has no parquet files in {repo}. "
                              f"ASR configs: {asr}")
        refs = []
        for i, f in enumerate(files):
            cached = try_to_load_from_cache(repo, f.path, cache_dir=cache_dir,
                                            revision=sha, repo_type="dataset")
            refs.append(ShardRef(config, f.path,
                                 shard_index_from_name(f.path, i), int(f.size or 0),
                                 Path(cached) if isinstance(cached, str) else None))
        total = sum(r.size for r in refs)
        n_cached = sum(r.local_path is not None for r in refs)
        print(f"  {config}: {len(refs)} shards, {total / 1e9:.2f} GB total, {n_cached} cached "
              "(only the needed shards are downloaded)")
        sizes[config] = {"n_shards": len(refs), "total_bytes": total, "cached": n_cached}
        per_config.append(refs)
    return _interleave(per_config), sizes


# ============================================================
# METADATA SCAN
# ============================================================

def _open_parquet(ref: ShardRef, repo: str, sha: str, fs_holder: Dict):
    if ref.local_path is not None:
        return pq.ParquetFile(str(ref.local_path)), None
    if "fs" not in fs_holder:
        from huggingface_hub import HfFileSystem
        fs_holder["fs"] = HfFileSystem()
    fh = fs_holder["fs"].open(f"datasets/{repo}@{sha}/{ref.name}", "rb",
                              block_size=RANGE_READ_BLOCK_SIZE)
    return pq.ParquetFile(fh), fh


def scan_metadata(shards: List[ShardRef], args: argparse.Namespace, repo: str,
                  sha: str, stats: Dict) -> pd.DataFrame:
    target = int(math.ceil(args.clips_total * args.oversample))
    counts: Dict[str, int] = {}
    seen_pairs: set = set()
    meta_rows: List[dict] = []
    fs_holder: Dict = {}
    t0 = time.time()
    for k, ref in enumerate(shards, start=1):
        t_shard = time.time()
        pf, fh = _open_parquet(ref, repo, sha, fs_holder)
        try:
            cols = metadata_columns(pf.schema_arrow)
            records = pf.read(columns=cols).to_pylist()
        finally:
            if fh is not None:
                fh.close()
        before = len(counts)
        kept = 0
        accent_raw = accent_raw_for(ref.config)
        for row, r in enumerate(records):
            stats["rows_scanned"] += 1
            attrs = r.get("other_attributes")
            raw_id = extract_speaker_id(attrs)
            if not raw_id:
                stats["no_speaker_id"] += 1
                continue
            sentence = normalize_transcript(r.get("answer"))
            if not sentence:
                stats["empty_transcript"] += 1
                continue
            pair = (raw_id, transcript_key(sentence))
            if pair in seen_pairs:
                stats["duplicate_utterance"] += 1
                continue
            seen_pairs.add(pair)
            counts[raw_id] = counts.get(raw_id, 0) + 1
            kept += 1
            meta_rows.append({
                "clip_id":        clip_id_for(ref.config, ref.shard_idx, row),
                "speaker_id":     f"{ID_PREFIX}{raw_id}",
                "raw_speaker_id": raw_id,
                "accent_group":   ACCENT_GROUP,
                "accent_raw":     accent_raw,
                "sentence":       sentence,
                "duration_sec":   duration_from_attrs(attrs),
                "shard_key":      ref.name,
                "row":            row,
            })
        stats["shards_scanned"].append(ref.name)
        new = len(counts) - before
        capacity = pool_capacity(counts, args.max_clips_per_speaker)
        print(f"  shard {k} ({ascii_safe(ref.name)}): {len(records):,} rows, kept {kept:,}, "
              f"speakers {len(counts):,} (+{new:,}), capacity {capacity:,}/{target:,}, "
              f"{time.time() - t_shard:.1f} s")
        stop, reason = scan_should_stop(capacity, target, new, len(counts),
                                        args.saturation, k, args.max_shards)
        if stop:
            stats["stop_reason"] = reason
            break
    else:
        stats["stop_reason"] = "all_shards_scanned"
    stats["speakers_seen"] = len(counts)
    stats["scan_sec"] = round(time.time() - t0, 1)
    return pd.DataFrame(meta_rows).reset_index(drop=True)


# ============================================================
# DOWNLOAD + PROCESSING
# ============================================================

def download_needed(needed: List[ShardRef], repo: str, sha: str,
                    cache_dir: Optional[Path], no_download: bool, stats: Dict) -> None:
    missing = [r for r in needed if r.local_path is None]
    stats["download_bytes"] += sum(r.size for r in missing)
    if not missing:
        print("All needed shards are already in the local HF cache.")
        return
    if no_download:
        raise SourceError("Shards not cached yet (and --no_download): "
                          f"{[r.name for r in missing]}")
    from huggingface_hub import constants, hf_hub_download
    cache_root = Path(cache_dir) if cache_dir else Path(constants.HF_HUB_CACHE)
    total = sum(r.size for r in missing)
    print(f"Downloading {len(missing)} shard(s), {total / 1e9:.2f} GB -> {cache_root}")
    ensure_free_space(cache_root, total)
    for r in missing:
        path = hf_hub_download(repo, r.name, repo_type="dataset", revision=sha,
                               cache_dir=str(cache_dir) if cache_dir else None)
        r.local_path = Path(path)
        print(f"  downloaded {r.name}")


def process_selected(selected: pd.DataFrame, shards: Dict[str, ShardRef],
                     args: argparse.Namespace, stats: Dict) -> List[dict]:
    rows: List[dict] = []
    t0 = time.time()
    for shard_key, g in selected.groupby("shard_key", sort=False):
        wanted = {int(r["row"]): r for r in g.to_dict("records")}
        pf = pq.ParquetFile(str(shards[shard_key].local_path))
        offset = 0
        for batch in pf.iter_batches(columns=["context"], batch_size=64):
            for i, ctx in enumerate(batch.column("context").to_pylist()):
                info = wanted.get(offset + i)
                if info is None:
                    continue
                data = (ctx or {}).get("bytes")
                if not data:
                    stats["skips"]["missing_audio"] += 1
                    continue
                try:
                    x, sr = decode_audio_bytes(data)
                    y, reason = preprocess_waveform(x, sr, args.min_duration, args.max_duration)
                except Exception as exc:
                    print(f"  decode error ({ascii_safe(info['clip_id'])}): {ascii_safe(exc)}")
                    y, reason = None, "decode_error"
                if y is None:
                    stats["skips"][reason] += 1
                    continue
                rows.append({
                    "clip_id":        info["clip_id"],
                    "source":         SOURCE,
                    "speaker_id":     info["speaker_id"],
                    "accent_raw":     info["accent_raw"],
                    "accent_group":   ACCENT_GROUP,
                    "is_singaporean": True,
                    "duration_sec":   float(len(y) / TARGET_SR),
                    "sentence":       info["sentence"],
                    "sampling_rate":  TARGET_SR,
                    "audio_flac":     encode_flac(y),
                    "selection_rank": int(info["selection_rank"]),
                    "raw_speaker_id": info["raw_speaker_id"],
                })
                if len(rows) % 500 == 0:
                    print(f"  kept {len(rows):,} clips ({time.time() - t0:.0f} s)")
            offset += batch.num_rows
        print(f"  processed {ascii_safe(shard_key)}: {len(wanted):,} selected rows")
    return rows


def load_extension(args, shards, sha):
    if not args.extra_manifest or "ASR-PART1-Train" not in args.configs:
        return []
    manifest = pd.read_csv(args.extra_manifest, dtype={"speaker_id": str})
    if manifest.clip_id.duplicated().any():
        raise SourceError("Duplicate PART1 addition IDs.")
    files = sorted(args.part1_extension.glob("part-*.parquet"))
    if files:
        rows = [r for f in files for r in pq.read_table(f, schema=SUBSET_SCHEMA).to_pylist()]
    else:
        rows = []
        for shard, group in manifest.groupby("source_shard", sort=True):
            ref = shards.get(shard)
            if ref is None:
                raise SourceError(f"Extension shard unavailable: {shard}")
            if args.no_download and ref.local_path is None:
                raise SourceError("PART1 additions are not cached; rerun without --no_download.")
            revisions = group.source_revision.unique()
            if len(revisions) != 1 or (sha is not None and revisions[0] != sha):
                raise SourceError("PART1 addition revision differs from --hf_revision.")
            pf, fh = _open_parquet(ref, args.hf_repo, sha, {})
            try:
                for index, entries in group.groupby("source_row_group", sort=True):
                    records = pf.read_row_group(int(index), columns=["context", "answer", "other_attributes"]).to_pylist()
                    for entry in entries.itertuples(index=False):
                        record = records[int(entry.row_in_group)]
                        speaker = extract_speaker_id(record["other_attributes"])
                        if f"mnsc_{speaker}" != entry.speaker_id or transcript_key(record["answer"]) != transcript_key(entry.sentence):
                            raise SourceError(f"PART1 addition identity mismatch: {entry.clip_id}")
                        x, sr = decode_audio_bytes(record["context"]["bytes"])
                        y, reason = preprocess_waveform(x, sr, args.min_duration, args.max_duration)
                        if y is None:
                            raise SourceError(f"PART1 addition {entry.clip_id}: {reason}")
                        rows.append({"clip_id": entry.clip_id, "source": SOURCE,
                                     "speaker_id": entry.speaker_id, "accent_raw": accent_raw_for("ASR-PART1-Train"),
                                     "accent_group": ACCENT_GROUP, "is_singaporean": True,
                                     "duration_sec": len(y) / TARGET_SR, "sentence": normalize_transcript(entry.sentence),
                                     "sampling_rate": TARGET_SR, "audio_flac": encode_flac(y), "split": "train"})
            finally:
                if fh is not None:
                    fh.close()
            print(f"  Prepared {len(rows):,}/{len(manifest):,} PART1 additions", flush=True)
        write_subset_shards(pa.Table.from_pylist(rows, schema=SUBSET_SCHEMA), args.part1_extension)
    if len(rows) != len(manifest) or {r["clip_id"] for r in rows} != set(manifest.clip_id):
        raise SourceError("Prepared PART1 additions differ from the manifest.")
    for row in rows:
        if (row["source"] != SOURCE or row["accent_raw"] != accent_raw_for("ASR-PART1-Train")
                or row["sampling_rate"] != TARGET_SR
                or not args.min_duration <= row["duration_sec"] <= args.max_duration):
            raise SourceError("Invalid prepared PART1 addition.")
        row["raw_speaker_id"] = row["speaker_id"].removeprefix(ID_PREFIX)
        row["sentence"] = normalize_transcript(row["sentence"])
    return rows


def choose_rows(extra, rows, target, cap, seen_pairs, seen_audio, counts):
    chosen = []
    for row in extra + sorted(rows, key=lambda r: r["selection_rank"]):
        pair = (row["speaker_id"], transcript_key(row["sentence"]))
        audio = hashlib.sha256(decode_audio_bytes(row["audio_flac"])[0].tobytes()).digest()
        if pair in seen_pairs or audio in seen_audio or counts.get(row["speaker_id"], 0) >= cap:
            continue
        seen_pairs.add(pair)
        seen_audio.add(audio)
        counts[row["speaker_id"]] = counts.get(row["speaker_id"], 0) + 1
        chosen.append(row)
        if len(chosen) == target:
            break
    if len(chosen) != target:
        raise SourceError(f"Only {len(chosen):,}/{target:,} clips survived. Increase --oversample, --max_shards or --max_clips_per_speaker.")
    if not {r["clip_id"] for r in extra} <= {r["clip_id"] for r in chosen}:
        raise SourceError("PART1 additions exceed the quota or contain duplicates.")
    return chosen


def print_summary_table(summary: pd.DataFrame) -> None:
    print()
    print("=" * 70)
    print("SUBSET SUMMARY")
    print("=" * 70)
    print(f"{'accent_group':<18} {'split':<6} {'clips':>7} {'speakers':>9} {'hours':>8}")
    print("-" * 70)
    for r in summary.itertuples(index=False):
        print(f"{r.accent_group:<18} {r.split:<6} {r.n_clips:>7,} "
              f"{r.n_speakers:>9,} {r.hours:>8.3f}")


# ============================================================
# CLI
# ============================================================

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build the Singaporean positive class from MNSC as "
                    "FLAC-in-parquet shards (no audio files written)."
    )
    p.add_argument("--configs", nargs="+", default=list(DEFAULT_CONFIGS),
                   help="MNSC configs (default: ASR-PART1-Train and ASR-PART2-Train).")
    p.add_argument("--allow_test_configs", action="store_true",
                   help="Allow ASR-*-Test configs.")
    p.add_argument("--source_dir", type=Path, default=None,
                   help="Local copy laid out like the Hub (<dir>/<config>/*.parquet); "
                        "no network access.")
    p.add_argument("--select_new_clips", action="store_true",
                   help="Sample a custom dataset instead of reproducing the fixed 40,000-clip manifest.\n"
                        "Required when changing selection, split or audio settings.")
    p.add_argument("--data_dir", type=Path, default=DATA_DIR,
                   help="Data folder (default: <project root>/data).")
    p.add_argument("--hf_repo", default=HF_REPO_DEFAULT)
    p.add_argument("--hf_revision", default="fb35c8c24be1056293c511ae47177e915a3ed757")
    p.add_argument("--cache_dir", type=Path, default=None,
                   help="HF cache dir (default: the shared ~/.cache/huggingface/hub).")
    p.add_argument("--no_download", action="store_true",
                   help="Never download shards; fail if a needed shard is not cached "
                        "(metadata is still range-read remotely).")
    p.add_argument("--clips_total", type=int, default=20000)
    p.add_argument("--max_clips_per_speaker", type=int, default=30)
    p.add_argument("--oversample", type=float, default=3.0)
    p.add_argument("--min_duration", type=float, default=3.0)
    p.add_argument("--max_duration", type=float, default=6.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val_fraction", type=float, default=0.15)
    p.add_argument("--test_fraction", type=float, default=0.15)
    p.add_argument("--max_shards", type=int, default=16,
                   help="Maximum metadata shards scanned per config.")
    p.add_argument("--saturation", type=float, default=0.01,
                   help="Stop scanning when a shard adds < this fraction of new speakers.")
    p.add_argument("--part1_extension", type=Path, default=None,
                   help="Prepared PART1 additions; downloaded automatically if absent.")
    p.add_argument("--extra_manifest", default=str(EXTRA_MANIFEST),
                   help="PART1 addition manifest (empty string disables additions).")
    p.add_argument("--shard_rows", type=int, default=SHARD_ROWS)
    args = p.parse_args(argv)
    args.configs = list(dict.fromkeys(c.strip() for c in args.configs if c.strip()))
    args.part1_extension = args.part1_extension or args.data_dir / "mnsc_part1_extra"
    return args


def validate_args(args: argparse.Namespace) -> Optional[str]:
    for name in ("val_fraction", "test_fraction"):
        v = getattr(args, name)
        if not (0.0 <= v < 1.0):
            return f"--{name} must be in [0, 1) (got {v})"
    if args.val_fraction + args.test_fraction >= 1.0:
        return "--val_fraction + --test_fraction must be < 1"
    if args.clips_total < len(args.configs):
        return "--clips_total must be at least the number of configs"
    if args.min_duration >= args.max_duration:
        return "--min_duration must be smaller than --max_duration"
    for name in ("clips_total", "max_clips_per_speaker", "shard_rows", "max_shards"):
        if getattr(args, name) <= 0:
            return f"--{name} must be positive"
    if args.oversample < 1.0:
        return "--oversample must be >= 1.0"
    if not (0.0 <= args.saturation < 1.0):
        return "--saturation must be in [0, 1)"
    if not args.configs:
        return "--configs must name at least one config"
    tests = [c for c in args.configs if is_test_config(c)]
    if tests and not args.allow_test_configs:
        return f"refusing {tests}. {TEST_CONFIGS_POLICY}"
    return None


# ============================================================
# MAIN
# ============================================================

def main(argv: Optional[List[str]] = None) -> int:
    global LAST_STATS
    args = parse_args(argv)
    if not args.select_new_clips:
        from reference_dataset import prepare_reference
        return prepare_reference("mnsc", args)
    error = validate_args(args)
    if error:
        print(f"ERROR: {error}")
        return 1
    paths = subset_paths(args.data_dir, SUBSET_NAME)
    mode = "local" if args.source_dir is not None else "hf"

    print("=" * 70)
    print(f"PREPARE MNSC ({ACCENT_GROUP} positives)")
    print("=" * 70)
    print(f"Source                : {args.source_dir or f'HF {args.hf_repo} @ {args.hf_revision}'}")
    print(f"Configs               : {args.configs}")
    print(f"Clips total           : {args.clips_total} (oversample {args.oversample})")
    print(f"Max clips per speaker : {args.max_clips_per_speaker}")
    print(f"Duration range        : {args.min_duration} - {args.max_duration} s (after trim)")
    print(f"Split (val/test)      : {args.val_fraction} / {args.test_fraction} by speaker")
    print(f"Seed                  : {args.seed}")
    print(f"License               : {LICENSE_NOTE}")

    stats: Dict = {
        "rows_scanned": 0, "no_speaker_id": 0, "empty_transcript": 0,
        "duplicate_utterance": 0, "shards_scanned": [], "stop_reason": "",
        "speakers_seen": 0, "download_bytes": 0,
        "skips": {"too_short": 0, "too_long": 0, "silent": 0,
                  "decode_error": 0, "missing_audio": 0},
    }
    sha, card_license, config_sizes = None, None, {}
    try:
        print()
        print("=" * 70)
        print("SHARDS")
        print("=" * 70)
        if mode == "local":
            shards = list_local_shards(args.source_dir, args.configs)
        else:
            sha, card_license = hf_dataset_info(args.hf_repo, args.hf_revision)
            print(f"  Resolved revision : {sha}")
            print(f"  Card license      : {card_license}")
            shards, config_sizes = list_hf_shards(args.hf_repo, sha, args.configs,
                                                  args.cache_dir)
        if any("PART3" in c.upper() for c in args.configs):
            print("  NOTE: PART3 clips are mostly 25-30 s raw; most are rejected by "
                  f"the {args.max_duration} s cap.")

        print()
        print("=" * 70)
        print("METADATA SCAN (answer + speaker ids only)")
        print("=" * 70)
        by_name = {r.name: r for r in shards}
        extra = load_extension(args, by_name, sha)
        rows, seen_pairs, seen_audio, speaker_counts = [], set(), set(), {}
        plans, used_shards = [], []
        for i, config in enumerate(args.configs):
            quota = args.clips_total // len(args.configs) + (i < args.clips_total % len(args.configs))
            scan_args = argparse.Namespace(**{**vars(args), "clips_total": quota})
            meta = scan_metadata([r for r in shards if r.config == config], scan_args,
                                 args.hf_repo, sha, stats)
            if meta.empty:
                raise SourceError(f"No usable rows in {config}.")
            selected, plan = select_clips(
                meta, [ACCENT_GROUP], quota, args.max_clips_per_speaker,
                args.min_duration, args.max_duration, args.oversample, args.seed,
            )
            plans.extend(plan)
            print_plan(plan, quota)
            needed = [by_name[n] for n in dict.fromkeys(selected["shard_key"])]
            used_shards.extend(r.name for r in needed)
            if mode == "hf":
                download_needed(needed, args.hf_repo, sha, args.cache_dir,
                                args.no_download, stats)
            processed = process_selected(selected, by_name, args, stats)
            additions = extra if config == "ASR-PART1-Train" else []
            chosen = choose_rows(additions, processed, quota, args.max_clips_per_speaker,
                                 seen_pairs, seen_audio, speaker_counts)
            rows.extend(chosen)
            print(f"{config}: {len(chosen):,} clips")
        plan = plans
    except SourceError as exc:
        print(f"ERROR: {ascii_safe(exc)}")
        return 1
    except Exception as exc:
        print(f"ERROR: could not access MNSC: {ascii_safe(exc)}")
        return 1

    print()
    print("=" * 70)
    print("SPEAKER-DISJOINT SPLIT")
    print("=" * 70)
    kept = pd.DataFrame([{k: v for k, v in r.items() if k != "audio_flac"} for r in rows])
    split_map = assign_splits(kept, ACCENT_GROUP, args.val_fraction,
                              args.test_fraction, args.seed)
    kept["split"] = kept["speaker_id"].map(split_map)
    n_speakers = kept["speaker_id"].nunique()
    if n_speakers < MIN_SPEAKERS_WARNING:
        print(f"  WARNING: only {n_speakers} speakers in the final subset "
              f"(< {MIN_SPEAKERS_WARNING}).")

    out_rows = [{k: v for k, v in dict(r, split=s).items()
                 if k in SUBSET_SCHEMA.names}
                for r, s in zip(rows, kept["split"])]
    table = pa.Table.from_pylist(out_rows, schema=SUBSET_SCHEMA)
    n_shards = write_subset_shards(table, paths["subset_dir"], args.shard_rows)
    print(f"  Wrote {n_shards} shard(s), {len(out_rows):,} clips -> {paths['subset_dir']}")

    summary_df = kept.drop(columns=["selection_rank", "raw_speaker_id"])
    summary = summarise(summary_df)
    summary.to_csv(paths["summary_csv"], index=False)
    print_summary_table(summary)

    per_split = {split: {"n_clips": int(len(s)),
                         "speakers": sorted(s["speaker_id"].unique().tolist())}
                 for split, s in kept.groupby("split")}
    run_args = {k: v for k, v in vars(args).items()
                if k not in ("source_dir", "data_dir", "cache_dir",
                              "part1_extension", "extra_manifest")}
    info = {
        "source":           SOURCE,
        "dataset":          {"mode": mode,
                             "hf_repo": args.hf_repo if mode == "hf" else None,
                             "hf_revision": args.hf_revision if mode == "hf" else None,
                             "resolved_sha": sha,
                             "configs": args.configs,
                             "config_sizes": config_sizes,
                             "shards_scanned": stats["shards_scanned"],
                             "shards_used": used_shards,
                             "download_bytes": stats["download_bytes"]},
        "license":          {"note": LICENSE_NOTE, "card_license": card_license},
        "args":             run_args,
        "part1_additions":  len(extra),
        "accent_group":     ACCENT_GROUP,
        "accent_raw":       sorted(kept["accent_raw"].unique().tolist()),
        "speaker_id_source": "other_attributes.speaker1 -> speaker2 -> speaker",
        "scan":             {k: stats[k] for k in ("rows_scanned", "no_speaker_id",
                                                   "empty_transcript", "duplicate_utterance",
                                                   "stop_reason", "speakers_seen")},
        "test_configs_policy": TEST_CONFIGS_POLICY,
        "audio_processing": AUDIO_PROCESSING,
        "skips":            stats["skips"],
        "selection_plan":   plan,
        "sentence_column":  "normalised transcript",
        "split_strategy":   "speaker-disjoint, greedy clip-weighted, seeded",
        "groups":           {ACCENT_GROUP: per_split},
    }
    with open(paths["split_info_json"], "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=2)

    print()
    print("=" * 70)
    print("MNSC COMPLETE")
    print("=" * 70)
    print(f"  Saved shards    -> {paths['subset_dir']}")
    print(f"  Saved summary   -> {paths['summary_csv']}")
    print(f"  Saved split info-> {paths['split_info_json']}")
    print()
    print("Load in other scripts with:")
    print("  from cv_utils import load_accent_data")
    print("  df = load_accent_data(sources=['mnsc'], split='train')")

    LAST_STATS = {
        "mode":           mode,
        "shards_scanned": len(stats["shards_scanned"]),
        "stop_reason":    stats["stop_reason"],
        "scan":           info["scan"],
        "skips":          stats["skips"],
        "plan":           plan,
        "n_selected":     sum(p["selected_clips"] for p in plans),
        "n_kept":         len(out_rows),
        "n_shards":       n_shards,
        "download_bytes": stats["download_bytes"],
    }
    return 0


if __name__ == "__main__":
    sys.exit(main())
