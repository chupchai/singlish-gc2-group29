"""
prepare_speech_accent_archive.py

Build the "Korean" accent group from the Speech Accent Archive (George
Mason University) as FLAC-in-parquet shards with exactly the Common Voice
subset schema. No audio file is ever written to disk.

Why this source
---------------
Common Voice English has only ~3 labelled Korean speakers. In the Speech
Accent Archive every speaker reads the same ~30 s English paragraph
("Please call Stella..."); the live site lists ~99 native Korean speakers.

How the data is obtained
------------------------
1. Default (no account needed): the public Hugging Face mirror
   "CodecSR/speech_accent_archive_other", split "original" (9 parquet
   shards, columns id / accent / audio; 52 Korean speakers). Only the
   id/accent columns of each shard are range-read remotely (a few KB), then
   ONLY the shards that contain the requested languages are downloaded into
   <project root>/raw/speech_accent_archive_hf/ (~0.9 GB for Korean). This is a
   third-party redistribution of the archive, not an official GMU release.
2. Manual alternative (Kaggle mirror, needs a free Kaggle login):
     a. log in at https://www.kaggle.com
     b. open https://www.kaggle.com/datasets/rtatman/speech-accent-archive
        and click "Download" (zip with speakers_all.csv and
        recordings/recordings/<filename>.mp3)
     c. save the zip, e.g. as raw\\speech-accent-archive.zip
     d. python scripts\\prepare_speech_accent_archive.py --source_dir raw\\speech-accent-archive.zip
   The zip is read member by member WITHOUT extracting it; an unzipped
   folder works as well. --source_dir may also point at a folder of
   mirror-layout parquet files (id / accent / audio).

License / citation
------------------
    Free for non-commercial use with attribution (Kaggle lists
    CC BY-NC-SA 4.0). Cite: Weinberger, S. (2015). Speech Accent Archive.
    George Mason University. http://accent.gmu.edu

Chunking
--------
Each recording is one long paragraph, so it is cut into chunks (see
chunk_recording): frame RMS (25 ms / 10 ms, as the silence trim), frames
below --silence_db relative to the loudest frame are "silent"; runs of
silent frames lasting >= --min_pause are pause candidates and a cut is
placed at the pause centre. Greedily, each chunk ends at the longest pause
within [--chunk_target_min, --chunk_target_max] s (default 4-5.5 s) of its
start, else within [min_duration + 0.5, max_duration - 0.5] s, else
("fallback") at the quietest frame within the target window. A final
chunk shorter than --min_duration is merged into the previous one (if the
result stays <= --max_duration) or dropped. Every chunk then goes through
cv_utils.preprocess_waveform exactly like Common Voice (trim, 3-6 s
filter, -20 dBFS RMS + peak limit). All chunks of one recording share the
speaker_id "saa_<id>", so the speaker-disjoint split keeps them together.

Columns
-------
    source="speech_accent_archive", accent_group="Korean" (via
    NATIVE_LANGUAGE_GROUPS), accent_raw="Korean L1 (Speech Accent Archive)",
    is_singaporean=False, speaker_id="saa_<id>", clip_id="saa_<id>_cNN".
    sentence = "" (empty) for every chunk: chunk boundaries do not align
    with words and every speaker reads the same text; the full paragraph is
    stored once in the split-info JSON ("elicitation_paragraph").
    NOTE: every Korean clip therefore has the same lexical content
    (possible confound for the classifier).
    No age filter: the HF mirror has no age column and the archive's
    speakers are (almost) all adults.

Usage
-----
    python prepare_speech_accent_archive.py
    python prepare_speech_accent_archive.py --native_languages korean mandarin
    python prepare_speech_accent_archive.py --source_dir raw\\speech-accent-archive.zip

Outputs (written to <project root>/data/)
-----------------------------------------
    data/saa_subset/part-XXXX.parquet   -- cv_utils.SUBSET_SCHEMA
    data/saa_subset_summary.csv         -- clips / speakers / hours per group x split
    data/saa_subset_split_info.json     -- args, chunking, paragraph, speaker lists

Loading
-------
    from cv_utils import load_accent_data
    df = load_accent_data(sources=["speech_accent_archive"], split="train")
"""

import argparse
import json
import posixpath
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from cv_utils import (
    AUDIO_PROCESSING,
    DATA_DIR,
    RAW_DIR,
    SHARD_ROWS,
    SUBSET_SCHEMA,
    TARGET_SR,
    TRIM_FRAME_MS,
    TRIM_HOP_MS,
    ascii_safe,
    decode_audio_bytes,
    encode_flac,
    ensure_free_space,
    preprocess_waveform,
    resample_to_target,
    subset_paths,
    to_mono,
    write_subset_shards,
)
from extract_common_voice_subset import (
    assign_splits,
    print_plan,
    select_clips,
    summarise,
)


SOURCE = "speech_accent_archive"
SUBSET_NAME = "saa_subset"
ID_PREFIX = "saa_"
NATIVE_LANGUAGE_GROUPS = {"korean": "Korean", "mandarin": "Mainland Chinese"}
HF_REPO_DEFAULT = "CodecSR/speech_accent_archive_other"
MIRROR_COLUMNS = ("id", "accent", "audio")
ELICITATION_PARAGRAPH = (
    "Please call Stella. Ask her to bring these things with her from the "
    "store: Six spoons of fresh snow peas, five thick slabs of blue cheese, "
    "and maybe a snack for her brother Bob. We also need a small plastic "
    "snake and a big toy frog for the kids. She can scoop these things into "
    "three red bags, and we will go meet her Wednesday at the train station."
)
LICENSE_NOTE = (
    "Speech Accent Archive: free for non-commercial use with attribution "
    "(Kaggle lists CC BY-NC-SA 4.0). Cite Weinberger, S. (2015). Speech "
    "Accent Archive. George Mason University. http://accent.gmu.edu. "
    f"The default HF mirror ({HF_REPO_DEFAULT}) is a third-party redistribution."
)
KAGGLE_INSTRUCTIONS = (
    "Manual download (Kaggle mirror, needs a free Kaggle login):\n"
    "  1. Log in at https://www.kaggle.com\n"
    "  2. Open https://www.kaggle.com/datasets/rtatman/speech-accent-archive\n"
    "     and click 'Download' (zip with speakers_all.csv + recordings/recordings/*.mp3)\n"
    "  3. Save the zip, e.g. as raw\\speech-accent-archive.zip (no need to unzip)\n"
    "  4. python scripts\\prepare_speech_accent_archive.py --source_dir raw\\speech-accent-archive.zip\n"
    "Or run without --source_dir to use the Hugging Face mirror (no login)."
)

LAST_STATS: Dict = {}


class SourceError(Exception):
    """The requested recording source cannot be used."""


def accent_group_for(language: str) -> str:
    return NATIVE_LANGUAGE_GROUPS.get(language, language.title())


def _truthy(value) -> bool:
    return str(value).strip().lower() in ("true", "1", "yes", "t")


# ============================================================
# CHUNKING
# ============================================================

def chunk_recording(
    y: np.ndarray,
    sr: int,
    min_sec: float,
    max_sec: float,
    target_min: float,
    target_max: float,
    silence_db: float,
    min_pause: float,
) -> List[Tuple[int, int, str]]:
    """
    Split a mono signal into (start, end, method) sample spans of
    min_sec..max_sec seconds, preferring cuts at pauses (module docstring).
    method describes how the chunk's END was chosen: "pause", "fallback"
    or "end" (end of the voiced recording).
    """
    frame = max(1, int(round(sr * TRIM_FRAME_MS / 1000)))
    hop = max(1, int(round(sr * TRIM_HOP_MS / 1000)))
    if len(y) < frame:
        return []
    frames = np.lib.stride_tricks.sliding_window_view(y, frame)[::hop]
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    ref = rms.max()
    if ref <= 0:
        return []
    db = 20.0 * np.log10(np.maximum(rms, 1e-20) / ref)
    silent = db < silence_db
    voiced = np.nonzero(~silent)[0]
    if len(voiced) == 0:
        return []
    v_start = int(voiced[0] * hop)
    v_end = int(min(len(y), voiced[-1] * hop + frame))

    # Pause candidates: (cut sample, pause length in seconds).
    pauses: List[Tuple[int, float]] = []
    i, n = int(voiced[0]), int(voiced[-1])
    while i <= n:
        if silent[i]:
            j = i
            while j <= n and silent[j]:
                j += 1
            length = (j - i) * hop / sr
            if length >= min_pause:
                centre = int(((i + j) / 2.0) * hop + frame / 2.0)
                pauses.append((centre, length))
            i = j
        else:
            i += 1
    cuts = np.array([c for c, _ in pauses], dtype=np.int64)
    lengths = np.array([l for _, l in pauses], dtype=np.float64)

    def best_pause(lo: int, hi: int, mid: float) -> Optional[int]:
        if len(cuts) == 0:
            return None
        idx = np.nonzero((cuts >= lo) & (cuts <= hi))[0]
        if len(idx) == 0:
            return None
        # Longest pause; ties -> closest to the window midpoint.
        best = max(idx, key=lambda k: (round(lengths[k], 6), -abs(cuts[k] - mid)))
        return int(cuts[best])

    spans: List[Tuple[int, int, str]] = []
    pos = v_start
    while pos < v_end:
        if v_end - pos <= target_max * sr:
            spans.append((pos, v_end, "end"))
            break
        lo, hi = pos + int(target_min * sr), pos + int(target_max * sr)
        mid = (lo + hi) / 2.0
        cut = best_pause(lo, hi, mid)
        method = "pause"
        if cut is None:
            cut = best_pause(pos + int((min_sec + 0.5) * sr),
                             pos + int((max_sec - 0.5) * sr), mid)
        if cut is None:
            f_lo = max(0, lo // hop)
            f_hi = min(len(rms) - 1, max(f_lo, (hi - frame) // hop))
            k = f_lo + int(np.argmin(rms[f_lo:f_hi + 1]))
            cut = int(k * hop + frame // 2)
            method = "fallback"
        cut = min(max(cut, pos + 1), v_end)
        spans.append((pos, cut, method))
        pos = cut

    # Tail shorter than min_sec: merge into the previous chunk or drop it.
    if spans and (spans[-1][1] - spans[-1][0]) < min_sec * sr:
        tail = spans.pop()
        if spans and (tail[1] - spans[-1][0]) <= max_sec * sr:
            prev = spans.pop()
            spans.append((prev[0], tail[1], tail[2]))
    return spans


# ============================================================
# RECORDING SOURCES  -> iterators of (speaker_key, language, audio_bytes)
# ============================================================

Recording = Tuple[str, str, bytes]


def _parquet_has_mirror_columns(path: Path) -> bool:
    try:
        names = set(pq.read_schema(str(path)).names)
    except Exception:
        return False
    return set(MIRROR_COLUMNS) <= names


def _iter_mirror_parquets(files: List[Path], languages: List[str],
                          stats: Dict) -> Iterator[Recording]:
    wanted = set(languages)
    for f in files:
        pf = pq.ParquetFile(str(f))
        for batch in pf.iter_batches(columns=list(MIRROR_COLUMNS), batch_size=8):
            for r in batch.to_pylist():
                stats["rows_scanned"] += 1
                lang = str(r["accent"] or "").strip().lower()
                if lang not in wanted:
                    continue
                audio = r["audio"] or {}
                if not audio.get("bytes"):
                    stats["missing_audio"] += 1
                    continue
                yield str(r["id"]).strip(), lang, audio["bytes"]


def _kaggle_records(csv_bytes: bytes, languages: List[str], stats: Dict) -> List[Tuple[str, str]]:
    import io
    df = pd.read_csv(io.BytesIO(csv_bytes), dtype=str, keep_default_na=False)
    cols = {c.strip().lower(): c for c in df.columns}
    for need in ("filename", "native_language"):
        if need not in cols:
            raise SourceError(f"speakers_all.csv has no '{need}' column "
                              f"(columns: {list(df.columns)}).")
    missing_col = cols.get("file_missing?")
    wanted = set(languages)
    out = []
    for _, r in df.iterrows():
        stats["rows_scanned"] += 1
        lang = str(r[cols["native_language"]]).strip().lower()
        if lang not in wanted:
            continue
        if missing_col and _truthy(r[missing_col]):
            stats["missing_audio"] += 1
            continue
        key = str(r[cols["filename"]]).strip()
        if key:
            out.append((key, lang))
    return out


def iter_local(source_dir: Path, languages: List[str], stats: Dict) -> Tuple[str, Iterator[Recording]]:
    """Return (mode, iterator) for a Kaggle zip / folder or a mirror parquet folder."""
    source_dir = Path(source_dir)
    if source_dir.is_file() and source_dir.suffix.lower() == ".zip":
        zf = zipfile.ZipFile(source_dir)
        names = [i.filename for i in zf.infolist() if not i.is_dir()]
        csvs = [n for n in names if posixpath.basename(n).lower() == "speakers_all.csv"]
        if not csvs:
            zf.close()
            raise SourceError(f"No speakers_all.csv inside {source_dir.name}.")
        mp3s = {posixpath.splitext(posixpath.basename(n))[0].lower(): n
                for n in names if n.lower().endswith(".mp3")}
        records = _kaggle_records(zf.read(csvs[0]), languages, stats)

        def gen() -> Iterator[Recording]:
            try:
                for key, lang in records:
                    member = mp3s.get(key.lower())
                    if member is None:
                        stats["missing_audio"] += 1
                        continue
                    yield key, lang, zf.read(member)
            finally:
                zf.close()
        return "local_zip", gen()

    if not source_dir.is_dir():
        raise SourceError(f"--source_dir not found: {source_dir}")

    parquets = sorted(p for p in source_dir.rglob("*.parquet")
                      if _parquet_has_mirror_columns(p))
    if parquets:
        stats["shards_used"] = [p.name for p in parquets]
        return "local_parquet", _iter_mirror_parquets(parquets, languages, stats)

    csvs = sorted(source_dir.rglob("speakers_all.csv"))
    if not csvs:
        raise SourceError(
            f"{source_dir} contains neither speakers_all.csv (Kaggle layout) "
            "nor parquet files with id/accent/audio columns (HF mirror layout).")
    mp3s = {p.stem.lower(): p for p in source_dir.rglob("*.mp3")}
    records = _kaggle_records(csvs[0].read_bytes(), languages, stats)

    def gen_dir() -> Iterator[Recording]:
        for key, lang in records:
            path = mp3s.get(key.lower())
            if path is None:
                stats["missing_audio"] += 1
                continue
            yield key, lang, path.read_bytes()
    return "local_dir", gen_dir()


def iter_hf_mirror(repo: str, split: str, revision: str, cache_dir: Path,
                   languages: List[str], no_download: bool,
                   stats: Dict) -> Tuple[str, Iterator[Recording]]:
    """Find shards with the requested languages, download only those, iterate."""
    from huggingface_hub import HfApi, HfFileSystem, hf_hub_download

    api = HfApi(token=False if repo == HF_REPO_DEFAULT else None)
    files = sorted(f for f in api.list_repo_files(repo, repo_type="dataset", revision=revision)
                   if f.startswith(f"data/{split}-") and f.endswith(".parquet"))
    if not files:
        raise SourceError(f"No data/{split}-*.parquet files in {repo}.")
    print(f"Scanning id/accent columns of {len(files)} shard(s) in {repo} ({split})")
    fs = HfFileSystem(token=False if repo == HF_REPO_DEFAULT else None)
    wanted = set(languages)
    needed: List[str] = []
    for name in files:
        local = cache_dir / name
        if local.is_file():
            t = pq.read_table(str(local), columns=["id", "accent"])
        else:
            with fs.open(f"datasets/{repo}@{revision}/{name}", "rb") as fh:
                t = pq.ParquetFile(fh).read(columns=["id", "accent"])
        langs = {str(a).strip().lower() for a in t.column("accent").to_pylist()}
        hits = sorted(langs & wanted)
        print(f"  {name}: {t.num_rows} rows, requested languages: {hits or '-'}")
        if hits:
            needed.append(name)
    if not needed:
        raise SourceError(f"No recordings for {languages} in {repo} ({split}).")

    missing = [n for n in needed if not (cache_dir / n).is_file()]
    if missing:
        if no_download:
            raise SourceError(f"Shards not downloaded yet (and --no_download): {missing}")
        infos = api.get_paths_info(repo, missing, repo_type="dataset", revision=revision)
        total = sum(int(getattr(i, "size", 0) or 0) for i in infos)
        print(f"Downloading {len(missing)} shard(s), {total / 1e9:.2f} GB -> {cache_dir}")
        ensure_free_space(cache_dir, total)
        cache_dir.mkdir(parents=True, exist_ok=True)
        for n in missing:
            hf_hub_download(repo, n, repo_type="dataset", revision=revision,
                            local_dir=str(cache_dir), token=False if repo == HF_REPO_DEFAULT else None)
            print(f"  downloaded {n}")
    stats["shards_used"] = [posixpath.basename(n) for n in needed]
    return "hf_mirror", _iter_mirror_parquets([cache_dir / n for n in needed],
                                              languages, stats)


# ============================================================
# PROCESSING
# ============================================================

def process_recordings(recordings: Iterator[Recording], args: argparse.Namespace,
                       stats: Dict) -> List[dict]:
    rows: List[dict] = []
    methods: Counter = Counter()
    seen_keys: set = set()
    for key, lang, data in recordings:
        if key in seen_keys:
            stats["duplicate_recordings"] += 1
            continue
        seen_keys.add(key)
        stats["recordings"] += 1
        stats["speakers_by_language"][lang] = stats["speakers_by_language"].get(lang, 0) + 1
        try:
            x, sr = decode_audio_bytes(data)
            y = resample_to_target(to_mono(np.asarray(x, dtype=np.float32)), sr)
        except Exception as exc:
            print(f"  decode error ({ascii_safe(key)}): {ascii_safe(exc)}")
            stats["skips"]["decode_error"] += 1
            continue
        stats["recording_sec"].append(len(y) / TARGET_SR)
        spans = chunk_recording(y, TARGET_SR, args.min_duration, args.max_duration,
                                args.chunk_target_min, args.chunk_target_max,
                                args.silence_db, args.min_pause)
        if not spans:
            stats["recordings_without_chunks"] += 1
        group = accent_group_for(lang)
        for idx, (s, e, method) in enumerate(spans):
            methods[method] += 1
            stats["chunks"] += 1
            z, reason = preprocess_waveform(y[s:e], TARGET_SR,
                                            args.min_duration, args.max_duration)
            if z is None:
                stats["skips"][reason] += 1
                continue
            rows.append({
                "clip_id":        f"{ID_PREFIX}{key}_c{idx:02d}",
                "source":         SOURCE,
                "speaker_id":     f"{ID_PREFIX}{key}",
                "accent_raw":     f"{lang.title()} L1 (Speech Accent Archive)",
                "accent_group":   group,
                "is_singaporean": False,
                "duration_sec":   float(len(z) / TARGET_SR),
                "sentence":       "",
                "sampling_rate":  TARGET_SR,
                "audio_flac":     encode_flac(z),
            })
    stats["chunk_methods"] = dict(methods)
    return rows


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
        description="Build the Korean group from the Speech Accent Archive as "
                    "chunked FLAC-in-parquet shards (no audio files written)."
    )
    p.add_argument("--source_dir", type=Path, default=None,
                   help="Kaggle zip / unzipped folder, or a folder of mirror parquet "
                        "files (default: use the Hugging Face mirror).")
    p.add_argument("--select_new_clips", action="store_true",
                   help="Sample a custom dataset instead of reproducing the fixed 40,000-clip manifest.\n"
                        "Required when changing selection, split or audio settings.")
    p.add_argument("--data_dir", type=Path, default=DATA_DIR,
                   help="Data folder (default: <project root>/data).")
    p.add_argument("--hf_repo", default=HF_REPO_DEFAULT)
    p.add_argument("--hf_split", default="original",
                   help="Mirror split (only 'original' is unmodified audio).")
    p.add_argument("--hf_revision", default="23016f358efe530a1cda02098da9dd1996cd30d7")
    p.add_argument("--cache_dir", type=Path, default=RAW_DIR / "speech_accent_archive_hf",
                   help="Where mirror shards are downloaded "
                        "(default: <project root>/raw/speech_accent_archive_hf).")
    p.add_argument("--no_download", action="store_true",
                   help="Never download mirror shards; fail if they are missing.")
    p.add_argument("--native_languages", nargs="+", default=["korean"],
                   help="Native languages to keep (default: korean).")
    p.add_argument("--min_duration", type=float, default=3.0)
    p.add_argument("--max_duration", type=float, default=6.0)
    p.add_argument("--chunk_target_min", type=float, default=4.0)
    p.add_argument("--chunk_target_max", type=float, default=5.5)
    p.add_argument("--silence_db", type=float, default=-35.0,
                   help="Frames below this (dB re loudest frame) count as silence.")
    p.add_argument("--min_pause", type=float, default=0.15,
                   help="Minimum pause length (s) to cut at.")
    p.add_argument("--max_clips_per_speaker", type=int, default=15)
    p.add_argument("--clips_per_accent", type=int, default=329)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val_fraction", type=float, default=0.15)
    p.add_argument("--test_fraction", type=float, default=0.15)
    p.add_argument("--shard_rows", type=int, default=SHARD_ROWS)
    args = p.parse_args(argv)
    args.native_languages = list(dict.fromkeys(
        l.strip().lower() for l in args.native_languages if l.strip()))
    return args


def validate_args(args: argparse.Namespace) -> Optional[str]:
    for name in ("val_fraction", "test_fraction"):
        v = getattr(args, name)
        if not (0.0 <= v < 1.0):
            return f"--{name} must be in [0, 1) (got {v})"
    if args.val_fraction + args.test_fraction >= 1.0:
        return "--val_fraction + --test_fraction must be < 1"
    if args.min_duration >= args.max_duration:
        return "--min_duration must be smaller than --max_duration"
    for name in ("clips_per_accent", "max_clips_per_speaker", "shard_rows"):
        if getattr(args, name) <= 0:
            return f"--{name} must be positive"
    if not (args.min_duration <= args.chunk_target_min < args.chunk_target_max
            <= args.max_duration):
        return ("need --min_duration <= --chunk_target_min < --chunk_target_max "
                "<= --max_duration")
    if args.silence_db >= 0:
        return "--silence_db must be negative"
    if args.min_pause <= 0:
        return "--min_pause must be positive"
    if not args.native_languages:
        return "--native_languages must name at least one language"
    return None


# ============================================================
# MAIN
# ============================================================

def main(argv: Optional[List[str]] = None) -> int:
    global LAST_STATS
    args = parse_args(argv)
    if not args.select_new_clips:
        from reference_dataset import prepare_reference
        return prepare_reference("speech_accent_archive", args)
    error = validate_args(args)
    if error:
        print(f"ERROR: {error}")
        return 1
    paths = subset_paths(args.data_dir, SUBSET_NAME)
    groups_requested = {l: accent_group_for(l) for l in args.native_languages}

    print("=" * 70)
    print("PREPARE SPEECH ACCENT ARCHIVE")
    print("=" * 70)
    print(f"Source                : {args.source_dir or f'HF mirror {args.hf_repo} ({args.hf_split})'}")
    print(f"Native languages      : {groups_requested}")
    print(f"Chunk target          : {args.chunk_target_min} - {args.chunk_target_max} s "
          f"(pause >= {args.min_pause} s at {args.silence_db} dB)")
    print(f"Duration range        : {args.min_duration} - {args.max_duration} s (after trim)")
    print(f"Clips per accent      : {args.clips_per_accent}")
    print(f"Max clips per speaker : {args.max_clips_per_speaker}")
    print(f"Split (val/test)      : {args.val_fraction} / {args.test_fraction} by speaker")
    print(f"Seed                  : {args.seed}")
    print(f"License               : {LICENSE_NOTE}")

    stats: Dict = {
        "rows_scanned": 0, "missing_audio": 0, "duplicate_recordings": 0,
        "recordings": 0, "recordings_without_chunks": 0, "chunks": 0,
        "speakers_by_language": {}, "recording_sec": [], "shards_used": [],
        "skips": {"too_short": 0, "too_long": 0, "silent": 0, "decode_error": 0},
    }
    try:
        if args.source_dir is not None:
            mode, recordings = iter_local(args.source_dir, args.native_languages, stats)
        else:
            mode, recordings = iter_hf_mirror(
                args.hf_repo, args.hf_split, args.hf_revision, args.cache_dir,
                args.native_languages, args.no_download, stats)
    except SourceError as exc:
        print(f"ERROR: {ascii_safe(exc)}")
        print(KAGGLE_INSTRUCTIONS)
        return 1
    except Exception as exc:
        print(f"ERROR: could not access the recordings: {ascii_safe(exc)}")
        print(KAGGLE_INSTRUCTIONS)
        return 1

    print()
    print("=" * 70)
    print(f"DECODE + CHUNK + PREPROCESS ({mode}, in memory)")
    print("=" * 70)
    rows = process_recordings(recordings, args, stats)
    rec = stats["recording_sec"]
    print()
    for lang, n in sorted(stats["speakers_by_language"].items()):
        print(f"SPEAKERS ({lang:<10} -> {accent_group_for(lang)}) : {n:,}")
    print(f"Recordings decoded          : {len(rec):,}"
          + (f" (length {min(rec):.1f} - {max(rec):.1f} s, median {np.median(rec):.1f} s)"
             if rec else ""))
    print(f"Missing audio files         : {stats['missing_audio']:,}")
    print(f"Chunks cut (by end method)  : {stats['chunks']:,} {stats['chunk_methods']}")
    print(f"Kept after filters          : {len(rows):,} chunks")
    for k, v in stats["skips"].items():
        print(f"Skipped ({k:<12})     : {v:,}")
    print("NOTE: every speaker reads the same paragraph, so all clips share the "
          "same lexical content (sentence column left empty).")
    if not rows:
        print("ERROR: no chunks survived processing.")
        if mode != "hf_mirror":
            print(KAGGLE_INSTRUCTIONS)
        return 1

    meta = pd.DataFrame([{k: v for k, v in r.items() if k != "audio_flac"} for r in rows])
    groups = sorted(meta["accent_group"].unique())
    selected, plan = select_clips(
        meta, groups, args.clips_per_accent, args.max_clips_per_speaker,
        args.min_duration, args.max_duration, 1.0, args.seed,
    )
    print_plan(plan, args.clips_per_accent)

    print()
    print("=" * 70)
    print("SPEAKER-DISJOINT SPLIT")
    print("=" * 70)
    parts = []
    for group in groups:
        g = selected[selected["accent_group"] == group].sort_values("selection_rank").copy()
        split_map = assign_splits(g, group, args.val_fraction, args.test_fraction, args.seed)
        g["split"] = g["speaker_id"].map(split_map)
        parts.append(g)
    selected = pd.concat(parts, ignore_index=True)

    by_id = {r["clip_id"]: r for r in rows}
    out_rows = [dict(by_id[cid], split=split)
                for cid, split in zip(selected["clip_id"], selected["split"])]
    table = pa.Table.from_pylist(out_rows, schema=SUBSET_SCHEMA)
    n_shards = write_subset_shards(table, paths["subset_dir"], args.shard_rows)
    print(f"  Wrote {n_shards} shard(s), {len(out_rows):,} clips -> {paths['subset_dir']}")

    kept = selected.drop(columns=["selection_rank"])
    summary = summarise(kept)
    summary.to_csv(paths["summary_csv"], index=False)
    print_summary_table(summary)

    per_group = {}
    for group, g in kept.groupby("accent_group"):
        per_group[group] = {
            split: {"n_clips": int(len(s)),
                    "speakers": sorted(s["speaker_id"].unique().tolist())}
            for split, s in g.groupby("split")
        }
    run_args = {k: v for k, v in vars(args).items()
                if k not in ("source_dir", "data_dir", "cache_dir")}
    info = {
        "source":           SOURCE,
        "dataset":          {"mode": mode, "hf_repo": args.hf_repo if mode == "hf_mirror" else None,
                             "hf_split": args.hf_split if mode == "hf_mirror" else None,
                             "hf_revision": args.hf_revision if mode == "hf_mirror" else None,
                             "shards_used": stats["shards_used"]},
        "license":          LICENSE_NOTE,
        "args":             run_args,
        "native_languages": groups_requested,
        "speakers_by_language": stats["speakers_by_language"],
        "elicitation_paragraph": ELICITATION_PARAGRAPH,
        "sentence_column":  "empty; every speaker reads elicitation_paragraph",
        "chunking":         {
            "chunk_target_min": args.chunk_target_min,
            "chunk_target_max": args.chunk_target_max,
            "silence_db":       args.silence_db,
            "min_pause":        args.min_pause,
            "min_duration":     args.min_duration,
            "max_duration":     args.max_duration,
            "chunks_cut":       stats["chunks"],
            "end_methods":      stats["chunk_methods"],
            "recordings_without_chunks": stats["recordings_without_chunks"],
        },
        "audio_processing": AUDIO_PROCESSING,
        "missing_audio":    stats["missing_audio"],
        "skips":            stats["skips"],
        "selection_plan":   plan,
        "split_strategy":   "speaker-disjoint within each accent group, "
                            "greedy clip-weighted, seeded",
        "groups":           per_group,
    }
    with open(paths["split_info_json"], "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=2)

    print()
    print("=" * 70)
    print("SPEECH ACCENT ARCHIVE COMPLETE")
    print("=" * 70)
    print(f"  Saved shards    -> {paths['subset_dir']}")
    print(f"  Saved summary   -> {paths['summary_csv']}")
    print(f"  Saved split info-> {paths['split_info_json']}")
    print()
    print("Load in other scripts with:")
    print("  from cv_utils import load_accent_data")
    print("  df = load_accent_data(sources=['speech_accent_archive'], split='train')")

    LAST_STATS = {
        "mode":     mode,
        **{k: v for k, v in stats.items() if k != "recording_sec"},
        "groups":   groups,
        "plan":     plan,
        "n_kept_chunks": len(rows),
        "n_selected": len(out_rows),
        "n_shards": n_shards,
    }
    return 0


if __name__ == "__main__":
    sys.exit(main())
