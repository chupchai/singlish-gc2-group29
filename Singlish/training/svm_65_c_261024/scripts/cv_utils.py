"""
cv_utils.py

Shared helpers for the Common Voice (non-Singaporean accent) data pipeline.

Purpose
-------
Everything the three pipeline scripts have in common lives here:

    - project paths (all built from this file's location, no absolute paths)
    - streaming access to the Common Voice archive WITHOUT extracting it
      (tar / tar.gz / tar.bz2 / tar.xz via tarfile 'r|*', zip via zipfile)
    - the audio preprocessing chain applied to every kept clip
    - FLAC encode / decode helpers (audio is stored as FLAC bytes in parquet)
    - TSV reading for Common Voice metadata
    - load_cv_subset(), the public loader for the extracted subset

Audio preprocessing chain (preprocess_waveform)
-----------------------------------------------
Apply the SAME chain to the MNSC (Singaporean) clips, otherwise the
classifier can learn source differences (loudness, leading silence,
sample rate) instead of accent. Call cv_utils.preprocess_waveform(x, sr,
min_duration, max_duration) on the MNSC audio to match exactly.

    1. Decode MP3 with soundfile (libsndfile) from memory -> float32.
    2. Mono: mean over channels.
    3. Resample to 16 kHz with soxr.resample(x, sr, 16000, quality="HQ").
    4. Trim leading/trailing silence: frame RMS with 25 ms frames and
       10 ms hop; keep the span from the first to the last frame whose RMS
       is within 40 dB of the loudest frame; pad 100 ms on each side
       (clamped to the clip). Same idea as librosa.effects.trim(top_db=40)
       plus padding, implemented in numpy.
    5. Duration filter on the TRIMMED length: keep
       min_duration <= len / 16000 <= max_duration.
    6. Reject as "silent" if RMS < 1e-5.
    7. Loudness-normalise to -20 dBFS RMS; then, if the peak exceeds 0.99,
       scale down so the peak is exactly 0.99 (no clipping).
    8. Encode as FLAC, 16-bit PCM (PCM_16), 16 kHz mono, in memory.

Usage
-----
    from cv_utils import load_cv_subset
    train_df = load_cv_subset(split="train")      # decoded float32 audio
    meta_df  = load_cv_subset(decode_audio=False) # keeps audio_flac bytes

    # All sources (Common Voice + speechocean762 + Speech Accent Archive + MNSC):
    from cv_utils import load_accent_data
    train_df = load_accent_data(split="train")
    kor_df   = load_accent_data(sources=["speech_accent_archive"])

Sources and their subset folders (SOURCE_SUBSETS)
-------------------------------------------------
    common_voice           data/cv_subset/          extract_common_voice_subset.py
    speechocean762         data/speechocean_subset/ prepare_speechocean762.py
    speech_accent_archive  data/saa_subset/         prepare_speech_accent_archive.py
    mnsc                   data/mnsc_subset/        prepare_mnsc_singaporean.py

Every subset uses exactly SUBSET_SCHEMA, so they concatenate directly.

Outputs
-------
    This module writes nothing on its own.
"""

import csv
import io
import posixpath
import shutil
import tarfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
import soxr


# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent              # scripts/ holds code only
RAW_DIR      = PROJECT_ROOT / "raw"           # downloaded archive (never extracted)
DATA_DIR     = PROJECT_ROOT / "data"          # pipeline outputs
ENV_FILE     = PROJECT_ROOT / ".env"


def data_paths(data_dir: Path) -> Dict[str, Path]:
    """Return every pipeline output path under ``data_dir``."""
    data_dir = Path(data_dir)
    return {
        "data_dir":           data_dir,
        "cv_metadata_dir":    data_dir / "cv_metadata",
        "validated_parquet":  data_dir / "cv_validated_with_groups.parquet",
        "accent_counts_csv":  data_dir / "cv_accent_counts.csv",
        "subset_dir":         data_dir / "cv_subset",
        "staging_dir":        data_dir / "cv_subset" / "_staging",
        "subset_summary_csv": data_dir / "cv_subset_summary.csv",
        "split_info_json":    data_dir / "cv_subset_split_info.json",
    }


_DEFAULT_PATHS        = data_paths(DATA_DIR)
CV_METADATA_DIR       = _DEFAULT_PATHS["cv_metadata_dir"]
CV_VALIDATED_PARQUET  = _DEFAULT_PATHS["validated_parquet"]
CV_ACCENT_COUNTS_CSV  = _DEFAULT_PATHS["accent_counts_csv"]
CV_SUBSET_DIR         = _DEFAULT_PATHS["subset_dir"]
CV_SUBSET_SUMMARY_CSV = _DEFAULT_PATHS["subset_summary_csv"]
CV_SUBSET_SPLIT_INFO  = _DEFAULT_PATHS["split_info_json"]

# source name -> subset folder under data/ (also the default load order).
SOURCE_SUBSETS = {
    "common_voice":          "cv_subset",
    "speechocean762":        "speechocean_subset",
    "speech_accent_archive": "saa_subset",
    "mnsc":                  "mnsc_subset",
}


def subset_paths(data_dir: Path, subset_name: str) -> Dict[str, Path]:
    """Output paths of one subset, e.g. data/<name>/, data/<name>_summary.csv."""
    data_dir = Path(data_dir)
    return {
        "subset_dir":      data_dir / subset_name,
        "summary_csv":     data_dir / f"{subset_name}_summary.csv",
        "split_info_json": data_dir / f"{subset_name}_split_info.json",
    }


# ============================================================
# CONSTANTS
# ============================================================

TARGET_SR       = 16_000
TRIM_TOP_DB     = 40.0
TRIM_FRAME_MS   = 25
TRIM_HOP_MS     = 10
TRIM_PAD_SEC    = 0.1
SILENT_RMS      = 1e-5
TARGET_RMS_DBFS = -20.0
PEAK_LIMIT      = 0.99
FLAC_SUBTYPE    = "PCM_16"

AUDIO_PROCESSING = {
    "description": (
        "decode (soundfile) -> mono (channel mean) -> resample to 16 kHz "
        "(soxr HQ) -> trim leading/trailing silence (frame RMS within "
        "TRIM_TOP_DB of loudest frame, TRIM_PAD_SEC padding) -> duration "
        "filter on trimmed length -> reject RMS < SILENT_RMS -> normalise "
        "to TARGET_RMS_DBFS RMS, then limit peak to PEAK_LIMIT -> FLAC "
        "PCM_16"
    ),
    "target_sr":       TARGET_SR,
    "resampler":       "soxr.resample(quality='HQ')",
    "trim_top_db":     TRIM_TOP_DB,
    "trim_frame_ms":   TRIM_FRAME_MS,
    "trim_hop_ms":     TRIM_HOP_MS,
    "trim_pad_sec":    TRIM_PAD_SEC,
    "silent_rms":      SILENT_RMS,
    "target_rms_dbfs": TARGET_RMS_DBFS,
    "peak_limit":      PEAK_LIMIT,
    "flac_subtype":    FLAC_SUBTYPE,
}

# Groups that are never used as default negatives.
SPECIAL_GROUPS = ("Singaporean", "Malaysian", "Ambiguous", "Unmapped", "Unlabelled")

SHARD_ROWS = 2000

SUBSET_SCHEMA = pa.schema([
    ("clip_id",        pa.string()),
    ("source",         pa.string()),
    ("speaker_id",     pa.string()),
    ("accent_raw",     pa.string()),
    ("accent_group",   pa.string()),
    ("is_singaporean", pa.bool_()),
    ("duration_sec",   pa.float32()),
    ("sentence",       pa.string()),
    ("sampling_rate",  pa.int32()),
    ("audio_flac",     pa.large_binary()),
    ("split",          pa.string()),
])

# Staging shards (pass 2, before splits are assigned): no split, plus rank.
STAGING_SCHEMA = pa.schema(
    [f for f in SUBSET_SCHEMA if f.name != "split"]
    + [pa.field("selection_rank", pa.int64())]
)


# ============================================================
# ARCHIVES
# ============================================================

ARCHIVE_SUFFIXES = (".tar.gz", ".tgz", ".tar", ".tar.bz2", ".tar.xz", ".zip")


def find_latest_archive(raw_dir: Path = RAW_DIR) -> Path:
    """
    Return the newest (by mtime) complete archive in ``raw_dir``.

    Partial downloads (``*.part``) and resume sidecars (``*.checksum``)
    are ignored.
    """
    raw_dir = Path(raw_dir)
    candidates = []
    if raw_dir.is_dir():
        for p in raw_dir.iterdir():
            name = p.name.lower()
            if not p.is_file() or name.endswith((".part", ".checksum")):
                continue
            if name.endswith(ARCHIVE_SUFFIXES):
                candidates.append(p)
    if not candidates:
        raise FileNotFoundError(
            f"No Common Voice archive found in {raw_dir}.\n"
            "Run download_common_voice.py first, or pass --archive <path>."
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def detect_archive_type(path: Path) -> str:
    """Detect 'tar' (any compression) or 'zip' from magic bytes."""
    with open(path, "rb") as fh:
        head = fh.read(512)
    if head.startswith(b"\x1f\x8b"):                   # gzip
        return "tar"
    if head.startswith(b"BZh"):                         # bzip2
        return "tar"
    if head.startswith(b"\xfd7zXZ\x00"):                # xz
        return "tar"
    if head.startswith(b"PK\x03\x04") or head.startswith(b"PK\x05\x06"):
        return "zip"
    if len(head) >= 262 and head[257:262] == b"ustar":  # plain tar
        return "tar"
    raise ValueError(
        f"Unrecognised archive format for {path} "
        f"(first bytes: {head[:8].hex()}). Expected tar(.gz/.bz2/.xz) or zip."
    )


@dataclass
class ArchiveEntry:
    """One regular file inside the archive.

    For tar archives ``read()`` is only valid until the iterator advances
    to the next entry (tar streams cannot seek backwards).
    """
    name: str
    size: int
    read: Callable[[], bytes]

    @property
    def basename(self) -> str:
        return posixpath.basename(self.name)


class ArchiveStreamer:
    """
    Iterate the regular files of an archive without extracting it.

    tar*: tarfile.open(fileobj=..., mode='r|*') -- one forward pass, no
          seeking. Unread member data is skipped by tarfile.
    zip : zipfile.ZipFile -- members are only read when entry.read() is
          called, so unwanted members cost nothing.

    Progress attributes:
        position     compressed bytes consumed (tar) / member index (zip)
        total        archive size in bytes (tar) / member count (zip)
        members_seen regular files yielded so far
        bytes_seen   uncompressed bytes of the members yielded so far
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.kind = detect_archive_type(self.path)
        self.members_seen = 0
        self.bytes_seen = 0
        self._fh = None
        self._tar = None
        self._zip = None
        self._zip_index = 0
        if self.kind == "tar":
            self.total = self.path.stat().st_size
        else:
            self._zip = zipfile.ZipFile(self.path)
            self.total = len(self._zip.infolist())

    @property
    def position(self) -> int:
        if self.kind == "tar":
            try:
                return self._fh.tell() if self._fh else 0
            except (ValueError, OSError):
                return self.total
        return self._zip_index

    @property
    def fraction(self) -> float:
        return (self.position / self.total) if self.total else 0.0

    def __iter__(self) -> Iterator[ArchiveEntry]:
        if self.kind == "tar":
            yield from self._iter_tar()
        else:
            yield from self._iter_zip()

    def _iter_tar(self) -> Iterator[ArchiveEntry]:
        self._fh = open(self.path, "rb")
        self._tar = tarfile.open(fileobj=self._fh, mode="r|*")
        for member in self._tar:
            if not member.isfile():
                continue
            self.members_seen += 1
            self.bytes_seen += member.size

            def _read(m=member) -> bytes:
                f = self._tar.extractfile(m)
                return f.read() if f is not None else b""

            yield ArchiveEntry(member.name, member.size, _read)

    def _iter_zip(self) -> Iterator[ArchiveEntry]:
        for i, info in enumerate(self._zip.infolist()):
            self._zip_index = i + 1
            if info.is_dir():
                continue
            self.members_seen += 1
            self.bytes_seen += info.file_size
            yield ArchiveEntry(
                info.filename, info.file_size,
                lambda inf=info: self._zip.read(inf),
            )

    def close(self) -> None:
        for obj in (self._tar, self._fh, self._zip):
            try:
                if obj is not None:
                    obj.close()
            except Exception:
                pass
        self._tar = self._fh = self._zip = None

    def __enter__(self) -> "ArchiveStreamer":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def ascii_safe(text: str) -> str:
    """Make text printable on any Windows console code page."""
    return str(text).encode("ascii", "backslashreplace").decode("ascii")


def format_seconds(sec: float) -> str:
    sec = max(0, int(sec))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}"


class ProgressTimer:
    """Decide when to print a progress line (every N members or T seconds)."""

    def __init__(self, every_members: int, every_seconds: float = 30.0):
        self.every_members = max(1, int(every_members))
        self.every_seconds = every_seconds
        self.start = time.time()
        self._last_time = self.start
        self._last_members = 0

    @property
    def elapsed(self) -> float:
        return time.time() - self.start

    def due(self, members: int) -> bool:
        now = time.time()
        if (members - self._last_members >= self.every_members
                or now - self._last_time >= self.every_seconds):
            self._last_members = members
            self._last_time = now
            return True
        return False

    def eta(self, fraction: float) -> str:
        if fraction <= 0:
            return "?"
        return format_seconds(self.elapsed * (1.0 - fraction) / fraction)


# ============================================================
# AUDIO
# ============================================================

def decode_audio_bytes(data: bytes) -> Tuple[np.ndarray, int]:
    """Decode an encoded audio file (MP3/FLAC/WAV/...) from memory."""
    audio, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    return audio, int(sr)


def to_mono(x: np.ndarray) -> np.ndarray:
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x.astype(np.float32, copy=False)


def resample_to_target(x: np.ndarray, sr: int, target_sr: int = TARGET_SR) -> np.ndarray:
    if sr == target_sr:
        return x.astype(np.float32, copy=False)
    return soxr.resample(x, sr, target_sr, quality="HQ").astype(np.float32, copy=False)


def trim_silence(x: np.ndarray, sr: int) -> np.ndarray:
    """Trim leading/trailing silence (see module docstring, step 4)."""
    frame = max(1, int(round(sr * TRIM_FRAME_MS / 1000)))
    hop = max(1, int(round(sr * TRIM_HOP_MS / 1000)))
    if len(x) == 0:
        return x
    if len(x) < frame:
        frames = x[None, :]
    else:
        frames = np.lib.stride_tricks.sliding_window_view(x, frame)[::hop]
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    ref = rms.max()
    if ref <= 0:
        return x
    with np.errstate(divide="ignore"):
        db = 20.0 * np.log10(np.maximum(rms, 1e-20) / ref)
    voiced = np.nonzero(db > -TRIM_TOP_DB)[0]
    pad = int(round(TRIM_PAD_SEC * sr))
    start = max(0, voiced[0] * hop - pad)
    end = min(len(x), voiced[-1] * hop + frame + pad)
    return x[start:end]


def normalize_loudness(x: np.ndarray) -> Optional[np.ndarray]:
    """-20 dBFS RMS, then peak-limit to 0.99. Returns None if silent."""
    rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2))) if len(x) else 0.0
    if rms < SILENT_RMS:
        return None
    y = x * (10.0 ** (TARGET_RMS_DBFS / 20.0) / rms)
    peak = float(np.max(np.abs(y)))
    if peak > PEAK_LIMIT:
        y = y * (PEAK_LIMIT / peak)
    return y.astype(np.float32)


def preprocess_waveform(
    x: np.ndarray,
    sr: int,
    min_duration: Optional[float],
    max_duration: Optional[float],
) -> Tuple[Optional[np.ndarray], str]:
    """
    Full chain (module docstring steps 2-7). Returns (audio, reason) where
    reason is "ok", "too_short", "too_long" or "silent"; audio is None
    unless reason == "ok". Output is float32, mono, 16 kHz.
    """
    y = resample_to_target(to_mono(np.asarray(x, dtype=np.float32)), sr)
    y = trim_silence(y, TARGET_SR)
    duration = len(y) / TARGET_SR
    if min_duration is not None and duration < min_duration:
        return None, "too_short"
    if max_duration is not None and duration > max_duration:
        return None, "too_long"
    y = normalize_loudness(y)
    if y is None:
        return None, "silent"
    return y, "ok"


def encode_flac(x: np.ndarray, sr: int = TARGET_SR) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, x, sr, format="FLAC", subtype=FLAC_SUBTYPE)
    return buf.getvalue()


def decode_flac(data: bytes) -> np.ndarray:
    audio, _ = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    return audio


# ============================================================
# TSV
# ============================================================

def read_cv_tsv(source: Union[Path, str, bytes]) -> pd.DataFrame:
    """Read a Common Voice TSV (all columns as str, no NA conversion)."""
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    else:
        source = str(source)
    return pd.read_csv(
        source,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        quoting=csv.QUOTE_NONE,
        on_bad_lines="warn",
        encoding="utf-8",
    )


def find_column(
    df: pd.DataFrame, candidates: Sequence[str], required: bool = True
) -> Optional[str]:
    """Return the first candidate column present in df (case-insensitive)."""
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    if required:
        raise KeyError(
            f"None of the columns {list(candidates)} found. "
            f"Columns present: {list(df.columns)}"
        )
    return None


# ============================================================
# LOADER  (public API for training scripts)
# ============================================================

def load_cv_subset(
    split: Optional[str] = None,
    accent_groups: Optional[Sequence[str]] = None,
    decode_audio: bool = True,
    data_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Load the extracted Common Voice subset as a pandas DataFrame.

    Parameters
    ----------
    split         : "train", "val", "test" or None (all splits)
    accent_groups : optional list of accent_group values to keep
    decode_audio  : True  -> drop audio_flac, add "audio" (float32 numpy
                             array, 16 kHz mono)
                    False -> keep audio_flac bytes; decode lazily with
                             cv_utils.decode_flac(row.audio_flac)
    data_dir      : defaults to <project root>/data

    Memory: decoded audio is ~64 KB per second. The 30,000-clip dataset
    needs several GB decoded, so prefer
    split="train" / a few accent_groups, or decode_audio=False.

    Example
    -------
    >>> from cv_utils import load_cv_subset
    >>> train = load_cv_subset(split="train")
    >>> train[["clip_id", "accent_group", "duration_sec"]].head()
    """
    paths = data_paths(Path(data_dir) if data_dir else DATA_DIR)
    shards = sorted(paths["subset_dir"].glob("part-*.parquet"))
    if not shards:
        raise FileNotFoundError(
            f"No subset shards found in {paths['subset_dir']}.\n"
            "Run extract_common_voice_subset.py first."
        )

    filters: List[tuple] = []
    if split is not None:
        filters.append(("split", "=", split))
    if accent_groups:
        filters.append(("accent_group", "in", list(accent_groups)))

    tables = [
        pq.read_table(str(s), filters=filters or None, schema=SUBSET_SCHEMA)
        for s in shards
    ]
    df = pa.concat_tables(tables).to_pandas()

    if decode_audio:
        df["audio"] = [decode_flac(b) for b in df["audio_flac"]]
        df = df.drop(columns=["audio_flac"])
    return df.reset_index(drop=True)


def load_accent_data(
    sources: Optional[Sequence[str]] = None,
    split: Optional[str] = None,
    accent_groups: Optional[Sequence[str]] = None,
    decode_audio: bool = True,
    data_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Load and concatenate the subsets of several sources (same DataFrame
    format as load_cv_subset).

    Parameters
    ----------
    sources       : names from SOURCE_SUBSETS (default: all, in that order).
                    Sources whose subset folder has no shards are skipped
                    with a printed note.
    split, accent_groups, decode_audio, data_dir : as in load_cv_subset.

    Example
    -------
    >>> from cv_utils import load_accent_data
    >>> train = load_accent_data(split="train")
    >>> train.groupby(["source", "accent_group"]).size()
    """
    names = list(SOURCE_SUBSETS) if sources is None else list(dict.fromkeys(sources))
    unknown = [s for s in names if s not in SOURCE_SUBSETS]
    if unknown:
        raise ValueError(
            f"Unknown source(s): {unknown}. Valid sources: {list(SOURCE_SUBSETS)}"
        )
    root = Path(data_dir) if data_dir else DATA_DIR

    filters: List[tuple] = []
    if split is not None:
        filters.append(("split", "=", split))
    if accent_groups:
        filters.append(("accent_group", "in", list(accent_groups)))

    tables: List[pa.Table] = []
    found_any = False
    for name in names:
        subset_dir = subset_paths(root, SOURCE_SUBSETS[name])["subset_dir"]
        shards = sorted(subset_dir.glob("part-*.parquet"))
        if not shards:
            print(f"Note: no shards for {name} in {subset_dir} - skipped.")
            continue
        found_any = True
        tables.extend(
            pq.read_table(str(s), filters=filters or None, schema=SUBSET_SCHEMA)
            for s in shards
        )
    if not found_any:
        raise FileNotFoundError(
            f"No subset shards found for sources {names} under {root}.\n"
            "Run extract_common_voice_subset.py, prepare_speechocean762.py, "
            "prepare_speech_accent_archive.py and/or prepare_mnsc_singaporean.py first."
        )

    df = pa.concat_tables(tables).to_pandas()
    if decode_audio:
        df["audio"] = [decode_flac(b) for b in df["audio_flac"]]
        df = df.drop(columns=["audio_flac"])
    return df.reset_index(drop=True)


# ============================================================
# HELPERS FOR THE EXTRA SOURCES (prepare_*.py)
# ============================================================

def write_subset_shards(
    table: pa.Table, subset_dir: Path, shard_rows: int = SHARD_ROWS
) -> int:
    """
    Write ``table`` as part-XXXX.parquet shards (SUBSET_SCHEMA) into
    ``subset_dir``. Old part-*.parquet files there are removed first.
    Returns the number of shards written.
    """
    subset_dir = Path(subset_dir)
    subset_dir.mkdir(parents=True, exist_ok=True)
    for old in subset_dir.glob("part-*.parquet"):
        old.unlink()
    table = table.select(SUBSET_SCHEMA.names).cast(SUBSET_SCHEMA)
    n_shards = 0
    for start in range(0, table.num_rows, shard_rows):
        pq.write_table(table.slice(start, shard_rows),
                       str(subset_dir / f"part-{n_shards:04d}.parquet"))
        n_shards += 1
    return n_shards


def ensure_free_space(path: Path, needed_bytes: int, margin: float = 1.2) -> None:
    """Raise OSError if the drive holding ``path`` has < needed*margin free."""
    probe = Path(path).resolve()
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    free = shutil.disk_usage(str(probe)).free
    need = int(needed_bytes * margin)
    print(f"Free disk space: {free / 1e9:.1f} GB "
          f"(need ~{need / 1e9:.2f} GB incl. {margin:.1f}x margin)")
    if free < need:
        raise OSError(
            f"Not enough free disk space at {probe}: {free / 1e9:.2f} GB free, "
            f"{need / 1e9:.2f} GB needed."
        )
