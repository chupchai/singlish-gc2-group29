"""
scan_common_voice_metadata.py   (pass 1)

Stream the Common Voice archive, keep ONLY its *.tsv metadata files, and
map every validated clip's accent string to an accent group.

Purpose
-------
The archive is never extracted. Only *.tsv members are written to disk
(data/cv_metadata/); audio clips are skipped inside the stream. From
validated.tsv each clip gets an `accent_group` using ACCENT_RULES below.
The per-group counts tell you how much data each accent really has before
committing to pass 2 (extract_common_voice_subset.py).

Edit ACCENT_RULES (top of this file) and re-run if the Unmapped list shows
accent strings you want to include.

Usage
-----
    python scan_common_voice_metadata.py                       # newest archive in raw/
    python scan_common_voice_metadata.py --archive ../raw/cv.tar.gz
    python scan_common_voice_metadata.py --full_scan           # never stop early

Tar archives are read in one forward pass. If all TSVs (validated.tsv and
clip_durations.tsv) appear before the first audio clip the scan stops at
the first clip; otherwise it reads through the whole archive.

Outputs (written to <project root>/data/)
-----------------------------------------
    data/cv_metadata/*.tsv                   -- TSV files copied from the archive
    data/cv_validated_with_groups.parquet    -- validated.tsv + accent_group (no audio)
    data/cv_accent_counts.csv                -- clips / speakers / hours per group
"""

import argparse
import posixpath
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from cv_utils import (
    DATA_DIR,
    RAW_DIR,
    ArchiveStreamer,
    ProgressTimer,
    ascii_safe,
    data_paths,
    find_column,
    find_latest_archive,
    format_seconds,
    read_cv_tsv,
)


# ============================================================
# ACCENT RULES  (edit me)
# ============================================================
# Ordered list of (accent_group, [regex, ...]). Each '|'-separated part of
# the raw accent string is matched case-insensitively with re.search;
# FIRST MATCH WINS, so keep SPECIFIC rules above GENERAL ones.
# group None = force "Unmapped" (exclusions that would otherwise hit a
# later rule, e.g. "West Indies" -> \bindia, "Latin American" -> american).
# Never use the word "english" as a keyword: it is in almost every string.
#
# Multi-accent strings ("A|B"): if all mapped parts agree -> that group;
# if they disagree -> "Ambiguous"; if none map -> "Unmapped";
# empty accent field -> "Unlabelled".
# Singaporean is kept as its own group and is NEVER a default negative.

ACCENT_RULES = [
    # --- exclusions first: stop false positives below ---
    (None,               [r"west indi", r"caribbean", r"jamaica", r"bermuda",
                          r"(south|latin|central) america",
                          r"(american|native)\s+indian"]),   # not South Asian
    ("Singaporean",      [r"singapor"]),
    ("Malaysian",        [r"malaysia"]),
    ("Hong Kong",        [r"hong\s*kong"]),          # before Mainland Chinese
    ("Taiwanese",        [r"taiwan"]),               # before Mainland Chinese
    ("Mainland Chinese", [r"china", r"chinese", r"mandarin"]),
    ("Korean",           [r"korea"]),
    ("Japanese",         [r"japan"]),
    ("Filipino",         [r"philippin", r"filipin", r"tagalog"]),
    ("Indonesian",       [r"indonesia"]),
    ("Vietnamese",       [r"vietnam"]),
    ("Thai",             [r"\bthai"]),
    ("Myanmar",          [r"myanmar", r"burm"]),
    # \bindian?\b: the trailing \b stops "Indiana" matching
    ("South Asian",      [r"\bindian?\b", r"south\s*asia", r"pakistan", r"bangladesh",
                          r"sri\s*lanka", r"tamil", r"nepal"]),
    ("American",         [r"united states", r"\bamerican\b", r"\busa\b"]),
    ("British",          [r"england", r"british", r"united kingdom", r"\buk\b",
                          r"scotland", r"scottish", r"wales", r"welsh"]),
    ("Australian",       [r"australia"]),
]

_COMPILED_RULES = [
    (group, [re.compile(p, re.IGNORECASE) for p in patterns])
    for group, patterns in ACCENT_RULES
]

# Tar members we treat as audio clips (for the early-stop decision).
_CLIP_SUFFIXES = (".mp3", ".wav", ".flac", ".ogg", ".opus", ".m4a")

# Needed before we may stop at the first clip.
_REQUIRED_FOR_EARLY_STOP = ("validated.tsv", "clip_durations.tsv")

LAST_STATS: Dict = {}


# ============================================================
# ACCENT MAPPING
# ============================================================

def map_accent_part(part: str) -> Optional[str]:
    """Map ONE accent string (no '|') to a group, or None if unmapped."""
    for group, patterns in _COMPILED_RULES:
        if any(p.search(part) for p in patterns):
            return group          # None for exclusion rules
    return None


def map_accent(raw: str) -> str:
    """Map a raw (possibly '|'-separated) accent field to an accent_group."""
    raw = (raw or "").strip()
    if not raw:
        return "Unlabelled"
    parts = [p.strip() for p in raw.split("|") if p.strip()]
    groups = {g for g in (map_accent_part(p) for p in parts) if g is not None}
    if len(groups) == 1:
        return groups.pop()
    if len(groups) > 1:
        return "Ambiguous"
    return "Unmapped"


# ============================================================
# PASS 1: STREAM THE ARCHIVE, KEEP ONLY TSVs
# ============================================================

def _is_clip(name: str) -> bool:
    lower = name.lower()
    return lower.endswith(_CLIP_SUFFIXES) or "/clips/" in "/" + lower


def stream_tsvs(
    archive: Path,
    meta_dir: Path,
    progress_every: int,
    full_scan: bool,
) -> Dict:
    """Copy every *.tsv member of the archive into meta_dir."""
    print("=" * 70)
    print("PASS 1: STREAMING ARCHIVE (TSV FILES ONLY)")
    print("=" * 70)
    print(f"Archive : {archive.name}")
    print(f"Size    : {archive.stat().st_size / 1e9:.3f} GB")

    meta_dir.mkdir(parents=True, exist_ok=True)
    found: Dict[str, str] = {}     # basename -> member name kept
    stopped_early = False
    first_clip_seen = False
    timer = ProgressTimer(progress_every)

    with ArchiveStreamer(archive) as stream:
        print(f"Type    : {stream.kind}")
        print()
        for entry in stream:
            name = entry.name
            base = entry.basename

            if base.lower().endswith(".tsv"):
                prev = found.get(base)
                keep = prev is None or ("/en/" not in "/" + prev and "/en/" in "/" + name)
                if prev is not None:
                    print(f"  WARNING: duplicate TSV name {ascii_safe(base)}: "
                          f"{ascii_safe(prev)} vs {ascii_safe(name)} -> "
                          f"keeping {ascii_safe(name if keep else prev)}")
                if keep:
                    (meta_dir / base).write_bytes(entry.read())
                    found[base] = name
                    print(f"  Saved TSV {ascii_safe(name)} "
                          f"({entry.size / 1e6:.1f} MB)")

            elif stream.kind == "tar" and not first_clip_seen and _is_clip(name):
                first_clip_seen = True
                have_all = all(t in found for t in _REQUIRED_FOR_EARLY_STOP)
                if have_all and not full_scan:
                    stopped_early = True
                    print("  First audio clip reached and all required TSVs "
                          "already captured -> stopping early.")
                    break
                missing = [t for t in _REQUIRED_FOR_EARLY_STOP if t not in found]
                if missing:
                    print(f"  First audio clip reached; still missing {missing} "
                          "-> reading through the whole archive.")
                else:
                    print("  First audio clip reached; --full_scan set "
                          "-> reading the whole archive.")

            if timer.due(stream.members_seen):
                pct = 100.0 * stream.fraction
                print(f"  members {stream.members_seen:,} | "
                      f"{stream.bytes_seen / 1e9:.2f} GB uncompressed | "
                      f"{pct:5.1f}% of archive | "
                      f"elapsed {format_seconds(timer.elapsed)} | "
                      f"ETA {timer.eta(stream.fraction)}")

        members_seen = stream.members_seen
        bytes_seen = stream.bytes_seen
        kind = stream.kind

    print()
    print(f"Members seen   : {members_seen:,}")
    print(f"Bytes seen     : {bytes_seen / 1e9:.3f} GB (uncompressed)")
    print(f"Elapsed        : {format_seconds(timer.elapsed)}")
    print(f"Stopped early  : {stopped_early}")
    print(f"TSVs found     : {sorted(found)}")
    return {
        "archive_type":  kind,
        "stopped_early": stopped_early,
        "members_seen":  members_seen,
        "bytes_seen":    bytes_seen,
        "tsvs_found":    sorted(found),
        "elapsed_sec":   timer.elapsed,
    }


# ============================================================
# BUILD METADATA TABLE
# ============================================================

META_SCHEMA = pa.schema([
    ("clip_id",      pa.string()),
    ("path",         pa.string()),
    ("speaker_id",   pa.string()),
    ("sentence",     pa.string()),
    ("accent_raw",   pa.string()),
    ("accent_group", pa.string()),
    ("duration_sec", pa.float32()),
    ("gender",       pa.string()),
    ("age",          pa.string()),
])


def load_durations(meta_dir: Path) -> Optional[pd.Series]:
    """Return clip basename -> duration in seconds, or None if unavailable."""
    path = meta_dir / "clip_durations.tsv"
    if not path.is_file():
        return None
    df = read_cv_tsv(path)
    clip_col = find_column(df, ["clip", "path"], required=False)
    dur_col = next((c for c in df.columns if c.lower().startswith("duration")), None)
    if clip_col is None or dur_col is None:
        print(f"  WARNING: clip_durations.tsv has unexpected columns {list(df.columns)}; "
              "ignoring durations.")
        return None
    ms = pd.to_numeric(df[dur_col], errors="coerce")
    scale = 1000.0 if "ms" in dur_col.lower() or ms.median() > 100 else 1.0
    clips = df[clip_col].map(posixpath.basename)
    return pd.Series((ms / scale).to_numpy(), index=clips.to_numpy())


def build_metadata(meta_dir: Path) -> pd.DataFrame:
    print()
    print("=" * 70)
    print("MAPPING ACCENTS (validated.tsv)")
    print("=" * 70)

    validated = read_cv_tsv(meta_dir / "validated.tsv")
    print(f"validated.tsv rows : {len(validated):,}")
    print(f"Columns            : {list(validated.columns)}")

    path_col = find_column(validated, ["path"])
    spk_col = find_column(validated, ["client_id"])
    acc_col = find_column(validated, ["accents", "accent"])
    sent_col = find_column(validated, ["sentence"], required=False)
    gender_col = find_column(validated, ["gender"], required=False)
    age_col = find_column(validated, ["age"], required=False)
    print(f"Accent column      : {acc_col}")

    def col(name: Optional[str]) -> pd.Series:
        if name is None:
            return pd.Series([""] * len(validated), index=validated.index, dtype=object)
        return validated[name].astype(object)

    paths = validated[path_col].map(posixpath.basename)
    meta = pd.DataFrame({
        "clip_id":    paths.map(lambda p: posixpath.splitext(p)[0]),
        "path":       paths,
        "speaker_id": col(spk_col),
        "sentence":   col(sent_col),
        "accent_raw": col(acc_col).map(lambda s: (s or "").strip()),
    })

    # Map each distinct raw string once (validated.tsv has millions of rows).
    mapping = {raw: map_accent(raw) for raw in meta["accent_raw"].unique()}
    meta["accent_group"] = meta["accent_raw"].map(mapping)

    durations = load_durations(meta_dir)
    if durations is not None:
        durations = durations[~durations.index.duplicated()]
        meta["duration_sec"] = meta["path"].map(durations).astype("float32")
        n_known = int(meta["duration_sec"].notna().sum())
        print(f"Durations          : {n_known:,} / {len(meta):,} clips from clip_durations.tsv")
    else:
        meta["duration_sec"] = np.full(len(meta), np.nan, dtype="float32")
        print("Durations          : clip_durations.tsv not available")

    meta["gender"] = col(gender_col)
    meta["age"] = col(age_col)
    return meta


def accent_counts(meta: pd.DataFrame) -> pd.DataFrame:
    g = meta.groupby("accent_group")
    per_spk = meta.groupby(["accent_group", "speaker_id"]).size()
    counts = pd.DataFrame({
        "n_clips":    g.size(),
        "n_speakers": g["speaker_id"].nunique(),
        "median_clips_per_speaker": per_spk.groupby(level=0).median(),
    })
    if meta["duration_sec"].notna().any():
        counts["total_hours"] = (g["duration_sec"].sum() / 3600.0).round(3)
    else:
        counts["total_hours"] = np.nan
    counts = counts.reset_index().sort_values(
        ["n_speakers", "n_clips"], ascending=False
    ).reset_index(drop=True)
    return counts


def top_unmapped(meta: pd.DataFrame, n: int) -> pd.DataFrame:
    un = meta[meta["accent_group"] == "Unmapped"]
    if un.empty:
        return pd.DataFrame(columns=["accent_raw", "n_clips", "n_speakers"])
    g = un.groupby("accent_raw")
    out = pd.DataFrame({
        "n_clips": g.size(),
        "n_speakers": g["speaker_id"].nunique(),
    }).reset_index()
    return out.sort_values(["n_clips", "n_speakers"], ascending=False).head(n)


# ============================================================
# CLI
# ============================================================

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pass 1: stream the Common Voice archive, keep only TSVs, "
                    "map accents to groups and count clips/speakers."
    )
    p.add_argument("--archive", type=Path, default=None,
                   help="Archive path (default: newest archive in <project root>/raw).")
    p.add_argument("--data_dir", type=Path, default=DATA_DIR,
                   help="Output folder (default: <project root>/data).")
    p.add_argument("--progress_every", type=int, default=20000,
                   help="Print progress every N members (and at least every 30 s).")
    p.add_argument("--full_scan", action="store_true",
                   help="Never stop early; read the whole archive.")
    p.add_argument("--top_unmapped", type=int, default=50,
                   help="How many Unmapped raw accent strings to print (default 50).")
    return p.parse_args(argv)


# ============================================================
# MAIN
# ============================================================

def main(argv: Optional[List[str]] = None) -> int:
    global LAST_STATS
    args = parse_args(argv)
    paths = data_paths(args.data_dir)

    try:
        archive = args.archive if args.archive else find_latest_archive(RAW_DIR)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        return 1
    if not archive.is_file():
        print(f"ERROR: archive not found: {archive}")
        return 1

    stats = stream_tsvs(archive, paths["cv_metadata_dir"],
                        args.progress_every, args.full_scan)
    if "validated.tsv" not in stats["tsvs_found"]:
        print()
        print("ERROR: validated.tsv was not found in the archive.")
        print(f"TSV files found: {stats['tsvs_found'] or 'none'}")
        LAST_STATS = {"archive": archive.name, **stats}
        return 1

    meta = build_metadata(paths["cv_metadata_dir"])

    paths["data_dir"].mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(meta, schema=META_SCHEMA, preserve_index=False)
    pq.write_table(table, str(paths["validated_parquet"]))

    counts = accent_counts(meta)
    counts.to_csv(paths["accent_counts_csv"], index=False)

    print()
    print("=" * 70)
    print("ACCENT GROUP COUNTS (sorted by speakers)")
    print("=" * 70)
    print(f"{'accent_group':<18} {'clips':>10} {'speakers':>9} "
          f"{'med/spk':>8} {'hours':>9}")
    print("-" * 70)
    for r in counts.itertuples(index=False):
        hours = "" if pd.isna(r.total_hours) else f"{r.total_hours:9.2f}"
        print(f"{r.accent_group:<18} {r.n_clips:>10,} {r.n_speakers:>9,} "
              f"{r.median_clips_per_speaker:>8.1f} {hours:>9}")

    unmapped = top_unmapped(meta, args.top_unmapped)
    print()
    print("=" * 70)
    print(f"TOP {args.top_unmapped} UNMAPPED RAW ACCENT STRINGS")
    print("=" * 70)
    if unmapped.empty:
        print("(none)")
    for r in unmapped.itertuples(index=False):
        print(f"{r.n_clips:>8,} clips {r.n_speakers:>6,} spk  {ascii_safe(r.accent_raw)}")

    print()
    print("=" * 70)
    print("PASS 1 COMPLETE")
    print("=" * 70)
    print(f"  Saved TSVs      -> {paths['cv_metadata_dir']}")
    print(f"  Saved metadata  -> {paths['validated_parquet']}")
    print(f"  Saved counts    -> {paths['accent_counts_csv']}")
    print()
    print("If useful accents are Unmapped, edit ACCENT_RULES at the top of")
    print("scan_common_voice_metadata.py and re-run. Next step:")
    print("  python scripts/extract_common_voice_subset.py")

    LAST_STATS = {
        "archive": archive.name,
        **stats,
        "n_validated_rows": len(meta),
        "groups": {r.accent_group: {"n_clips": int(r.n_clips),
                                    "n_speakers": int(r.n_speakers)}
                   for r in counts.itertuples(index=False)},
    }
    return 0


if __name__ == "__main__":
    sys.exit(main())
