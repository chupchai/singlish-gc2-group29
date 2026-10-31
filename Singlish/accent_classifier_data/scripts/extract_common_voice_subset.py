"""
extract_common_voice_subset.py   (pass 2)

Select a speaker-diverse subset of non-Singaporean Common Voice clips,
stream the archive ONCE, and store only the selected clips as FLAC bytes
inside sharded parquet files. No audio file is ever written to disk.

Purpose
-------
1. Selection (before streaming), per accent group:
     - optional duration pre-filter from clip_durations.tsv
       (raw duration in [min_duration, max_duration + 2 s]; trimming only
       shortens clips, the final filter is on the trimmed length)
     - shuffle speakers (seeded), then round-robin one clip per speaker per
       round, at most --max_clips_per_speaker per speaker, until
       ceil(clips_per_accent * oversample) clips -> maximises distinct
       speakers. Each clip gets a selection_rank (round-robin order).
2. One streaming pass over the archive. Selected clips are decoded in
   memory and run through cv_utils.preprocess_waveform (mono, 16 kHz,
   silence trim, duration filter, -20 dBFS RMS + 0.99 peak limit; see the
   cv_utils module docstring for the exact method), encoded as FLAC in
   memory and buffered into staging parquet shards. The stream stops as
   soon as every selected clip has been seen.
3. Each group is trimmed to --clips_per_accent (lowest selection_rank
   first), then split train/val/test BY SPEAKER within each accent group
   (speaker-disjoint, seeded, clip-weighted).

Usage
-----
    python extract_common_voice_subset.py
    python extract_common_voice_subset.py --accent_groups American British "Hong Kong"
    python extract_common_voice_subset.py --clips_per_accent 1500 --max_clips_per_speaker 20

Outputs (written to <project root>/data/)
-----------------------------------------
    data/cv_subset/part-XXXX.parquet   -- ~2000 rows per shard, columns:
        clip_id, source, speaker_id, accent_raw, accent_group,
        is_singaporean, duration_sec, sentence, sampling_rate,
        audio_flac, split
    data/cv_subset_summary.csv         -- clips / speakers / hours per group x split
    data/cv_subset_split_info.json     -- args, audio processing, speaker lists

Loading
-------
    from cv_utils import load_cv_subset
    train_df = load_cv_subset(split="train")   # column "audio": float32, 16 kHz
"""

import argparse
import json
import math
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from cv_utils import (
    AUDIO_PROCESSING,
    DATA_DIR,
    RAW_DIR,
    SHARD_ROWS,
    SPECIAL_GROUPS,
    STAGING_SCHEMA,
    SUBSET_SCHEMA,
    TARGET_SR,
    ArchiveStreamer,
    ProgressTimer,
    ascii_safe,
    data_paths,
    decode_audio_bytes,
    encode_flac,
    find_latest_archive,
    format_seconds,
    preprocess_waveform,
)


PREFILTER_SLACK_SEC = 2.0
MIN_SPEAKERS_WARNING = 50
SOURCE = "common_voice"
from reference_dataset import CLIP_TARGETS

LAST_STATS: Dict = {}


# ============================================================
# SELECTION (before streaming)
# ============================================================

def select_clips(
    meta: pd.DataFrame,
    groups: List[str],
    clips_per_accent: int,
    max_clips_per_speaker: int,
    min_duration: float,
    max_duration: float,
    oversample: float,
    seed: int,
) -> Tuple[pd.DataFrame, List[dict]]:
    """Round-robin selection across speakers. Returns (selected, plan)."""
    selected = []
    plan = []

    for group in groups:
        count = clips_per_accent[group] if isinstance(clips_per_accent, dict) else clips_per_accent
        target = int(math.ceil(count * oversample))
        sub = meta[meta["accent_group"] == group]
        n_available_raw = len(sub)
        known = sub["duration_sec"].notna()
        in_range = (sub["duration_sec"] >= min_duration) & (
            sub["duration_sec"] <= max_duration + PREFILTER_SLACK_SEC)
        sub = sub[~known | in_range]

        # Sort first so the seeded shuffles do not depend on input row order.
        sub = sub.sort_values(["speaker_id", "clip_id"])
        labels = sub.index.to_numpy()
        rng = random.Random(f"{seed}:{group}")
        by_spk: Dict[str, List[int]] = {}
        for spk, pos in sub.groupby("speaker_id", sort=True).indices.items():
            rows = labels[pos].tolist()
            rng.shuffle(rows)
            by_spk[spk] = rows
        speakers = sorted(by_spk)
        rng.shuffle(speakers)

        taken: List[int] = []
        for rnd in range(max_clips_per_speaker):
            progressed = False
            for spk in speakers:
                rows = by_spk[spk]
                if rnd < len(rows):
                    taken.append(rows[rnd])
                    progressed = True
                    if len(taken) >= target:
                        break
            if len(taken) >= target or not progressed:
                break

        chosen = sub.loc[taken].copy()
        chosen["selection_rank"] = np.arange(len(chosen), dtype=np.int64)
        selected.append(chosen)
        plan.append({
            "accent_group":        group,
            "available_clips":     int(n_available_raw),
            "after_prefilter":     int(len(sub)),
            "available_speakers":  int(len(speakers)),
            "selected_speakers":   int(chosen["speaker_id"].nunique()),
            "selected_clips":      int(len(chosen)),
            "target_with_oversample": target,
        })

    sel = pd.concat(selected, ignore_index=True) if selected else meta.iloc[0:0]
    return sel, plan


def print_plan(plan: List[dict], clips_per_accent: int) -> None:
    print()
    print("=" * 70)
    print("SELECTION PLAN (per accent group)")
    print("=" * 70)
    print(f"{'accent_group':<18} {'avail clips':>11} {'prefilt':>8} "
          f"{'avail spk':>9} {'sel spk':>8} {'sel clips':>9} {'target':>7}")
    print("-" * 70)
    for p in plan:
        print(f"{p['accent_group']:<18} {p['available_clips']:>11,} "
              f"{p['after_prefilter']:>8,} {p['available_speakers']:>9,} "
              f"{p['selected_speakers']:>8,} {p['selected_clips']:>9,} "
              f"{p['target_with_oversample']:>7,}")
    for p in plan:
        if p["selected_speakers"] < MIN_SPEAKERS_WARNING:
            print(f"  WARNING: {p['accent_group']} has only "
                  f"{p['selected_speakers']} speakers (< {MIN_SPEAKERS_WARNING}); "
                  "results for this group may not generalise.")
        count = clips_per_accent[p["accent_group"]] if isinstance(clips_per_accent, dict) else clips_per_accent
        if p["selected_clips"] < count:
            print(f"  WARNING: {p['accent_group']} cannot reach "
                  f"{count} clips (selected {p['selected_clips']}).")


# ============================================================
# STREAMING PASS
# ============================================================

def _write_staging(rows: List[dict], staging_dir: Path, index: int) -> Path:
    table = pa.Table.from_pylist(rows, schema=STAGING_SCHEMA)
    path = staging_dir / f"stage-{index:04d}.parquet"
    pq.write_table(table, str(path))
    return path


def stream_and_process(
    archive: Path,
    selected: pd.DataFrame,
    staging_dir: Path,
    min_duration: float,
    max_duration: float,
    shard_rows: int,
    progress_every: int,
) -> Dict:
    print()
    print("=" * 70)
    print("PASS 2: STREAMING ARCHIVE (selected clips only, in memory)")
    print("=" * 70)
    print(f"Archive : {archive.name}")
    print(f"Selected clips to find : {len(selected):,}")
    print()

    wanted: Dict[str, dict] = {
        r["path"]: r for r in selected.to_dict("records")
    }
    seen: set = set()
    buffer: List[dict] = []
    shard_index = 0
    kept = 0
    skips = {"too_short": 0, "too_long": 0, "silent": 0, "decode_error": 0}
    stopped_early = False
    timer = ProgressTimer(progress_every)

    def progress(stream: ArchiveStreamer) -> None:
        print(f"  members {stream.members_seen:,} | found {len(seen):,}/{len(wanted):,} | "
              f"kept {kept:,} | {100.0 * stream.fraction:5.1f}% of archive | "
              f"elapsed {format_seconds(timer.elapsed)} | "
              f"ETA (full scan) {timer.eta(stream.fraction)}")

    with ArchiveStreamer(archive) as stream:
        for entry in stream:
            base = entry.basename
            if base in wanted and base not in seen:
                seen.add(base)
                info = wanted[base]
                try:
                    audio, sr = decode_audio_bytes(entry.read())
                    y, reason = preprocess_waveform(audio, sr, min_duration, max_duration)
                except Exception as exc:
                    print(f"  decode error ({ascii_safe(base)}): {ascii_safe(exc)}")
                    y, reason = None, "decode_error"

                if y is None:
                    skips[reason] += 1
                else:
                    group = info["accent_group"]
                    buffer.append({
                        "clip_id":        info["clip_id"],
                        "source":         SOURCE,
                        "speaker_id":     info["speaker_id"],
                        "accent_raw":     info["accent_raw"],
                        "accent_group":   group,
                        "is_singaporean": group == "Singaporean",
                        "duration_sec":   float(len(y) / TARGET_SR),
                        "sentence":       info["sentence"],
                        "sampling_rate":  TARGET_SR,
                        "audio_flac":     encode_flac(y),
                        "selection_rank": int(info["selection_rank"]),
                    })
                    kept += 1
                    if len(buffer) >= shard_rows:
                        _write_staging(buffer, staging_dir, shard_index)
                        shard_index += 1
                        buffer = []

                if len(seen) == len(wanted):
                    stopped_early = True
                    progress(stream)
                    print("  All selected clips seen -> stopping the stream early.")
                    break

            if timer.due(stream.members_seen):
                progress(stream)

        members_seen = stream.members_seen
        fraction = stream.fraction

    if buffer:
        _write_staging(buffer, staging_dir, shard_index)
        shard_index += 1

    not_found = len(wanted) - len(seen)
    print()
    print(f"Members scanned     : {members_seen:,} ({100.0 * fraction:.1f}% of archive)")
    print(f"Selected clips seen : {len(seen):,} / {len(wanted):,}")
    print(f"Not found in archive: {not_found:,}")
    print(f"Kept after filters  : {kept:,}")
    for k, v in skips.items():
        print(f"Skipped ({k:<12}): {v:,}")
    print(f"Elapsed             : {format_seconds(timer.elapsed)}")
    return {
        "members_seen":   members_seen,
        "selected_seen":  len(seen),
        "not_found":      not_found,
        "kept_processed": kept,
        "skips":          skips,
        "stopped_early":  stopped_early,
        "staging_shards": shard_index,
        "elapsed_sec":    timer.elapsed,
    }


# ============================================================
# TRIM + SPEAKER-DISJOINT SPLIT
# ============================================================

def assign_splits(
    group_df: pd.DataFrame,
    group: str,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> Dict[str, str]:
    """Greedy clip-weighted speaker split within one accent group."""
    counts = group_df.groupby("speaker_id").size()
    speakers = sorted(counts.index)
    rng = random.Random(f"{seed}:{group}:split")
    rng.shuffle(speakers)
    n = len(speakers)
    total = int(counts.sum())

    if n < 3:
        print(f"  WARNING: {group} has only {n} speaker(s); all clips go to train.")
        return {s: "train" for s in speakers}

    assignment: Dict[str, str] = {}
    i = 0

    def take(split: str, fraction: float, reserve: int) -> None:
        nonlocal i
        if fraction <= 0:
            return
        target = fraction * total
        c = 0
        chosen = 0
        while i < n - reserve and (chosen == 0 or c < target):
            assignment[speakers[i]] = split
            c += int(counts[speakers[i]])
            chosen += 1
            i += 1

    take("test", test_fraction, reserve=1 + (1 if val_fraction > 0 else 0))
    take("val", val_fraction, reserve=1)
    for s in speakers[i:]:
        assignment[s] = "train"
    return assignment


def finalise(
    staging_dir: Path,
    subset_dir: Path,
    groups: List[str],
    clips_per_accent: int,
    val_fraction: float,
    test_fraction: float,
    seed: int,
    shard_rows: int,
) -> pd.DataFrame:
    print()
    print("=" * 70)
    print("TRIM TO TARGET + SPEAKER-DISJOINT SPLIT")
    print("=" * 70)

    stage_files = sorted(staging_dir.glob("stage-*.parquet"))
    meta_cols = [f.name for f in STAGING_SCHEMA if f.name != "audio_flac"]
    meta = pd.concat(
        [pq.read_table(str(f), columns=meta_cols).to_pandas() for f in stage_files],
        ignore_index=True,
    )

    keep_parts = []
    for group in groups:
        g = meta[meta["accent_group"] == group].sort_values("selection_rank")
        count = clips_per_accent[group] if isinstance(clips_per_accent, dict) else clips_per_accent
        g = g.head(count).copy()
        if g.empty:
            print(f"  WARNING: {group}: no clips kept.")
            continue
        split_map = assign_splits(g, group, val_fraction, test_fraction, seed)
        g["split"] = g["speaker_id"].map(split_map)
        keep_parts.append(g)
    kept = pd.concat(keep_parts, ignore_index=True)
    split_of = dict(zip(kept["clip_id"], kept["split"]))

    # Rewrite staging shards -> final shards (one staging shard in memory at a time).
    subset_dir.mkdir(parents=True, exist_ok=True)
    out_index = 0
    pending: List[pa.Table] = []
    pending_rows = 0

    def flush(force: bool = False) -> None:
        nonlocal out_index, pending, pending_rows
        while pending_rows >= shard_rows or (force and pending_rows > 0):
            table = pa.concat_tables(pending)
            n = min(shard_rows, table.num_rows)
            pq.write_table(table.slice(0, n),
                           str(subset_dir / f"part-{out_index:04d}.parquet"))
            out_index += 1
            rest = table.slice(n)
            pending = [rest] if rest.num_rows else []
            pending_rows = rest.num_rows

    for f in stage_files:
        t = pq.read_table(str(f))
        ids = t.column("clip_id").to_pylist()
        mask = pa.array([cid in split_of for cid in ids])
        t = t.filter(mask)
        if t.num_rows == 0:
            continue
        splits = pa.array([split_of[cid] for cid in t.column("clip_id").to_pylist()],
                          type=pa.string())
        t = t.drop_columns(["selection_rank"]).append_column("split", splits)
        t = t.select(SUBSET_SCHEMA.names).cast(SUBSET_SCHEMA)
        pending.append(t)
        pending_rows += t.num_rows
        flush()
    flush(force=True)

    shutil.rmtree(staging_dir, ignore_errors=True)
    print(f"  Wrote {out_index} shard(s), {len(kept):,} clips -> {subset_dir}")
    return kept


def summarise(kept: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (group, split), g in kept.groupby(["accent_group", "split"]):
        rows.append({"accent_group": group, "split": split,
                     "n_clips": len(g), "n_speakers": g["speaker_id"].nunique(),
                     "hours": round(float(g["duration_sec"].sum()) / 3600.0, 4)})
    for group, g in kept.groupby("accent_group"):
        rows.append({"accent_group": group, "split": "ALL",
                     "n_clips": len(g), "n_speakers": g["speaker_id"].nunique(),
                     "hours": round(float(g["duration_sec"].sum()) / 3600.0, 4)})
    order = {"train": 0, "val": 1, "test": 2, "ALL": 3}
    df = pd.DataFrame(rows)
    df["_o"] = df["split"].map(order)
    return df.sort_values(["accent_group", "_o"]).drop(columns="_o").reset_index(drop=True)


# ============================================================
# CLI
# ============================================================

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pass 2: extract a speaker-diverse Common Voice subset as "
                    "FLAC-in-parquet shards (no audio files written)."
    )
    p.add_argument("--archive", type=Path, default=None,
                   help="Archive path (default: newest archive in <project root>/raw).")
    p.add_argument("--select_new_clips", action="store_true",
                   help="Sample a custom dataset instead of reproducing the fixed 40,000-clip manifest.\n"
                        "Required when changing selection, split or audio settings.")
    p.add_argument("--data_dir", type=Path, default=DATA_DIR,
                   help="Data folder (default: <project root>/data).")
    p.add_argument("--accent_groups", nargs="+", default=None,
                   help="Accent groups to extract (default: the six Common Voice training groups).")
    p.add_argument("--clips_per_accent", type=int, default=None,
                   help="Override the default per-accent targets (18,871 clips total).")
    p.add_argument("--max_clips_per_speaker", type=int, default=200)
    p.add_argument("--min_duration", type=float, default=3.0)
    p.add_argument("--max_duration", type=float, default=6.0)
    p.add_argument("--oversample", type=float, default=3.0,
                   help="Select this many times clips_per_accent to absorb "
                        "clips lost to filters (default 3.0).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val_fraction", type=float, default=0.15)
    p.add_argument("--test_fraction", type=float, default=0.15)
    p.add_argument("--shard_rows", type=int, default=SHARD_ROWS)
    p.add_argument("--progress_every", type=int, default=20000,
                   help="Print progress every N members (and at least every 30 s).")
    return p.parse_args(argv)


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
        if getattr(args, name) is not None and getattr(args, name) <= 0:
            return f"--{name} must be positive"
    if args.oversample < 1.0:
        return "--oversample must be >= 1.0"
    return None


# ============================================================
# MAIN
# ============================================================

def main(argv: Optional[List[str]] = None) -> int:
    global LAST_STATS
    args = parse_args(argv)
    if not args.select_new_clips:
        from reference_dataset import prepare_reference
        return prepare_reference("common_voice", args)
    paths = data_paths(args.data_dir)

    error = validate_args(args)
    if error:
        print(f"ERROR: {error}")
        return 1

    if not paths["validated_parquet"].is_file():
        print(f"ERROR: {paths['validated_parquet']} not found.")
        print("Run scan_common_voice_metadata.py first.")
        return 1
    try:
        archive = args.archive if args.archive else find_latest_archive(RAW_DIR)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        return 1
    if not archive.is_file():
        print(f"ERROR: archive not found: {archive}")
        return 1

    meta = pq.read_table(str(paths["validated_parquet"])).to_pandas()
    meta = meta.drop_duplicates(subset="path").reset_index(drop=True)
    conflicts = meta.groupby("speaker_id")["accent_group"].nunique()
    meta = meta[~meta["speaker_id"].isin(conflicts[conflicts > 1].index)].reset_index(drop=True)
    available = sorted(meta["accent_group"].unique())
    if args.accent_groups is None and args.clips_per_accent is None:
        missing = sorted(set(CLIP_TARGETS) - set(available))
        if missing:
            print(f"ERROR: missing Common Voice groups: {missing}")
            return 1
    if args.accent_groups:
        unknown = [g for g in args.accent_groups if g not in available]
        if unknown:
            print(f"ERROR: unknown accent group(s): {unknown}")
            print(f"Available groups: {available}")
            return 1
        groups = list(dict.fromkeys(args.accent_groups))
    else:
        groups = [g for g in available if g not in SPECIAL_GROUPS
                  and (args.clips_per_accent is not None or g in CLIP_TARGETS)]
    if args.clips_per_accent is None and any(g not in CLIP_TARGETS for g in groups):
        print("ERROR: use --clips_per_accent for custom accent groups.")
        return 1
    targets = args.clips_per_accent if args.clips_per_accent is not None else CLIP_TARGETS
    if not groups:
        print("ERROR: no accent groups to extract.")
        return 1

    print("=" * 70)
    print("EXTRACT COMMON VOICE SUBSET")
    print("=" * 70)
    print(f"Archive               : {archive.name}")
    print(f"Accent groups         : {groups}")
    print(f"Clips per accent      : {targets}")
    print(f"Max clips per speaker : {args.max_clips_per_speaker}")
    print(f"Duration range        : {args.min_duration} - {args.max_duration} s (after trim)")
    print(f"Oversample            : {args.oversample}")
    print(f"Split (val/test)      : {args.val_fraction} / {args.test_fraction} by speaker")
    print(f"Seed                  : {args.seed}")

    selected, plan = select_clips(
        meta, groups, targets, args.max_clips_per_speaker,
        args.min_duration, args.max_duration, args.oversample, args.seed,
    )
    print_plan(plan, targets)
    if selected.empty:
        print("ERROR: nothing selected.")
        return 1

    # Remove only script-owned outputs from a previous run.
    subset_dir, staging_dir = paths["subset_dir"], paths["staging_dir"]
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    for old in subset_dir.glob("part-*.parquet"):
        old.unlink()
    staging_dir.mkdir(parents=True, exist_ok=True)

    stream_stats = stream_and_process(
        archive, selected, staging_dir, args.min_duration, args.max_duration,
        args.shard_rows, args.progress_every,
    )
    if stream_stats["kept_processed"] == 0:
        shutil.rmtree(staging_dir, ignore_errors=True)
        print("ERROR: no clips survived processing.")
        return 1

    kept = finalise(staging_dir, subset_dir, groups, targets,
                    args.val_fraction, args.test_fraction, args.seed, args.shard_rows)

    if args.clips_per_accent is None:
        counts = kept.groupby("accent_group").size().to_dict()
        expected = {g: targets[g] for g in groups}
        if counts != expected:
            print(f"ERROR: clip shortfall: got {counts}, expected {expected}. "
                  "Increase --oversample or --max_clips_per_speaker and rerun.")
            return 1

    summary = summarise(kept)
    summary.to_csv(paths["subset_summary_csv"], index=False)

    print()
    print("=" * 70)
    print("SUBSET SUMMARY")
    print("=" * 70)
    print(f"{'accent_group':<18} {'split':<6} {'clips':>7} {'speakers':>9} {'hours':>8}")
    print("-" * 70)
    for r in summary.itertuples(index=False):
        print(f"{r.accent_group:<18} {r.split:<6} {r.n_clips:>7,} "
              f"{r.n_speakers:>9,} {r.hours:>8.3f}")

    per_group = {}
    for group, g in kept.groupby("accent_group"):
        per_group[group] = {
            split: {"n_clips": int(len(s)),
                    "speakers": sorted(s["speaker_id"].unique().tolist())}
            for split, s in g.groupby("split")
        }
    run_args = {k: v for k, v in vars(args).items() if k not in ("archive", "data_dir")}
    info = {
        "source":           SOURCE,
        "archive":          archive.name,
        "args":             run_args,
        "accent_groups":    groups,
        "audio_processing": AUDIO_PROCESSING,
        "prefilter_slack_sec": PREFILTER_SLACK_SEC,
        "selection_plan":   plan,
        "stream":           {k: v for k, v in stream_stats.items() if k != "elapsed_sec"},
        "split_strategy":   "speaker-disjoint within each accent group, "
                            "greedy clip-weighted, seeded",
        "groups":           per_group,
    }
    with open(paths["split_info_json"], "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=2)

    print()
    print("=" * 70)
    print("PASS 2 COMPLETE")
    print("=" * 70)
    print(f"  Saved shards    -> {subset_dir}")
    print(f"  Saved summary   -> {paths['subset_summary_csv']}")
    print(f"  Saved split info-> {paths['split_info_json']}")
    print()
    print("Load in other scripts with:")
    print("  from cv_utils import load_cv_subset")
    print("  train_df = load_cv_subset(split='train')")

    LAST_STATS = {
        "archive": archive.name,
        "groups": groups,
        "plan": plan,
        **stream_stats,
        "n_kept": int(len(kept)),
    }
    return 0


if __name__ == "__main__":
    sys.exit(main())
