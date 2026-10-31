"""
prepare_speechocean762.py

Build the "Mainland Chinese" accent group from speechocean762 (English read
by native Mandarin speakers) as FLAC-in-parquet shards with exactly the
Common Voice subset schema. No audio file is ever written to disk.

Why this source
---------------
Common Voice English has only ~12 labelled Mainland Chinese speakers, far
too few to train on. speechocean762 has 250 native-Mandarin speakers
(5,000 utterances, 20 per speaker), about half of them children.

Dataset / license
-----------------
    Hugging Face : mispeech/speechocean762 (public, no token needed)
                   data/train-00000-of-00001.parquet, data/test-00000-of-00001.parquet
                   (~644 MB; the HF card tags it Apache-2.0)
    OpenSLR      : https://www.openslr.org/101/  (CC BY 4.0)
    Citation     : Zhang J. et al. (2021), "speechocean762: An Open-Source
                   Non-native English Speech Corpus For Pronunciation
                   Assessment", arXiv:2104.01378.

Purpose
-------
1. Download the two parquet files into <project root>/raw/speechocean762/
   (only if missing; free disk space is checked first). The original HF
   train/test split is ignored; we make our own speaker-disjoint split.
2. Keep ADULT speakers only (age >= --min_age, default 18). Child speech
   would let the classifier learn "child voice" instead of accent.
3. Every adult clip goes through cv_utils.preprocess_waveform, the same
   chain as Common Voice (mono, 16 kHz soxr, silence trim, 3-6 s filter on
   the trimmed length, -20 dBFS RMS + 0.99 peak limit), then FLAC in memory.
   Many utterances are short (median raw ~3.5 s), so the 3 s minimum drops
   a large share; that is expected and reported.
4. Selection and split reuse extract_common_voice_subset.select_clips
   (round-robin over speakers, --max_clips_per_speaker cap, up to
   --clips_per_accent clips, all available if fewer) and assign_splits
   (speaker-disjoint, same seed / fractions as the CV script).

Columns
-------
    source="speechocean762", accent_group="Mainland Chinese",
    accent_raw="Mandarin L1 (speechocean762)", is_singaporean=False,
    speaker_id="so762_<speaker>", clip_id="so762_<audio file stem>",
    sentence = the dataset's transcript as provided (UPPERCASE, not re-cased).

Usage
-----
    python prepare_speechocean762.py
    python prepare_speechocean762.py --min_age 18 --clips_per_accent 800
    python prepare_speechocean762.py --no_download --source_dir <folder with the parquet files>

Outputs (written to <project root>/data/)
-----------------------------------------
    data/speechocean_subset/part-XXXX.parquet   -- cv_utils.SUBSET_SCHEMA
    data/speechocean_subset_summary.csv         -- clips / speakers / hours per split
    data/speechocean_subset_split_info.json     -- args, adult filter, speaker lists

Loading
-------
    from cv_utils import load_accent_data
    df = load_accent_data(sources=["speechocean762"], split="train")
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

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


SOURCE = "speechocean762"
SUBSET_NAME = "speechocean_subset"
ACCENT_GROUP = "Mainland Chinese"
ACCENT_RAW = "Mandarin L1 (speechocean762)"
ID_PREFIX = "so762_"
HF_REPO_DEFAULT = "mispeech/speechocean762"
HF_FILES = ["data/train-00000-of-00001.parquet", "data/test-00000-of-00001.parquet"]
READ_COLUMNS = ["speaker", "gender", "age", "text", "audio"]
LICENSE_NOTE = (
    "speechocean762: CC BY 4.0 (OpenSLR 101; HF card tags Apache-2.0). "
    "Cite Zhang et al. 2021, arXiv:2104.01378."
)

LAST_STATS: Dict = {}


def download_instructions(repo: str, source_dir: Path) -> str:
    return (
        "speechocean762 parquet files not found.\n"
        "Either run without --no_download (downloads ~644 MB automatically,\n"
        "no account needed):\n"
        "    python scripts\\prepare_speechocean762.py\n"
        f"or download {', '.join(HF_FILES)} manually from\n"
        f"    https://huggingface.co/datasets/{repo}/tree/main/data\n"
        f"into {source_dir} (or pass --source_dir <folder>).\n"
        "Original corpus: https://www.openslr.org/101/"
    )


def find_parquets(source_dir: Path) -> List[Path]:
    if not source_dir.is_dir():
        return []
    return sorted(p for p in source_dir.rglob("*.parquet")
                  if ".cache" not in p.parts)


def download(repo: str, revision: str, source_dir: Path) -> None:
    from huggingface_hub import HfApi, hf_hub_download

    infos = HfApi().get_paths_info(repo, HF_FILES, repo_type="dataset",
                                   revision=revision)
    total = sum(int(getattr(i, "size", 0) or 0) for i in infos)
    print(f"Downloading {len(HF_FILES)} file(s), {total / 1e6:.0f} MB "
          f"from {repo} -> {source_dir}")
    ensure_free_space(source_dir, total)
    source_dir.mkdir(parents=True, exist_ok=True)
    for name in HF_FILES:
        hf_hub_download(repo, name, repo_type="dataset", revision=revision,
                        local_dir=str(source_dir))
        print(f"  downloaded {name}")


# ============================================================
# PROCESSING
# ============================================================

def process_files(
    files: List[Path],
    min_age: int,
    min_duration: float,
    max_duration: float,
) -> Dict:
    """Decode + preprocess every adult clip. Returns rows and counters."""
    rows: List[dict] = []
    used_ids: set = set()
    all_speakers: set = set()
    adult_speakers: set = set()
    n_total = n_adult = 0
    skips = {"too_short": 0, "too_long": 0, "silent": 0, "decode_error": 0}

    for f in files:
        hf_split = f.name.split("-")[0]
        pf = pq.ParquetFile(str(f))
        row_in_file = 0
        for batch in pf.iter_batches(columns=READ_COLUMNS, batch_size=64):
            for r in batch.to_pylist():
                row_in_file += 1
                n_total += 1
                speaker = str(r["speaker"])
                all_speakers.add(speaker)
                age = r["age"]
                if age is None or int(age) < min_age:
                    continue
                n_adult += 1
                adult_speakers.add(speaker)

                audio = r["audio"] or {}
                stem = Path(str(audio.get("path") or "")).stem
                clip_id = f"{ID_PREFIX}{stem}" if stem else ""
                if not stem or clip_id in used_ids:
                    clip_id = f"{ID_PREFIX}{speaker}_{hf_split}_{row_in_file:04d}"
                assert clip_id not in used_ids, f"duplicate clip_id {clip_id}"
                used_ids.add(clip_id)

                try:
                    x, sr = decode_audio_bytes(audio["bytes"])
                    y, reason = preprocess_waveform(x, sr, min_duration, max_duration)
                except Exception as exc:
                    print(f"  decode error ({ascii_safe(clip_id)}): {ascii_safe(exc)}")
                    y, reason = None, "decode_error"
                if y is None:
                    skips[reason] += 1
                    continue
                rows.append({
                    "clip_id":        clip_id,
                    "source":         SOURCE,
                    "speaker_id":     f"{ID_PREFIX}{speaker}",
                    "accent_raw":     ACCENT_RAW,
                    "accent_group":   ACCENT_GROUP,
                    "is_singaporean": False,
                    "duration_sec":   float(len(y) / TARGET_SR),
                    "sentence":       str(r["text"] or ""),
                    "sampling_rate":  TARGET_SR,
                    "audio_flac":     encode_flac(y),
                })
        print(f"  processed {f.name}: {row_in_file:,} rows")

    return {
        "rows":                 rows,
        "total_clips":          n_total,
        "total_speakers":       len(all_speakers),
        "adult_clips":          n_adult,
        "adult_speakers":       len(adult_speakers),
        "child_clips_excluded": n_total - n_adult,
        "child_speakers_excluded": len(all_speakers - adult_speakers),
        "skips":                skips,
    }


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
        description="Build the Mainland Chinese group from speechocean762 as "
                    "FLAC-in-parquet shards (no audio files written)."
    )
    p.add_argument("--source_dir", type=Path, default=RAW_DIR / "speechocean762",
                   help="Folder holding (or receiving) the speechocean762 parquet "
                        "files (default: <project root>/raw/speechocean762).")
    p.add_argument("--select_new_clips", action="store_true",
                   help="Sample a custom dataset instead of reproducing the fixed 40,000-clip manifest.\n"
                        "Required when changing selection, split or audio settings.")
    p.add_argument("--data_dir", type=Path, default=DATA_DIR,
                   help="Data folder (default: <project root>/data).")
    p.add_argument("--hf_repo", default=HF_REPO_DEFAULT)
    p.add_argument("--hf_revision", default="06385584fad212b26134c656fdd3ccf9f093f33e")
    p.add_argument("--no_download", action="store_true",
                   help="Never download; fail with instructions if files are missing.")
    p.add_argument("--min_age", type=int, default=18,
                   help="Keep speakers with age >= this (default 18 = adults only).")
    p.add_argument("--clips_per_accent", type=int, default=800)
    p.add_argument("--max_clips_per_speaker", type=int, default=15)
    p.add_argument("--min_duration", type=float, default=3.0)
    p.add_argument("--max_duration", type=float, default=6.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val_fraction", type=float, default=0.15)
    p.add_argument("--test_fraction", type=float, default=0.15)
    p.add_argument("--shard_rows", type=int, default=SHARD_ROWS)
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
        if getattr(args, name) <= 0:
            return f"--{name} must be positive"
    if args.min_age < 0:
        return "--min_age must be >= 0"
    return None


# ============================================================
# MAIN
# ============================================================

def main(argv: Optional[List[str]] = None) -> int:
    global LAST_STATS
    args = parse_args(argv)
    if not args.select_new_clips:
        from reference_dataset import prepare_reference
        return prepare_reference("speechocean762", args)
    error = validate_args(args)
    if error:
        print(f"ERROR: {error}")
        return 1
    paths = subset_paths(args.data_dir, SUBSET_NAME)

    print("=" * 70)
    print(f"PREPARE SPEECHOCEAN762 ({ACCENT_GROUP})")
    print("=" * 70)
    print(f"Source dir            : {args.source_dir}")
    print(f"HF repo               : {args.hf_repo} @ {args.hf_revision}")
    print(f"Min age               : {args.min_age}")
    print(f"Clips per accent      : {args.clips_per_accent}")
    print(f"Max clips per speaker : {args.max_clips_per_speaker}")
    print(f"Duration range        : {args.min_duration} - {args.max_duration} s (after trim)")
    print(f"Split (val/test)      : {args.val_fraction} / {args.test_fraction} by speaker")
    print(f"Seed                  : {args.seed}")
    print(f"License               : {LICENSE_NOTE}")

    files = find_parquets(args.source_dir)
    if not files:
        if args.no_download:
            print(f"ERROR: {download_instructions(args.hf_repo, args.source_dir)}")
            return 1
        try:
            download(args.hf_repo, args.hf_revision, args.source_dir)
        except Exception as exc:
            print(f"ERROR: download failed: {ascii_safe(exc)}")
            print(download_instructions(args.hf_repo, args.source_dir))
            return 1
        files = find_parquets(args.source_dir)
        if not files:
            print(f"ERROR: {download_instructions(args.hf_repo, args.source_dir)}")
            return 1

    print()
    print("=" * 70)
    print("DECODE + PREPROCESS (adults only, in memory)")
    print("=" * 70)
    print(f"Files : {[f.name for f in files]}")
    res = process_files(files, args.min_age, args.min_duration, args.max_duration)
    rows = res["rows"]
    kept_speakers = len({r["speaker_id"] for r in rows})
    print()
    print(f"All clips / speakers        : {res['total_clips']:,} / {res['total_speakers']:,}")
    print(f"Excluded (age < {args.min_age}) clips/spk : "
          f"{res['child_clips_excluded']:,} / {res['child_speakers_excluded']:,}")
    print(f"ADULT speakers / clips      : {res['adult_speakers']:,} / {res['adult_clips']:,}")
    print(f"Kept after filters          : {len(rows):,} clips / {kept_speakers:,} speakers")
    for k, v in res["skips"].items():
        print(f"Skipped ({k:<12})     : {v:,}")
    if not rows:
        print("ERROR: no clips survived processing.")
        return 1

    meta = pd.DataFrame([{k: v for k, v in r.items() if k != "audio_flac"} for r in rows])
    selected, plan = select_clips(
        meta, [ACCENT_GROUP], args.clips_per_accent, args.max_clips_per_speaker,
        args.min_duration, args.max_duration, 1.0, args.seed,
    )
    print_plan(plan, args.clips_per_accent)

    print()
    print("=" * 70)
    print("SPEAKER-DISJOINT SPLIT")
    print("=" * 70)
    split_map = assign_splits(selected, ACCENT_GROUP, args.val_fraction,
                              args.test_fraction, args.seed)
    selected = selected.sort_values("selection_rank").copy()
    selected["split"] = selected["speaker_id"].map(split_map)

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
    run_args = {k: v for k, v in vars(args).items() if k not in ("source_dir", "data_dir")}
    info = {
        "source":           SOURCE,
        "dataset":          {"hf_repo": args.hf_repo, "hf_revision": args.hf_revision,
                             "files": [f.name for f in files]},
        "license":          LICENSE_NOTE,
        "args":             run_args,
        "accent_groups":    [ACCENT_GROUP],
        "audio_processing": AUDIO_PROCESSING,
        "adult_filter":     {
            "min_age":              args.min_age,
            "total_speakers":       res["total_speakers"],
            "total_clips":          res["total_clips"],
            "adult_speakers":       res["adult_speakers"],
            "adult_clips":          res["adult_clips"],
            "child_clips_excluded": res["child_clips_excluded"],
            "child_speakers_excluded": res["child_speakers_excluded"],
        },
        "skips":            res["skips"],
        "selection_plan":   plan,
        "sentence_column":  "transcript 'text' as provided (uppercase)",
        "split_strategy":   "speaker-disjoint within each accent group, "
                            "greedy clip-weighted, seeded",
        "groups":           per_group,
    }
    with open(paths["split_info_json"], "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=2)

    print()
    print("=" * 70)
    print("SPEECHOCEAN762 COMPLETE")
    print("=" * 70)
    print(f"  Saved shards    -> {paths['subset_dir']}")
    print(f"  Saved summary   -> {paths['summary_csv']}")
    print(f"  Saved split info-> {paths['split_info_json']}")
    print()
    print("Load in other scripts with:")
    print("  from cv_utils import load_accent_data")
    print("  df = load_accent_data(sources=['speechocean762'], split='train')")

    LAST_STATS = {
        **{k: v for k, v in res.items() if k != "rows"},
        "kept_clips":    len(rows),
        "kept_speakers": kept_speakers,
        "plan":          plan,
        "n_selected":    len(out_rows),
        "n_shards":      n_shards,
    }
    return 0


if __name__ == "__main__":
    sys.exit(main())
