"""
audit_test_speaker_overlap.py

Read-only audit: do the speakers in the fixed MNSC test manifest (ASR-PART1-Test 3000 +
ASR-PART3-Test 1000 rows) also occur in the training / validation data of our two
fine-tuned Whisper models, and what are the test WER/CER on genuinely unseen speakers?

No model inference. Reads only: the test manifest, the cached HF parquet metadata
(other_attributes column only, no audio), the speaker split files, the 15k clip ids and
the existing per-row eval CSVs in results/. Writes ONE new file
(results/test_speaker_overlap.csv by default) and refuses to overwrite an existing file
unless --overwrite is given.

Two notions of "seen" are reported:
  * id      : the test row's primary speaker id (extract_speaker_id_any, the same logic
              preprocess_mnsc_data.py used to build split.train_speakers) is in the
              model's train (or val) speaker list.
  * strict  : ANY voice in the test clip (PART3 clips are two-person conversations, so both
              speaker1 and speaker2) was heard in ANY of the model's training clips (both
              speakers of every training clip). This is the leakage-relevant definition.

Models:
  15k : data/part3_15k/speaker_split_info.json; training clips = data/part3_15k/train shards
        (clip_id 'ASR-PART3-Train/<shard>:<row>' points at the source row).
  4k  : data/speaker_split_info.json (legacy); its 5000 clips are the first 5000 rows of
        ASR-PART3-Train (verified here by speaker multiset), train/val by speaker list.

Usage (from the project root):
    python scripts/audit_test_speaker_overlap.py
    python scripts/audit_test_speaker_overlap.py --out results/test_speaker_overlap_v2.csv
    # Held-out build check (read-only, no output file; exit 0 = all required counts 0):
    python scripts/audit_test_speaker_overlap.py --data_name part3_15k_heldout
    # ... plus the extra test set (build_heldout_extra_test.py):
    python scripts/audit_test_speaker_overlap.py --data_name part3_15k_heldout --extra_test part3_15k_heldout_extra_test
"""

import argparse
import glob
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import json
import pandas as pd
import pyarrow.parquet as pq

# Same scripts/ folder: reuse the exact speaker-id extraction and metric code
from preprocess_mnsc_data import DATASET_ID, HELDOUT_MANIFEST_NAME, extract_speaker_id_any, row_voices
from evaluate_whisper_speaker_split import corpus_metric, n_runaway

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
MANIFEST_PATH = DATA_DIR / "mnsc_extended_evaluation_manifest.csv"
SPLIT_15K = DATA_DIR / "part3_15k" / "speaker_split_info.json"
SPLIT_4K = DATA_DIR / "speaker_split_info.json"
LEGACY_TRAIN = DATA_DIR / "preprocessed_train.parquet"
LEGACY_VAL = DATA_DIR / "preprocessed_val.parquet"
TEST_CONFIGS = ["ASR-PART1-Test", "ASR-PART3-Test"]
TRAIN_CONFIG = "ASR-PART3-Train"
CLIP_ID_RE = re.compile(r"^(?P<cfg>[^/]+)/(?P<shard>\d+):(?P<row>\d+)$")

# Per-row test eval CSVs (no _val) -> model; the plain 4k file predates the eval_ naming.
LEGACY_4K_EVAL = "whisper_small_mnsc_extended_evaluation.csv"
LEGACY_4K_SUMMARY = "whisper_small_mnsc_extended_summary.csv"


# ============================================================
# SOURCE METADATA (no audio)
# ============================================================

def snapshot_dir(arg: Optional[str]) -> Path:
    """Local snapshot of the HF dataset; never downloads."""
    if arg:
        return Path(arg)
    from huggingface_hub import snapshot_download
    try:
        return Path(snapshot_download(repo_id=DATASET_ID, repo_type="dataset",
                                      allow_patterns=[f"{c}/*" for c in TEST_CONFIGS],
                                      local_files_only=True))
    except Exception as exc:
        sys.exit(f"ERROR: MNSC test parquet files not in the local HF cache ({exc}). "
                 "Run evaluate_whisper_speaker_split.py once, or pass --snapshot_dir.")


def read_attrs(paths: List[Path]) -> List[dict]:
    """other_attributes of every row, in global order across the sorted shards."""
    out: List[dict] = []
    for p in paths:
        out.extend(pq.read_table(p, columns=["other_attributes"]).column(0).to_pylist())
    return out


def voices(attrs: dict) -> Tuple[str, str, str]:
    """(speaker1 id, speaker2 id, conversation id); '' when absent (PART1)."""
    s1 = ((attrs.get("speaker1") or {}).get("speaker_id") or "").strip()
    s2 = ((attrs.get("speaker2") or {}).get("speaker_id") or "").strip()
    return s1, s2, str(attrs.get("conversation_id") or "").strip()


class TrainSource:
    """Lazy per-shard access to ASR-PART3-Train other_attributes."""

    def __init__(self, snap: Path):
        self.paths = sorted((snap / TRAIN_CONFIG).glob("*.parquet"))
        self.cache: Dict[int, List[dict]] = {}

    def row(self, shard: int, row: int) -> dict:
        if shard >= len(self.paths):
            raise FileNotFoundError(
                f"{TRAIN_CONFIG} shard {shard} not in the local HF cache "
                f"({len(self.paths)} shards cached)")
        if shard not in self.cache:
            self.cache[shard] = read_attrs([self.paths[shard]])
        return self.cache[shard][row]

    def first_rows(self, n: int) -> List[dict]:
        out: List[dict] = []
        shard = 0
        while len(out) < n:
            if shard >= len(self.paths):
                raise FileNotFoundError(f"need {n} {TRAIN_CONFIG} rows, cache has {len(out)}")
            if shard not in self.cache:
                self.cache[shard] = read_attrs([self.paths[shard]])
            out.extend(self.cache[shard])
            shard += 1
        return out[:n]


# ============================================================
# MODEL TRAINING / VALIDATION SETS
# ============================================================

def model_sets(split_file: Path, clips: List[Tuple[str, dict]]) -> dict:
    """clips: (split 'train'/'val', source other_attributes). Returns id lists + heard voices."""
    split = json.loads(split_file.read_text(encoding="utf-8"))["split"]
    res = {"train_ids": set(split["train_speakers"]), "val_ids": set(split["val_speakers"])}
    for part in ("train", "val"):
        v: Set[str] = set()
        c: Set[str] = set()
        n = 0
        for p, attrs in clips:
            if p != part:
                continue
            s1, s2, conv = voices(attrs)
            v.update(x for x in (s1, s2) if x)
            c.add(conv)
            n += 1
        res[f"{part}_voices"], res[f"{part}_convs"], res[f"{part}_clips"] = v, c, n
    res["expected"] = (split.get("train_samples"), split.get("val_samples"))
    return res


def clips_15k(src: TrainSource) -> List[Tuple[str, dict]]:
    return clips_for(DATA_DIR / "part3_15k", src)


def clips_for(data_dir: Path, src: TrainSource) -> List[Tuple[str, dict]]:
    """(split, source other_attributes) of every clip in data_dir/{train,val}."""
    out = []
    for part in ("train", "val"):
        for shard in sorted((data_dir / part).glob("*.parquet")):
            t = pq.read_table(shard, columns=["clip_id", "speaker_id"]).to_pylist()
            for r in t:
                m = CLIP_ID_RE.match(r["clip_id"])
                if not m or m["cfg"] != TRAIN_CONFIG:
                    raise ValueError(f"unexpected clip_id {r['clip_id']}")
                attrs = src.row(int(m["shard"]), int(m["row"]))
                if extract_speaker_id_any({"other_attributes": attrs}) != r["speaker_id"]:
                    raise ValueError(f"clip {r['clip_id']}: speaker mismatch with source row")
                out.append((part, attrs))
    return out


def clips_4k(src: TrainSource) -> List[Tuple[str, dict]]:
    split = json.loads(SPLIT_4K.read_text(encoding="utf-8"))["split"]
    train_ids, val_ids = set(split["train_speakers"]), set(split["val_speakers"])
    legacy_spk = (pq.read_table(LEGACY_TRAIN, columns=["speaker_id"]).column(0).to_pylist()
                  + pq.read_table(LEGACY_VAL, columns=["speaker_id"]).column(0).to_pylist())
    rows = src.first_rows(len(legacy_spk))
    ids = [extract_speaker_id_any({"other_attributes": a}) for a in rows]
    if Counter(ids) != Counter(legacy_spk):
        raise ValueError("legacy 4k files are not the first N ASR-PART3-Train rows "
                         "(speaker multiset differs); cannot reconstruct its clips")
    out = []
    for spk, attrs in zip(ids, rows):
        out.append(("train" if spk in train_ids else "val" if spk in val_ids else "?", attrs))
    return out


# ============================================================
# HELD-OUT DATASET AUDIT (--data_name)
# ============================================================

def all_voices(attrs: dict) -> Set[str]:
    """Every voice of a row (speaker1, speaker2, speaker); same rule as preprocessing."""
    return set(row_voices(attrs))


def audit_dataset(data_dir: Path, snap: Path) -> dict:
    """
    Check a held-out build (data_dir with heldout_test_manifest.json) against the test
    metadata in `snap`. Returns {"required": {name: count}, "consistency": {name: bool},
    "info": {name: value}}; the build is clean iff every required count is 0 and every
    consistency check is True.
    """
    man_path = data_dir / HELDOUT_MANIFEST_NAME
    if not man_path.exists():
        raise FileNotFoundError(f"no {HELDOUT_MANIFEST_NAME} in {data_dir}: not a held-out build "
                                "(preprocess_mnsc_data.py --heldout_test_conversations N)")
    man = json.loads(man_path.read_text(encoding="utf-8"))
    split = json.loads((data_dir / "speaker_split_info.json").read_text(encoding="utf-8"))["split"]
    held_convs, held_voices = set(man["conversations"]), set(man["voices"])

    p3 = read_attrs(sorted((snap / "ASR-PART3-Test").glob("*.parquet")))
    p1 = read_attrs(sorted((snap / "ASR-PART1-Test").glob("*.parquet")))
    p1_voices: Set[str] = set()
    for a in p1:
        p1_voices |= all_voices(a)

    clips = clips_for(data_dir, TrainSource(snap))
    sets = {part: {"voices": set(), "convs": set(), "n": 0} for part in ("train", "val")}
    for part, attrs in clips:
        s = sets[part]
        s["voices"] |= all_voices(attrs)
        conv = voices(attrs)[2]
        if conv:
            s["convs"].add(conv)
        s["n"] += 1
    tv_voices = sets["train"]["voices"] | sets["val"]["voices"]
    tv_convs = sets["train"]["convs"] | sets["val"]["convs"]

    rows = sorted(man["part3_test_row_indices"])
    recomputed = [i for i, a in enumerate(p3) if voices(a)[2] in held_convs]
    held_row_voices: Set[str] = set()
    for i in rows:
        held_row_voices |= all_voices(p3[i])
    p3_convs = {voices(a)[2] for a in p3} - {""}

    required = {}
    for part in ("train", "val"):
        s = sets[part]
        required[f"{part}_conversations_in_heldout"] = len(s["convs"] & held_convs)
        required[f"{part}_voices_in_heldout"] = len(s["voices"] & held_voices)
        required[f"{part}_voices_in_heldout_test_rows"] = len(s["voices"] & held_row_voices)
        required[f"{part}_voices_in_part1_test"] = len(s["voices"] & p1_voices)
    required["train_val_shared_voices"] = len(sets["train"]["voices"] & sets["val"]["voices"])
    required["train_val_shared_conversations"] = len(sets["train"]["convs"] & sets["val"]["convs"])
    required["heldout_test_rows_with_seen_voice"] = sum(
        1 for i in rows if all_voices(p3[i]) & tv_voices)
    consistency = {
        "manifest_rows_match_conversations": rows == recomputed,
        "manifest_rows_count": len(rows) == int(man["n_part3_test_rows"]),
        "heldout_row_voices_in_manifest": held_row_voices <= held_voices,
        "train_clips_match_split": sets["train"]["n"] == split["train_samples"],
        "val_clips_match_split": sets["val"]["n"] == split["val_samples"],
    }
    info = {
        "heldout_conversations": len(held_convs),
        "heldout_voices": len(held_voices),
        "heldout_part3_test_rows": len(rows),
        "part1_test_voices": len(p1_voices),
        "train_clips": sets["train"]["n"], "val_clips": sets["val"]["n"],
        "train_voices": len(sets["train"]["voices"]), "val_voices": len(sets["val"]["voices"]),
        "train_conversations": len(sets["train"]["convs"]),
        "val_conversations": len(sets["val"]["convs"]),
        "non_heldout_part3_test_conversations_in_train_val": len((p3_convs - held_convs) & tv_convs),
        "non_heldout_part3_test_rows_with_seen_voice": sum(
            1 for i, a in enumerate(p3) if voices(a)[2] not in held_convs and all_voices(a) & tv_voices),
    }
    return {"required": required, "consistency": consistency, "info": info}


def _clip_ids(data_dir: Path, parts=("train", "val")) -> Set[str]:
    out: Set[str] = set()
    for part in parts:
        for shard in sorted((data_dir / part).glob("*.parquet")):
            out.update(pq.read_table(shard, columns=["clip_id"]).column(0).to_pylist())
    return out


def _parse_clip_id(clip_id: str) -> Tuple[int, int]:
    m = CLIP_ID_RE.match(clip_id)
    if not m or m["cfg"] != TRAIN_CONFIG:
        raise ValueError(f"unexpected clip_id {clip_id}")
    return int(m["shard"]), int(m["row"])


def audit_extra_test(extra_dir: Path, data_dir: Path, snap: Path) -> dict:
    """
    Check an extra test set (build_heldout_extra_test.py) against its held-out build
    data_dir and the source metadata in `snap`. Same result shape as audit_dataset.
    """
    from build_heldout_extra_test import EXTRA_INFO_NAME, EXTRA_SPLIT_DIR, segment_key
    from preprocess_mnsc_data import DEFAULT_DATA_NAME
    info = json.loads((extra_dir / EXTRA_INFO_NAME).read_text(encoding="utf-8"))
    man = json.loads((data_dir / HELDOUT_MANIFEST_NAME).read_text(encoding="utf-8"))
    held_convs, held_voices = set(man["conversations"]), set(man["voices"])

    ids: List[str] = []
    for name in info["layout"]["shards"]:
        ids += pq.read_table(extra_dir / EXTRA_SPLIT_DIR / name, columns=["clip_id"]).column(0).to_pylist()
    src = TrainSource(snap)
    attrs = [src.row(*_parse_clip_id(c)) for c in ids]
    ex_convs = {voices(a)[2] for a in attrs} - {""}
    ex_voices: Set[str] = set()
    for a in attrs:
        ex_voices |= all_voices(a)

    tv_voices: Set[str] = set()
    tv_convs: Set[str] = set()
    for _, a in clips_for(data_dir, src):
        tv_voices |= all_voices(a)
        if voices(a)[2]:
            tv_convs.add(voices(a)[2])
    tv_ids = _clip_ids(data_dir)

    p1_voices: Set[str] = set()
    for a in read_attrs(sorted((snap / "ASR-PART1-Test").glob("*.parquet"))):
        p1_voices |= all_voices(a)
    p3 = read_attrs(sorted((snap / "ASR-PART3-Test").glob("*.parquet")))
    test_segs = {segment_key(p3[i]) for i in man["part3_test_row_indices"]} - {None}

    clips = info["clips"]
    dup = info["duplicates"]
    required = {
        "extra_conversations_not_in_manifest": len(ex_convs - held_convs),
        "extra_voices_not_in_manifest": len(ex_voices - held_voices),
        "extra_conversations_in_train_val": len(ex_convs & tv_convs),
        "extra_voices_in_train_val": len(ex_voices & tv_voices),
        "extra_clip_ids_in_train_val": len(set(ids) & tv_ids),
        "extra_voices_in_part1_test": len(ex_voices & p1_voices),
        "extra_segments_equal_heldout_test_rows": sum(1 for a in attrs if segment_key(a) in test_segs),
        "extra_duplicate_clip_ids": len(ids) - len(set(ids)),
    }
    consistency = {
        "n_clips_matches_info": len(ids) == int(info["n_clips"]) == len(clips),
        "clip_order_matches_info": ids == [c["clip_id"] for c in clips],
        "info_conversations_match_source": all(
            c["conversation_id"] == voices(a)[2] for c, a in zip(clips, attrs)),
        "seen_equals_build_exclusion": (info["n_heldout_conversation_rows_seen"]
                                        == info.get("expected_excluded_by_build")),
        "accounted": (int(info["n_clips"]) + int(dup["n_dropped"]) + int(info["skipped_audio"])
                      + int(info["skipped_text"]) == int(info["n_heldout_conversation_rows_seen"])),
    }
    out_info = {
        "n_clips": info["n_clips"], "n_conversations": len(ex_convs), "n_voices": len(ex_voices),
        "total_hours": info["total_hours"], "duplicates_dropped": dup["n_dropped"],
        "near_duplicate_clips": info["n_near_duplicate_clips"],
    }
    # Seen by the OLD models (they trained on many of these exact clips): info only.
    old = data_dir.parent / DEFAULT_DATA_NAME
    for part in ("train", "val"):
        key = f"extra_clips_in_{DEFAULT_DATA_NAME}_{part}"
        out_info[key] = (len(set(ids) & _clip_ids(old, (part,))) if (old / part).exists() else "n/a")
    if LEGACY_TRAIN.exists() and LEGACY_VAL.exists():
        n_legacy = (pq.ParquetFile(LEGACY_TRAIN).metadata.num_rows
                    + pq.ParquetFile(LEGACY_VAL).metadata.num_rows)
        offsets = [0]
        for p in src.paths:
            offsets.append(offsets[-1] + pq.ParquetFile(p).metadata.num_rows)
        out_info["extra_clips_in_legacy_4k"] = sum(
            1 for c in ids if offsets[_parse_clip_id(c)[0]] + _parse_clip_id(c)[1] < n_legacy)
    else:
        out_info["extra_clips_in_legacy_4k"] = "n/a"
    return {"required": required, "consistency": consistency, "info": out_info}


def _print_audit(res: dict) -> bool:
    print("\nREQUIRED ZERO")
    for k, v in res["required"].items():
        print(f"  {k:<44} {v:>6}  {'OK' if v == 0 else 'FAIL'}")
    print("\nCONSISTENCY")
    for k, v in res["consistency"].items():
        print(f"  {k:<44} {'OK' if v else 'FAIL'}")
    print("\nINFO")
    for k, v in res["info"].items():
        print(f"  {k:<52} {v}")
    return all(v == 0 for v in res["required"].values()) and all(res["consistency"].values())


def audit_dataset_main(args) -> int:
    """--data_name mode: print the audit table, write nothing; exit 0 only if all clean."""
    from preprocess_mnsc_data import dataset_dir
    data_dir = dataset_dir(args.data_name, Path(args.data_root))
    if not (data_dir / HELDOUT_MANIFEST_NAME).exists():
        print(f"ERROR: no {HELDOUT_MANIFEST_NAME} in {data_dir}; --data_name audits held-out builds "
              "only (preprocess_mnsc_data.py --heldout_test_conversations N).")
        return 1
    snap = snapshot_dir(args.snapshot_dir)
    print("HF snapshot:", snap)
    print("Dataset    :", data_dir)
    res = audit_dataset(data_dir, snap)
    if getattr(args, "extra_test", None):
        clean = _print_audit(res)
        extra_dir = dataset_dir(args.extra_test, Path(args.data_root))
        print("\n" + "=" * 70)
        print(f"EXTRA TEST SET {args.extra_test}  ({extra_dir})")
        print("=" * 70)
        clean = _print_audit(audit_extra_test(extra_dir, data_dir, snap)) and clean
        print("\nRESULT:", "PASS (all required counts 0)" if clean else "FAIL")
        return 0 if clean else 1
    print("\nREQUIRED ZERO")
    for k, v in res["required"].items():
        print(f"  {k:<44} {v:>6}  {'OK' if v == 0 else 'FAIL'}")
    print("\nCONSISTENCY")
    for k, v in res["consistency"].items():
        print(f"  {k:<44} {'OK' if v else 'FAIL'}")
    print("\nINFO")
    for k, v in res["info"].items():
        print(f"  {k:<52} {v}")
    clean = all(v == 0 for v in res["required"].values()) and all(res["consistency"].values())
    print("\nRESULT:", "PASS (all required counts 0)" if clean else "FAIL")
    return 0 if clean else 1


# ============================================================
# METRICS
# ============================================================

def eval_files() -> List[Tuple[str, str, Path]]:
    """(model, run label, path) of every per-row TEST eval CSV in results/."""
    out = []
    p = RESULTS_DIR / LEGACY_4K_EVAL
    if p.exists():
        out.append(("4k", "4k plain (" + p.name + ")", p))
    for p in sorted(RESULTS_DIR.glob("eval_*.csv")):
        if "_val" in p.stem or "_rows" in p.stem:
            continue
        if "_ho-" in p.stem or "_hx-" in p.stem or "-heldout" in p.stem:
            continue  # held-out subsets / extra set / held-out model: audited with --data_name
        m = re.search(r"mnsc-sg-(15k|4k)(.*)$", p.stem)
        if not m:
            continue
        out.append((m[1], f"{m[1]} {m[2].lstrip('_') or 'plain'}", p))
    order = {"15k": 0, "4k": 1}
    return sorted(out, key=lambda t: (order[t[0]], t[1]))


def summary_for(path: Path) -> Optional[Path]:
    if path.name == LEGACY_4K_EVAL:
        s = RESULTS_DIR / LEGACY_4K_SUMMARY
    else:
        s = RESULTS_DIR / path.name.replace("eval_", "summary_", 1)
    return s if s.exists() else None


def metrics(df: pd.DataFrame) -> dict:
    if len(df) == 0:
        return {"rows": 0, "corpus_WER": float("nan"), "corpus_CER": float("nan"), "loops": 0}
    return {"rows": len(df), "corpus_WER": corpus_metric(df, "WER"),
            "corpus_CER": corpus_metric(df, "CER"), "loops": n_runaway(df)}


# ============================================================
# MAIN
# ============================================================

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[1].strip())
    p.add_argument("--snapshot_dir", default=None,
                   help="Local snapshot of the HF dataset (default: resolve from the HF cache, never downloads).")
    p.add_argument("--out", default=str(RESULTS_DIR / "test_speaker_overlap.csv"),
                   help="Per-row output CSV (default results/test_speaker_overlap.csv).")
    p.add_argument("--overwrite", action="store_true", help="Allow replacing --out if it exists.")
    p.add_argument("--data_name", default=None,
                   help="Audit a held-out build data/<DATA_NAME>/ instead (read-only, writes nothing): "
                        "0 shared conversations/voices with its held-out PART3-Test rows and PART1-Test, "
                        "0 voices shared by train and val. Exit 0 only if clean.")
    p.add_argument("--data_root", default=str(DATA_DIR),
                   help="With --data_name: parent folder of the dataset (default <project>/data).")
    p.add_argument("--extra_test", default=None, metavar="EXTRA_NAME",
                   help="With --data_name: also audit data/<EXTRA_NAME>/ (build_heldout_extra_test.py): "
                        "all its conversations/voices in the manifest, none in train/val or PART1-Test, "
                        "no clip equal to a held-out PART3-Test row.")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.extra_test and not args.data_name:
        print("ERROR: --extra_test requires --data_name (the held-out build it was made from).")
        return 2
    if args.data_name:
        return audit_dataset_main(args)
    out_path = Path(args.out)
    if out_path.exists() and not args.overwrite:
        print(f"ERROR: {out_path} exists; pass --overwrite or a different --out.")
        return 2
    pd.set_option("display.width", 200)
    snap = snapshot_dir(args.snapshot_dir)
    print("HF snapshot:", snap)

    # ---------- test rows ----------
    manifest = pd.read_csv(MANIFEST_PATH)
    assert len(manifest) == 4000 and manifest["sample_id"].is_unique
    test_attrs = {c: read_attrs(sorted((snap / c).glob("*.parquet"))) for c in TEST_CONFIGS}

    # ---------- model sets ----------
    src = TrainSource(snap)
    m15 = model_sets(SPLIT_15K, clips_15k(src))
    m4 = model_sets(SPLIT_4K, clips_4k(src))
    models = {"15k": m15, "4k": m4}
    print("\nTRAINING / VALIDATION SETS (reconstructed from source rows)")
    for name, m in models.items():
        print(f"  {name}: train {len(m['train_ids'])} speaker ids / {m['train_clips']} clips "
              f"(expected {m['expected'][0]}), {len(m['train_voices'])} voices, {len(m['train_convs'])} conversations; "
              f"val {len(m['val_ids'])} ids / {m['val_clips']} clips (expected {m['expected'][1]}), "
              f"{len(m['val_voices'])} voices, {len(m['val_convs'])} conversations")
        leak_v = m["val_voices"] & m["train_voices"]
        print(f"       own val voices also heard in own train: {len(leak_v)} of {len(m['val_voices'])}; "
              f"val conversations also in train: {len(m['val_convs'] & m['train_convs'])} of {len(m['val_convs'])}")

    rows = []
    for _, r in manifest.iterrows():
        attrs = test_attrs[r["config"]][int(r["row_index"])]
        spk = extract_speaker_id_any({"other_attributes": attrs})
        s1, s2, conv = voices(attrs)
        vs = {x for x in (s1, s2) if x} or ({spk} if spk != "unknown" else set())
        rec = {"config": r["config"], "row_index": int(r["row_index"]), "sample_id": r["sample_id"],
               "speaker_id": spk, "speaker2_id": s2, "conversation_id": conv}
        for name, m in models.items():
            rec[f"seen_by_{name}_train"] = spk in m["train_ids"]
        for name, m in models.items():
            rec[f"seen_by_{name}_val"] = spk in m["val_ids"]
        for name, m in models.items():
            rec[f"voice_in_{name}_train"] = bool(vs & m["train_voices"])
            rec[f"voice_in_{name}_val"] = bool(vs & m["val_voices"])
            rec[f"conv_in_{name}_train"] = bool(conv) and conv in m["train_convs"]
        rows.append(rec)
    ov = pd.DataFrame(rows)
    ov["seen_strict_any"] = ov["voice_in_15k_train"] | ov["voice_in_4k_train"]

    # ---------- Q1: id availability ----------
    print("\nQ1. SPEAKER IDS IN THE EVALUATED TEST ROWS")
    for c in TEST_CONFIGS:
        g = ov[ov["config"] == c]
        ids = set(g["speaker_id"]) - {"unknown"}
        all_v = set(g["speaker_id"]) | set(g["speaker2_id"])
        all_v -= {"", "unknown"}
        print(f"  {c}: {len(g)} rows, missing primary id {int((g['speaker_id'] == 'unknown').sum())}, "
              f"{len(ids)} unique primary ids, {len(all_v)} unique voices (speaker1+speaker2), "
              f"{g['conversation_id'].replace('', pd.NA).nunique()} conversations, "
              f"rows with speaker2 {int((g['speaker2_id'] != '').sum())}")

    # ---------- Q2/Q3: overlap ----------
    print("\nQ2/Q3. OVERLAP WITH EACH MODEL'S SPEAKERS (speakers / evaluated rows)")
    tab = []
    for c in TEST_CONFIGS:
        g = ov[ov["config"] == c]
        test_ids = set(g["speaker_id"]) - {"unknown"}
        test_voices = (set(g["speaker_id"]) | set(g["speaker2_id"])) - {"", "unknown"}
        for name, m in models.items():
            for part in ("train", "val"):
                tab.append({
                    "test_config": c, "model": name, "list": part,
                    "ids_overlap": len(test_ids & m[f"{part}_ids"]),
                    "rows_id_seen": int(g[f"seen_by_{name}_{part}"].sum()),
                    "voices_overlap": len(test_voices & m[f"{part}_voices"]),
                    "rows_voice_heard": int(g[f"voice_in_{name}_{part}"].sum()),
                    "of_rows": len(g),
                })
        print(f"  {c}: rows unseen by BOTH models (strict) = "
              f"{int((~g['seen_strict_any']).sum())} of {len(g)}")
    print(pd.DataFrame(tab).to_string(index=False))

    # id namespaces across parts (string format)
    p1 = set(ov.loc[ov["config"] == "ASR-PART1-Test", "speaker_id"])
    print(f"\n  PART1-Test ids also in any PART3 train/val list (string match): "
          f"{len(p1 & (m15['train_ids'] | m15['val_ids'] | m4['train_ids'] | m4['val_ids']))}")

    ov.to_csv(out_path, index=False)
    print(f"\nWrote {out_path} ({len(ov)} rows)")

    # ---------- Q4: metrics ----------
    print("\nQ4. CORPUS WER / CER BY GROUP (existing per-row eval CSVs; no inference)")
    files = eval_files()
    res = []
    for model, label, path in files:
        df = pd.read_csv(path, keep_default_na=False)
        # newer eval CSVs carry their own cluster columns; `ov` provides them here
        df = df.drop(columns=[c for c in ("speaker_id", "conversation_id") if c in df.columns])
        df["reference"] = df["reference"].astype(str)
        df["prediction"] = df["prediction"].astype(str)
        df = df.merge(ov, on=["sample_id", "config", "row_index"], how="left", validate="1:1")
        assert df["speaker_id"].notna().all() and len(df) == 4000, f"{path.name}: rows do not match manifest"

        # sanity check vs the summary CSV
        s = summary_for(path)
        if s is not None:
            sm = pd.read_csv(s).set_index("group")
            for grp, sub in [("Overall", df)] + [(c, df[df["config"] == c]) for c in TEST_CONFIGS]:
                w = corpus_metric(sub, "WER")
                ok = abs(w - float(sm.loc[grp, "corpus_WER"])) < 1e-9 and int(sm.loc[grp, "samples"]) == len(sub)
                if not ok:
                    print(f"  SANITY MISMATCH {path.name} {grp}: {w} vs {sm.loc[grp, 'corpus_WER']}")
            print(f"  sanity {path.name} vs {s.name}: checked")
        else:
            print(f"  sanity {path.name}: no summary file")

        part3 = df[df["config"] == "ASR-PART3-Test"]
        part1 = df[df["config"] == "ASR-PART1-Test"]
        groups = [
            ("ALL 4000", df),
            ("PART1 all", part1),
            ("PART3 all", part3),
            ("PART3 id-seen by this model", part3[part3[f"seen_by_{model}_train"]]),
            ("PART3 id-unseen by this model", part3[~part3[f"seen_by_{model}_train"]]),
            ("PART3 voice-heard by this model", part3[part3[f"voice_in_{model}_train"]]),
            ("PART3 voice-unseen by this model", part3[~part3[f"voice_in_{model}_train"]]),
            ("PART3 unseen by BOTH (strict)", part3[~part3["seen_strict_any"]]),
            ("ALL unseen by BOTH (strict)", df[~df["seen_strict_any"]]),
        ]
        for gname, sub in groups:
            res.append({"run": label, "group": gname, **metrics(sub)})
    rt = pd.DataFrame(res)
    for col in ("corpus_WER", "corpus_CER"):
        rt[col] = rt[col].map(lambda x: f"{x:.4f}")
    for label in pd.unique(rt["run"]):
        print(f"\n  {label}")
        print(rt[rt["run"] == label].drop(columns="run").to_string(index=False))

    # ---------- Q5 helper: side-by-side on the fair subset ----------
    print("\nQ5. SIDE BY SIDE (PART3-Test)")
    piv = pd.DataFrame(res)
    piv = piv[piv["group"].isin(["PART3 all", "PART3 unseen by BOTH (strict)", "PART3 id-unseen by this model",
                                 "PART3 voice-unseen by this model"])]
    print(piv.pivot(index="run", columns="group", values="corpus_WER").round(4).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
