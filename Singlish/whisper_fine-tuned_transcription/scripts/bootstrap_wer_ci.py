"""
bootstrap_wer_ci.py

Cluster bootstrap confidence intervals for corpus WER / CER from per-row eval CSVs
(evaluate_whisper_speaker_split.py). Read-only on its inputs; writes ONE CSV to results/.

Rows of the same conversation (PART3) or speaker (PART1) are correlated, so whole
CLUSTERS are resampled with replacement, not single rows:
  ASR-PART1-Test                            -> speaker      (spk:<speaker_id>)
  ASR-PART3-Test, PART3-heldout-train-clips -> conversation (conv:<conversation_id>)
  *-val                                     -> conversation if present, else speaker
  Overall                                   -> stratified: each group's clusters resampled
                                               within the group, sums added
Corpus WER of a resample = sum(word errors) / sum(reference words) over its rows (CER:
characters); errors = S + D + I from jiwer, on the CSV's reference/prediction columns, which
already hold normalize_for_evaluation() output (exactly what the eval summary scores).
Percentile CIs. --paired compares two runs on identical rows with the SAME cluster draws.

Old CSVs without conversation_id / speaker_id columns get them joined by (config, row_index)
from the local HF snapshot of ASR-PART1-Test / ASR-PART3-Test (never downloads).
Val CSVs have an empty conversation_id, so val rows cluster by speaker. --val_conversations
(opt-in) joins the conversation of each val clip from the local ASR-PART3-Train snapshot via
its sample_id (clip id 'ASR-PART3-Train/<shard>:<row>'), in memory; the CSVs are not changed.

Usage (from the project root):
    python scripts/bootstrap_wer_ci.py results/eval_whisper-small-mnsc-sg-15k_nrng5.csv
    python scripts/bootstrap_wer_ci.py --paired results/eval_whisper-small-mnsc-sg-15k.csv results/eval_whisper-small-mnsc-sg-15k_nrng5.csv
    python scripts/bootstrap_wer_ci.py --paired --val_conversations results/eval_A_val_fb.csv results/eval_B_val_fb.csv
Output: results/bootstrap_<stem>_n<N>_s<seed>.csv  or
        results/bootstrap_paired_<stemA>__vs__<stemB>_n<N>_s<seed>.csv
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import jiwer
import numpy as np
import pandas as pd

from build_heldout_extra_test import EXTRA_TEST_CONFIG
from evaluate_whisper_speaker_split import corpus_metric
from preprocess_mnsc_data import DATASET_ID, extract_speaker_id_any, read_attrs_column, row_conversation

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RESULTS_DIR = PROJECT_ROOT / "results"

DEFAULT_N_BOOT = 1000
DEFAULT_CONFIDENCE = 0.95
DEFAULT_SEED = 42
PART1 = "ASR-PART1-Test"
PART3 = "ASR-PART3-Test"
VAL_SOURCE_CONFIG = "ASR-PART3-Train"  # val clips of part3_* builds come from here
CLIP_ID_RE = re.compile(r"^ASR-PART3-Train/(\d{5}):(\d+)$")
REQUIRED_COLUMNS = ["config", "row_index", "reference", "prediction"]
KEY = ["config", "row_index"]
COUNTS = ["w_err", "w_ref", "c_err", "c_ref"]
METRICS = {"WER": ("w_err", "w_ref"), "CER": ("c_err", "c_ref")}


# ============================================================
# INPUT
# ============================================================

def load_eval_csv(path) -> pd.DataFrame:
    """Per-row eval CSV; empty predictions stay '' (as compare_decoding_runs.load_run)."""
    df = pd.read_csv(path, keep_default_na=False)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    df["reference"] = df["reference"].astype(str)
    df["prediction"] = df["prediction"].astype(str)
    df["row_index"] = df["row_index"].astype(int)
    for c in ("conversation_id", "speaker_id"):
        if c in df.columns:
            df[c] = df[c].astype(str)
    return df


def row_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Adds w_err, w_ref, c_err, c_ref per row (jiwer defaults, no extra normalization)."""
    out = df.copy()
    w_err, w_ref, c_err, c_ref = [], [], [], []
    for ref, hyp in zip(out["reference"], out["prediction"]):
        w = jiwer.process_words(ref, hyp)
        c = jiwer.process_characters(ref, hyp)
        w_err.append(w.substitutions + w.deletions + w.insertions)
        w_ref.append(w.substitutions + w.deletions + w.hits)
        c_err.append(c.substitutions + c.deletions + c.insertions)
        c_ref.append(c.substitutions + c.deletions + c.hits)
    out["w_err"], out["w_ref"], out["c_err"], out["c_ref"] = w_err, w_ref, c_err, c_ref
    return out


def resolve_snapshot(arg: Optional[str]) -> Path:
    """Local snapshot of the HF test configs; never downloads."""
    if arg:
        return Path(arg)
    from huggingface_hub import snapshot_download
    try:
        return Path(snapshot_download(repo_id=DATASET_ID, repo_type="dataset",
                                      allow_patterns=[f"{PART1}/*", f"{PART3}/*"],
                                      local_files_only=True))
    except Exception as exc:
        raise ValueError(f"MNSC test parquet files not in the local HF cache ({exc}); "
                         "pass --snapshot_dir") from exc


def test_cluster_lookup(snap: Path) -> Dict[Tuple[str, int], Tuple[str, str]]:
    out = {}
    for config in (PART1, PART3):
        paths = [str(p) for p in sorted((Path(snap) / config).glob("*.parquet"))]
        for i, attrs in enumerate(read_attrs_column(paths, local=True)):
            out[(config, i)] = (row_conversation(attrs),
                                extract_speaker_id_any({"other_attributes": attrs}))
    return out


def val_conversation_lookup(sample_ids, snap: Path) -> Dict[str, Tuple[str, str]]:
    """clip_id -> (conversation_id, speaker_id) for val clip ids 'ASR-PART3-Train/<shard>:<row>'
    (preprocess_mnsc_data clip ids: shard = index of train-<shard>-of-*.parquet, row = row in
    that shard). Reads only the other_attributes column of the needed local shards."""
    wanted: Dict[int, List[Tuple[str, int]]] = {}
    for sid in pd.unique(pd.Series(list(sample_ids), dtype=str)):
        m = CLIP_ID_RE.match(sid)
        if not m:
            raise ValueError(f"val sample_id {sid!r} is not an {VAL_SOURCE_CONFIG} clip id")
        wanted.setdefault(int(m.group(1)), []).append((sid, int(m.group(2))))
    shards = sorted((Path(snap) / VAL_SOURCE_CONFIG).glob("*.parquet"))
    out: Dict[str, Tuple[str, str]] = {}
    for shard_idx, items in sorted(wanted.items()):
        prefix = f"train-{shard_idx:05d}-of-"
        match = [p for p in shards if p.name.startswith(prefix)]
        if len(match) != 1:
            raise ValueError(f"{VAL_SOURCE_CONFIG} shard {prefix}* not found in {snap}")
        attrs = read_attrs_column([str(match[0])], local=True)
        for sid, row in items:
            if row >= len(attrs):
                raise ValueError(f"{sid}: row {row} beyond {match[0].name} ({len(attrs)} rows)")
            a = attrs[row]
            out[sid] = (row_conversation(a), extract_speaker_id_any({"other_attributes": a}))
    return out


def val_shard_indices(sample_ids) -> List[int]:
    """Sorted source shard indices referenced by val clip ids (unparsable ids are skipped
    here and rejected by val_conversation_lookup)."""
    return sorted({int(m.group(1)) for m in map(CLIP_ID_RE.match, map(str, sample_ids)) if m})


def resolve_val_snapshot(arg: Optional[str], shard_indices: List[int]) -> Path:
    """Local snapshot holding the needed ASR-PART3-Train shards (val clips); never downloads.
    Only the shards in shard_indices must be cached (the cache usually holds a subset)."""
    if arg:
        return Path(arg)
    from huggingface_hub import snapshot_download
    patterns = [f"{VAL_SOURCE_CONFIG}/train-{i:05d}-of-*.parquet" for i in shard_indices]
    try:
        return Path(snapshot_download(repo_id=DATASET_ID, repo_type="dataset",
                                      allow_patterns=patterns or [f"{VAL_SOURCE_CONFIG}/*"],
                                      local_files_only=True))
    except Exception as exc:
        raise ValueError(f"{VAL_SOURCE_CONFIG} parquet files not in the local HF cache ({exc}); "
                         "pass --snapshot_dir") from exc


def _needs(config: str) -> str:
    if config == PART1:
        return "speaker_id"
    if config in (PART3, EXTRA_TEST_CONFIG):
        return "conversation_id"
    return "any"


def attach_clusters(df: pd.DataFrame, snap_arg: Optional[str] = None,
                    val_conversations: bool = False) -> pd.DataFrame:
    """Adds cluster ('spk:<id>' / 'conv:<id>') and cluster_unit. Missing ids of PART1/PART3
    rows are joined from the snapshot; any other row without ids raises ValueError.
    val_conversations=True fills empty conversation_id of *-val rows from the local
    ASR-PART3-Train snapshot via sample_id (in memory only), so they cluster by conversation;
    a looked-up speaker that differs from the row's speaker_id raises ValueError."""
    out = df.copy()
    for c in ("conversation_id", "speaker_id"):
        if c not in out.columns:
            out[c] = ""
    if val_conversations:
        is_val = out["config"].astype(str).str.endswith("-val") & (out["conversation_id"] == "")
        if is_val.any():
            if "sample_id" not in out.columns:
                raise ValueError("--val_conversations needs a sample_id column")
            sids = out.loc[is_val, "sample_id"].astype(str)
            lookup = val_conversation_lookup(sids, resolve_val_snapshot(snap_arg,
                                                                        val_shard_indices(sids)))
            for i in out.index[is_val]:
                sid = str(out.at[i, "sample_id"])
                conv, spk = lookup[sid]
                if not conv:
                    raise ValueError(f"{sid}: no conversation_id in the source data")
                row_spk = str(out.at[i, "speaker_id"])
                if row_spk and row_spk != spk:
                    raise ValueError(f"{sid}: speaker {row_spk!r} in the CSV but {spk!r} in the "
                                     "source data (wrong snapshot or clip id)")
                out.at[i, "conversation_id"] = conv
    need = out["config"].map(_needs)
    missing = (((need == "speaker_id") & (out["speaker_id"] == ""))
               | ((need == "conversation_id") & (out["conversation_id"] == ""))
               | ((need == "any") & (out["speaker_id"] == "") & (out["conversation_id"] == "")))
    if missing.any():
        joinable = out["config"].isin([PART1, PART3])
        bad = missing & ~joinable
        if bad.any():
            raise ValueError(f"rows of config {sorted(set(out.loc[bad, 'config']))} have no "
                             "conversation_id/speaker_id and cannot be joined from the test data")
        lookup = test_cluster_lookup(resolve_snapshot(snap_arg))
        for i in out.index[missing]:
            key = (out.at[i, "config"], int(out.at[i, "row_index"]))
            if key not in lookup:
                raise ValueError(f"row {key} not found in the test data snapshot")
            out.at[i, "conversation_id"], out.at[i, "speaker_id"] = lookup[key]

    clusters, units = [], []
    for config, conv, spk in zip(out["config"], out["conversation_id"], out["speaker_id"]):
        n = _needs(config)
        if n == "speaker_id" or (n == "any" and not conv):
            clusters.append(f"spk:{spk}" if spk else "")
            units.append("speaker")
        else:
            clusters.append(f"conv:{conv}" if conv else "")
            units.append("conversation")
    out["cluster"], out["cluster_unit"] = clusters, units
    empty = out["cluster"] == ""
    if empty.any():
        raise ValueError(f"empty cluster id for config(s) {sorted(set(out.loc[empty, 'config']))}")
    return out


# ============================================================
# BOOTSTRAP
# ============================================================

def group_order(df: pd.DataFrame) -> List[str]:
    configs = list(pd.unique(df["config"]))
    head = [c for c in (PART1, PART3, EXTRA_TEST_CONFIG) if c in configs]
    return head + sorted(c for c in configs if c not in head)


def cluster_sums(group: pd.DataFrame) -> Tuple[List[str], np.ndarray]:
    """(sorted cluster keys, array [n_clusters, 4] of summed COUNTS)."""
    g = group.groupby("cluster", sort=True)[COUNTS].sum()
    return list(g.index), g.to_numpy(dtype=np.int64)


def boot_draws(n_clusters: int, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, n_clusters, size=(n_boot, n_clusters))


def percentile_ci(stats: np.ndarray, confidence: float) -> Tuple[float, float]:
    lo, hi = np.percentile(stats, [100 * (1 - confidence) / 2, 100 * (1 + confidence) / 2])
    return float(lo), float(hi)


def _unit(group: pd.DataFrame) -> str:
    u = set(group["cluster_unit"])
    return u.pop() if len(u) == 1 else "mixed"


def _resampled_sums(arrays: List[np.ndarray], draws: List[np.ndarray]) -> np.ndarray:
    """[n_boot, 4] summed counts; groups (strata) added."""
    return sum(arr[d].sum(axis=1) for arr, d in zip(arrays, draws))


def _strata(df: pd.DataFrame, groups: List[str], n_boot: int, seed: int):
    """Per group: (name, rows, cluster sums, draws), each group with a fresh rng(seed),
    plus the stratified Overall draws (one rng(seed), groups in order)."""
    per = []
    for g in groups:
        sub = df[df["config"] == g]
        _, arr = cluster_sums(sub)
        per.append((g, sub, arr, boot_draws(len(arr), n_boot, np.random.default_rng(seed))))
    rng = np.random.default_rng(seed)
    overall_draws = [boot_draws(len(arr), n_boot, rng) for _, _, arr, _ in per]
    return per, overall_draws


def bootstrap_file(df: pd.DataFrame, n_boot: int = DEFAULT_N_BOOT,
                   confidence: float = DEFAULT_CONFIDENCE, seed: int = DEFAULT_SEED) -> List[dict]:
    """df: rows with COUNTS and cluster columns. One row per (group, metric)."""
    groups = group_order(df)
    per, overall_draws = _strata(df, groups, n_boot, seed)
    blocks = [(g, sub, _unit(sub), len(arr), [arr], [d]) for g, sub, arr, d in per]
    blocks.append(("Overall", df, "stratified", sum(len(a) for _, _, a, _ in per),
                   [a for _, _, a, _ in per], overall_draws))
    rows = []
    for name, sub, unit, n_clusters, arrays, draws in blocks:
        boot = _resampled_sums(arrays, draws)
        tot = sum(a.sum(axis=0) for a in arrays)
        for metric, (ei, ri) in METRICS.items():
            e, r = COUNTS.index(ei), COUNTS.index(ri)
            est = tot[e] / tot[r]
            ref = corpus_metric(sub, metric)
            assert abs(est - ref) < 1e-9, f"{name} {metric}: {est} != corpus_metric {ref}"
            stats = boot[:, e] / boot[:, r]
            lo, hi = percentile_ci(stats, confidence)
            rows.append({"group": name, "cluster_unit": unit, "n_rows": len(sub),
                         "n_clusters": n_clusters, "metric": metric, "estimate": float(est),
                         "ci_low": lo, "ci_high": hi, "boot_mean": float(stats.mean()),
                         "boot_std": float(stats.std(ddof=1)) if n_boot > 1 else 0.0,
                         "n_boot": n_boot, "confidence": confidence, "seed": seed})
    return rows


def bootstrap_paired(a: pd.DataFrame, b: pd.DataFrame, n_boot: int = DEFAULT_N_BOOT,
                     confidence: float = DEFAULT_CONFIDENCE, seed: int = DEFAULT_SEED) -> List[dict]:
    """diff = B - A per (group, metric); A and B use the same cluster draws."""
    cols = KEY + ["reference", "cluster", "cluster_unit"] + COUNTS
    m = a[cols].merge(b[cols], on=KEY, suffixes=("_a", "_b"))
    if len(m) == 0:
        raise ValueError("0 matching rows on [config, row_index] (different split/data?)")
    if (m["reference_a"] != m["reference_b"]).any():
        raise ValueError(f"{int((m['reference_a'] != m['reference_b']).sum())} matched rows have "
                         "different references; the runs are not on identical rows")
    if (m["cluster_a"] != m["cluster_b"]).any():
        raise ValueError("matched rows have different cluster ids in the two files")
    if len(m) < len(a) or len(m) < len(b):
        print(f"WARNING: unmatched rows: {len(a) - len(m)} only in A, {len(b) - len(m)} only in B; "
              f"using the {len(m)} matched rows.")
    base = m[KEY].copy()
    base["cluster"] = m["cluster_a"]
    base["cluster_unit"] = m["cluster_unit_a"]
    da = pd.concat([base, m[[f"{c}_a" for c in COUNTS]].set_axis(COUNTS, axis=1)], axis=1)
    db = pd.concat([base, m[[f"{c}_b" for c in COUNTS]].set_axis(COUNTS, axis=1)], axis=1)

    groups = group_order(da)
    per_a, overall_draws = _strata(da, groups, n_boot, seed)
    arrs_b = [cluster_sums(db[db["config"] == g])[1] for g in groups]
    blocks = [(g, sub, _unit(sub), len(arr), [arr], [ab], [d])
              for (g, sub, arr, d), ab in zip(per_a, arrs_b)]
    blocks.append(("Overall", da, "stratified", sum(len(x) for _, _, x, _ in per_a),
                   [x for _, _, x, _ in per_a], arrs_b, overall_draws))
    rows = []
    for name, sub, unit, n_clusters, arrays_a, arrays_b, draws in blocks:
        boot_a, boot_b = _resampled_sums(arrays_a, draws), _resampled_sums(arrays_b, draws)
        tot_a = sum(x.sum(axis=0) for x in arrays_a)
        tot_b = sum(x.sum(axis=0) for x in arrays_b)
        for metric, (ei, ri) in METRICS.items():
            e, r = COUNTS.index(ei), COUNTS.index(ri)
            va, vb = tot_a[e] / tot_a[r], tot_b[e] / tot_b[r]
            diff = boot_b[:, e] / boot_b[:, r] - boot_a[:, e] / boot_a[:, r]
            lo, hi = percentile_ci(diff, confidence)
            rows.append({"group": name, "cluster_unit": unit, "n_rows": len(sub),
                         "n_clusters": n_clusters, "metric": metric, "a": float(va), "b": float(vb),
                         "diff": float(vb - va), "ci_low": lo, "ci_high": hi,
                         "p_boot_b_not_better": float((diff >= 0).mean()),
                         "n_boot": n_boot, "confidence": confidence, "seed": seed})
    return rows


# ============================================================
# CLI
# ============================================================

def _stem(path) -> str:
    s = Path(path).stem
    return s[len("eval_"):] if s.startswith("eval_") else s


def default_out(files: List[str], paired: bool, n_boot: int, seed: int) -> Path:
    if paired:
        return RESULTS_DIR / f"bootstrap_paired_{_stem(files[0])}__vs__{_stem(files[1])}_n{n_boot}_s{seed}.csv"
    stem = _stem(files[0]) + (f"_plus{len(files) - 1}" if len(files) > 1 else "")
    return RESULTS_DIR / f"bootstrap_{stem}_n{n_boot}_s{seed}.csv"


def _fmt(df: pd.DataFrame, paired: bool) -> str:
    show = df.copy()
    if paired:
        for c in ("a", "b", "diff"):
            show[c] = show[c].map(lambda x: f"{x:.4f}")
        show["95%CI" if show["confidence"].iloc[0] == 0.95 else "CI"] = [
            f"[{lo:+.4f}, {hi:+.4f}]" for lo, hi in zip(df["ci_low"], df["ci_high"])]
        cols = ["group", "cluster_unit", "n_rows", "n_clusters", "metric", "a", "b", "diff",
                show.columns[-1], "p_boot_b_not_better"]
    else:
        show["estimate"] = show["estimate"].map(lambda x: f"{x:.4f}")
        show["95%CI" if show["confidence"].iloc[0] == 0.95 else "CI"] = [
            f"[{lo:.4f}, {hi:.4f}]" for lo, hi in zip(df["ci_low"], df["ci_high"])]
        cols = (["file"] if "file" in show.columns else []) + [
            "group", "cluster_unit", "n_rows", "n_clusters", "metric", "estimate", show.columns[-1]]
    return show[cols].to_string(index=False)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cluster bootstrap CIs for corpus WER/CER of per-row eval CSVs.")
    p.add_argument("files", nargs="+", help="Per-row eval_*.csv files (with --paired: exactly 2, A = baseline).")
    p.add_argument("--paired", action="store_true",
                   help="Compare two runs on identical rows: CI of WER(B) - WER(A), same cluster draws.")
    p.add_argument("--n_boot", type=int, default=DEFAULT_N_BOOT,
                   help=f"Bootstrap resamples (default {DEFAULT_N_BOOT}).")
    p.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE,
                   help=f"Percentile CI level (default {DEFAULT_CONFIDENCE}).")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"Random seed (default {DEFAULT_SEED}).")
    p.add_argument("--out", default=None, help="Output CSV (default results/bootstrap_<...>.csv).")
    p.add_argument("--overwrite", action="store_true", help="Allow replacing an existing --out.")
    p.add_argument("--snapshot_dir", default=None,
                   help="Local HF dataset snapshot for joining cluster ids into old CSVs "
                        "(default: the HF cache, never downloads).")
    p.add_argument("--val_conversations", action="store_true",
                   help="Join conversation_id for *-val rows from the local ASR-PART3-Train snapshot "
                        "via sample_id (in memory; files unchanged), so val rows cluster by "
                        "conversation. Default: val rows cluster by speaker.")
    args = p.parse_args(argv)
    if args.paired and len(args.files) != 2:
        p.error("--paired needs exactly two files")
    if not 0.0 < args.confidence < 1.0:
        p.error("--confidence must be in (0, 1)")
    if args.n_boot < 1:
        p.error("--n_boot must be >= 1")
    return args


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    out = Path(args.out) if args.out else default_out(args.files, args.paired, args.n_boot, args.seed)
    if any(out.resolve() == Path(f).resolve() for f in args.files):
        print(f"ERROR: --out {out} is one of the input files.")
        return 2
    if out.exists() and not args.overwrite:
        print(f"ERROR: {out} exists; pass --overwrite or a different --out.")
        return 2
    try:
        frames = [attach_clusters(row_counts(load_eval_csv(f)), args.snapshot_dir,
                                  args.val_conversations) for f in args.files]
        if args.paired:
            result = pd.DataFrame(bootstrap_paired(frames[0], frames[1], args.n_boot,
                                                   args.confidence, args.seed))
            result.insert(0, "file_b", Path(args.files[1]).name)
            result.insert(0, "file_a", Path(args.files[0]).name)
        else:
            parts = []
            for f, df in zip(args.files, frames):
                r = pd.DataFrame(bootstrap_file(df, args.n_boot, args.confidence, args.seed))
                r.insert(0, "file", Path(f).name)
                parts.append(r)
            result = pd.concat(parts, ignore_index=True)
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 2

    pd.set_option("display.width", 220)
    print("=" * 70)
    if args.paired:
        print(f"PAIRED CLUSTER BOOTSTRAP: B - A  (n_boot={args.n_boot}, seed={args.seed}, "
              f"{args.confidence:.0%} percentile CI)")
        print(f"  A = {args.files[0]}\n  B = {args.files[1]}")
        print("  diff < 0 means B has lower error; a CI that excludes 0 is a real difference.")
    else:
        print(f"CLUSTER BOOTSTRAP  (n_boot={args.n_boot}, seed={args.seed}, "
              f"{args.confidence:.0%} percentile CI)")
    print("=" * 70)
    print(_fmt(result, args.paired))
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False)
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
