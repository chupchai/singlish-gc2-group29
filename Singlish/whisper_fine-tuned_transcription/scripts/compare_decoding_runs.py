"""
Compare two or more per-row eval CSVs from evaluate_whisper_speaker_split.py.

Purpose: pick decoding settings (no_repeat_ngram_size, temperature fallback, thresholds)
by comparing runs on the SAME rows, e.g. several --split val runs of one model.
Prints only; writes nothing.

For every file: corpus WER/CER per group (Overall + each config) and the number of
loop rows (prediction > 3x the reference words, same check as the eval summary).
For every file after the first: rows joined on [config, row_index] with the first
file, better / worse / same row counts and the summed per-row WER gain / loss.

Usage (from the project root):
    python scripts/compare_decoding_runs.py results/eval_whisper-small-mnsc-sg-15k_val.csv results/eval_whisper-small-mnsc-sg-15k_val_nrng5.csv
"""

import argparse
import sys
from typing import List, Optional

import pandas as pd

# Same scripts/ folder: reuse the eval script's metric + loop heuristic
from evaluate_whisper_speaker_split import corpus_metric, n_runaway

REQUIRED_COLUMNS = ["config", "row_index", "reference", "prediction", "WER", "CER"]
KEY = ["config", "row_index"]


def load_run(path) -> pd.DataFrame:
    """Per-row eval CSV; empty predictions stay '' (jiwer needs strings)."""
    df = pd.read_csv(path, keep_default_na=False)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    df["reference"] = df["reference"].astype(str)
    df["prediction"] = df["prediction"].astype(str)
    return df


def _groups(df: pd.DataFrame):
    yield "Overall", df
    for config in pd.unique(df["config"]):
        yield config, df[df["config"] == config]


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Overall + per-config corpus WER/CER and loop-row counts."""
    rows = []
    for name, group in _groups(df):
        rows.append({
            "group": name,
            "samples": len(group),
            "corpus_WER": corpus_metric(group, "WER"),
            "corpus_CER": corpus_metric(group, "CER"),
            "n_pred_gt3x_ref": n_runaway(group),
        })
    return pd.DataFrame(rows)


def row_changes(base: pd.DataFrame, other: pd.DataFrame) -> dict:
    """Per-row WER changes of `other` vs `base`, joined on [config, row_index]."""
    merged = base.merge(other, on=KEY, suffixes=("_base", "_new"))
    table = []
    for name, group in _groups(merged):
        diff = group["WER_new"] - group["WER_base"]
        gain = float((-diff).clip(lower=0).sum())
        loss = float(diff.clip(lower=0).sum())
        table.append({
            "group": name,
            "matched": len(group),
            "better": int((diff < 0).sum()),
            "worse": int((diff > 0).sum()),
            "same": int((diff == 0).sum()),
            "wer_gain": gain,
            "wer_loss": loss,
            "net": gain - loss,
        })
    return {
        "table": pd.DataFrame(table),
        "n_matched": len(merged),
        "n_reference_mismatch": int((merged["reference_base"] != merged["reference_new"]).sum()),
        "n_unmatched_base": len(base) - len(merged),
        "n_unmatched_new": len(other) - len(merged),
    }


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Compare per-row eval CSVs (read-only).")
    p.add_argument("files", nargs="+", help="Two or more eval_*.csv files; the first is the baseline.")
    args = p.parse_args(argv)
    if len(args.files) < 2:
        p.error("need at least two eval CSV files")

    runs = []
    for path in args.files:
        try:
            runs.append(load_run(path))
        except (OSError, ValueError) as exc:
            print(f"ERROR: {exc}")
            return 2

    for path, df in zip(args.files, runs):
        print("=" * 70)
        print(path)
        print("=" * 70)
        print(summarize(df).to_string(index=False))
        print()

    base_path, base = args.files[0], runs[0]
    for path, df in zip(args.files[1:], runs[1:]):
        print("=" * 70)
        print(f"{path}\n  vs {base_path}  (per-row WER, joined on config + row_index)")
        print("=" * 70)
        ch = row_changes(base, df)
        if ch["n_matched"] == 0:
            print("WARNING: 0 matching rows (different split/data_name?); skipped.\n")
            continue
        if ch["n_unmatched_base"] or ch["n_unmatched_new"]:
            print(f"WARNING: unmatched rows: {ch['n_unmatched_base']} only in the first file, "
                  f"{ch['n_unmatched_new']} only in this file.")
        if ch["n_reference_mismatch"]:
            print(f"WARNING: {ch['n_reference_mismatch']} matched rows have different references "
                  "(different split/data_name?).")
        print(ch["table"].to_string(index=False))
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
