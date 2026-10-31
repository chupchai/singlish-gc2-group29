"""
compare_lr_sweep.py

Morning step of the learning-rate / seed sweep (run_lr_sweep.py). Reads the val fallback
results of the current model (whisper-small-mnsc-sg-15k-heldout) and of the 3 sweep runs,
prints one table and applies the decision rule. Read-only on all inputs; it never runs an
evaluation (and never a test-split one).

Table: run, LR, warmup, seed, best step / epoch / eval_loss (checkpoints/<run>/.../
trainer_state.json), val WER, val CER, loop rows (n_pred_gt3x_ref), and the paired cluster
bootstrap WER diff (run - current) with its 95 % CI (bootstrap_wer_ci.py; 1000 resamples,
seed 42; val rows clustered by conversation, joined from the local ASR-PART3-Train snapshot;
speaker clusters if that snapshot is missing).

Decision rule (chosen before the sweep):
    noise floor = |WER(seed run) - WER(current)|   (same config, only the seed differs)
    a run wins only if its paired WER diff vs the current model has a 95 % CI entirely
    below 0 AND its improvement WER(current) - WER(run) is larger than the noise floor.
    The seed run is the noise reference and is never a candidate. Several winners: the one
    with the lowest val WER. No winner: keep the current model.
Improvement and noise floor are taken from the paired comparison (matched rows; in a full
run these are all 1,411 val rows, so they equal the summary WERs).

Writes results/lr_sweep_comparison_val_fb.csv (smoke: ..._smoke_val_rows64_fb.csv) only when
every run has results, and never overwrites an existing file.

Usage (from the project root):
    python scripts/compare_lr_sweep.py
    python scripts/compare_lr_sweep.py --smoke      # after run_lr_sweep.py -SmokeTest
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from bootstrap_wer_ci import (attach_clusters, bootstrap_paired, load_eval_csv,
                              resolve_val_snapshot, row_counts, val_shard_indices)
from run_lr_sweep import (BASELINE_HPARAMS, BASELINE_RUN, DATA_NAME, PROJECT_ROOT,
                          SMOKE_EVAL_ROWS, SWEEP_RUNS, run_name, val_eval_path,
                          val_summary_path)

NOISE_RUN = SWEEP_RUNS[0]["name"]          # the seed run
DEFAULT_N_BOOT = 1000
DEFAULT_SEED = 42
HO_TEST_BASE = f"results\\eval_{BASELINE_RUN}_ho-part3_15k_heldout_fb.csv"
HX_TEST_BASE = f"results\\eval_{BASELINE_RUN}_hx-part3_15k_heldout_extra_test_fb.csv"
COLUMNS = ["run", "role", "learning_rate", "warmup_ratio", "seed", "hparams_source",
           "best_step", "best_epoch", "best_eval_loss", "val_WER", "val_CER", "loop_rows",
           "n_rows", "wer_diff_vs_current", "ci_low", "ci_high", "cluster_unit", "n_clusters",
           "ci_excludes_0_better", "improvement", "noise_floor", "beats_noise_floor", "wins",
           "status"]


# ============================================================
# INPUTS
# ============================================================

def read_summary(path: Path) -> Optional[dict]:
    """Overall row of a summary_*.csv, or None if the file is missing."""
    path = Path(path)
    if not path.exists():
        return None
    df = pd.read_csv(path)
    row = df[df["group"] == "Overall"].iloc[0]
    return {"val_WER": float(row["corpus_WER"]), "val_CER": float(row["corpus_CER"]),
            "loop_rows": int(row["n_pred_gt3x_ref"]), "samples": int(row["samples"])}


def best_from_trainer_state(ckpt_dir: Path) -> Optional[dict]:
    """best_step / best_epoch / best_eval_loss from the newest checkpoint's trainer_state.json
    (eval_history in resource_usage.json as fallback for the epoch); None if absent."""
    ckpt_dir = Path(ckpt_dir)
    states = []
    for p in ckpt_dir.glob("checkpoint-*/trainer_state.json"):
        with open(p, encoding="utf-8") as fh:
            states.append(json.load(fh))
    if not states:
        return None
    st = max(states, key=lambda s: s.get("global_step", 0))
    step, loss = st.get("best_global_step"), st.get("best_metric")
    epoch = next((e.get("epoch") for e in st.get("log_history", [])
                  if e.get("step") == step and "eval_loss" in e), None)
    if epoch is None:
        ru = ckpt_dir / "resource_usage.json"
        if ru.exists():
            with open(ru, encoding="utf-8") as fh:
                hist = json.load(fh).get("eval_history", [])
            epoch = next((e.get("epoch") for e in hist if e.get("step") == step), None)
    return {"best_step": step, "best_epoch": round(float(epoch), 2) if epoch is not None else None,
            "best_eval_loss": round(float(loss), 4) if loss is not None else None}


def run_hparams(ckpt_dir: Path, default: dict) -> Tuple[dict, str]:
    """(learning_rate, warmup_ratio, seed) from resource_usage.json 'hyperparameters', else the
    planned values (runs from before the flags existed have no record)."""
    ru = Path(ckpt_dir) / "resource_usage.json"
    if ru.exists():
        with open(ru, encoding="utf-8") as fh:
            hp = json.load(fh).get("hyperparameters")
        if isinstance(hp, dict):
            return {k: hp.get(k) for k in ("learning_rate", "warmup_ratio", "seed")}, "logged"
    return {k: default[k] for k in ("learning_rate", "warmup_ratio", "seed")}, "planned"


def val_conversations(root: Path) -> set:
    info = Path(root) / "data" / DATA_NAME / "speaker_split_info.json"
    with open(info, encoding="utf-8") as fh:
        return set(json.load(fh)["split"]["val_conversations"])


def load_clustered(csv: Path, cluster: str, snap: Optional[str], root: Path) -> pd.DataFrame:
    df = attach_clusters(row_counts(load_eval_csv(csv)), snap, cluster == "conversation")
    if cluster == "conversation":
        joined = set(df.loc[df["cluster_unit"] == "conversation", "conversation_id"])
        extra = joined - val_conversations(root)
        if extra:
            raise ValueError(f"{csv}: conversations {sorted(extra)[:5]} are not in the val split of "
                             f"data/{DATA_NAME}")
    return df


def resolve_cluster(cluster: str, base_csv: Path, snapshot_dir: Optional[str]) -> Tuple[str, Optional[str]]:
    """('conversation', snapshot) if the val source shards are available locally, else
    ('speaker', None) with a WARNING."""
    if cluster != "conversation":
        return cluster, snapshot_dir
    try:
        ids = pd.read_csv(base_csv, usecols=["sample_id"], keep_default_na=False)["sample_id"]
        snap = resolve_val_snapshot(snapshot_dir, val_shard_indices(ids))
        return "conversation", str(snap)
    except ValueError as exc:
        print(f"WARNING: cannot join val conversations ({exc}); clustering val rows by speaker.")
        return "speaker", None


def paired_diff(base: pd.DataFrame, run_df: pd.DataFrame, n_boot: int, seed: int) -> dict:
    res = pd.DataFrame(bootstrap_paired(base, run_df, n_boot, 0.95, seed))
    r = res[(res["group"] == "Overall") & (res["metric"] == "WER")].iloc[0]
    sub = res[(res["group"] != "Overall") & (res["metric"] == "WER")]
    return {"a": float(r["a"]), "b": float(r["b"]), "diff": float(r["diff"]),
            "ci_low": float(r["ci_low"]), "ci_high": float(r["ci_high"]),
            "n_clusters": int(r["n_clusters"]), "n_rows": int(r["n_rows"]),
            "cluster_unit": "/".join(sorted(set(sub["cluster_unit"])))}


# ============================================================
# DECISION
# ============================================================

def decide(table: pd.DataFrame) -> Tuple[Optional[str], List[str], pd.DataFrame]:
    """Apply the decision rule. table: one 'current' row, one 'noise' row (seed run) and
    'candidate' rows, with status, wer_diff_vs_current and ci_high. Returns (winner run or
    None, reasons, table with ci_excludes_0_better / improvement / noise_floor /
    beats_noise_floor / wins filled)."""
    t = table.copy()
    for c in ("ci_excludes_0_better", "beats_noise_floor", "wins"):
        t[c] = t[c].astype(object) if c in t.columns else None
    for c in ("improvement", "noise_floor"):
        t[c] = t[c].astype(object) if c in t.columns else None
    reasons: List[str] = []
    cur = t[t["role"] == "current"]
    noise = t[t["role"] == "noise"]
    if cur.empty or cur.iloc[0]["status"] != "ok":
        return None, ["current model val results missing: no decision"], t
    if noise.empty or noise.iloc[0]["status"] != "ok":
        reasons.append("seed run missing: noise floor not available, so no run can win")
        return None, reasons, t
    nf = abs(float(noise.iloc[0]["wer_diff_vs_current"]))
    ni = noise.index[0]
    t.at[ni, "improvement"] = -float(noise.iloc[0]["wer_diff_vs_current"])
    t.at[ni, "noise_floor"] = nf
    t.at[ni, "ci_excludes_0_better"] = bool(float(noise.iloc[0]["ci_high"]) < 0)
    reasons.append(f"noise floor = |WER(seed run) - WER(current)| = {nf:.4f}")
    winners = []
    for i, r in t[t["role"] == "candidate"].iterrows():
        if r["status"] != "ok":
            reasons.append(f"{r['run']}: pending (no val results)")
            continue
        imp = -float(r["wer_diff_vs_current"])
        ci_ok = bool(float(r["ci_high"]) < 0)
        beats = bool(imp > nf)
        win = ci_ok and beats
        t.at[i, "improvement"], t.at[i, "noise_floor"] = imp, nf
        t.at[i, "ci_excludes_0_better"], t.at[i, "beats_noise_floor"], t.at[i, "wins"] = ci_ok, beats, win
        reasons.append(f"{r['run']}: improvement {imp:+.4f}, 95% CI of diff "
                       f"[{float(r['ci_low']):+.4f}, {float(r['ci_high']):+.4f}] "
                       f"{'excludes 0 (better)' if ci_ok else 'does not exclude 0 on the better side'}; "
                       f"{'beats' if beats else 'does not beat'} the noise floor {nf:.4f} -> "
                       f"{'WINS' if win else 'no'}")
        if win:
            winners.append((float(r["val_WER"]), r["run"]))
    t.loc[t["role"] == "noise", "wins"] = False
    winner = min(winners)[1] if winners else None
    if len(winners) > 1:
        reasons.append(f"{len(winners)} runs win; picking the lowest val WER: {winner}")
    return winner, reasons, t


def test_commands(run: str) -> List[str]:
    return [
        f"python scripts\\evaluate_whisper_speaker_split.py --run_name {run} --heldout_test part3_15k_heldout",
        f"python scripts\\evaluate_whisper_speaker_split.py --run_name {run} --heldout_extra_test part3_15k_heldout_extra_test",
        f"python scripts\\bootstrap_wer_ci.py --paired {HO_TEST_BASE} results\\eval_{run}_ho-part3_15k_heldout_fb.csv",
        f"python scripts\\bootstrap_wer_ci.py --paired {HX_TEST_BASE} results\\eval_{run}_hx-part3_15k_heldout_extra_test_fb.csv",
    ]


def recommendation_text(winner: Optional[str], reasons: List[str]) -> str:
    if winner is None:
        if any("noise floor not available" in r for r in reasons):
            why = "noise floor not available"
        elif reasons and "missing" in reasons[0]:
            why = reasons[0]
        else:
            why = "no run beat the current model under the rule"
        return (f"Recommendation: keep the current model ({BASELINE_RUN}) ({why}).\n"
                "  Do not run the test splits for any sweep run.")
    lines = [f"Recommendation: test {winner} once.",
             "  Run these commands once (temperature fallback is the eval default). "
             "Test once, do not tune on these numbers:"]
    lines += [f"    {c}" for c in test_commands(winner)]
    return "\n".join(lines)


# ============================================================
# TABLE
# ============================================================

def build_table(smoke: bool, root: Path = PROJECT_ROOT, cluster: str = "conversation",
                snapshot_dir: Optional[str] = None, n_boot: int = DEFAULT_N_BOOT,
                seed: int = DEFAULT_SEED) -> Tuple[pd.DataFrame, List[Path]]:
    """(table, input files used). The current model always uses its full _val_fb files."""
    root = Path(root)
    rows, inputs = [], []
    base_sum = val_summary_path(BASELINE_RUN, False, root)
    base_csv = val_eval_path(BASELINE_RUN, False, root)
    specs = [(BASELINE_RUN, "current", BASELINE_HPARAMS, base_sum, base_csv)]
    for r in SWEEP_RUNS:
        name = run_name(r, smoke)
        specs.append((name, "noise" if r["name"] == NOISE_RUN else "candidate", r,
                      val_summary_path(name, smoke, root), val_eval_path(name, smoke, root)))

    base_df = None
    unit, snap = cluster, snapshot_dir
    if base_csv.exists() and base_sum.exists():
        unit, snap = resolve_cluster(cluster, base_csv, snapshot_dir)
        base_df = load_clustered(base_csv, unit, snap, root)
    for name, role, default, s_path, e_path in specs:
        hp, src = run_hparams(root / "checkpoints" / name, default)
        row = dict.fromkeys(COLUMNS)
        row.update({"run": name, "role": role, "hparams_source": src, **hp})
        row.update(best_from_trainer_state(root / "checkpoints" / name) or {})
        summ = read_summary(s_path)
        ok = summ is not None and e_path.exists() and (role == "current" or base_df is not None)
        row["status"] = "ok" if ok else "pending"
        if summ:
            row.update({k: summ[k] for k in ("val_WER", "val_CER", "loop_rows")})
            row["n_rows"] = summ["samples"]
        if ok:
            inputs += [s_path, e_path]
            if role != "current":
                d = paired_diff(base_df, load_clustered(e_path, unit, snap, root), n_boot, seed)
                row.update({"wer_diff_vs_current": d["diff"], "ci_low": d["ci_low"],
                            "ci_high": d["ci_high"], "cluster_unit": d["cluster_unit"],
                            "n_clusters": d["n_clusters"], "n_rows": d["n_rows"]})
        rows.append(row)
    return pd.DataFrame(rows, columns=COLUMNS), inputs


def default_out(smoke: bool, root: Path = PROJECT_ROOT) -> Path:
    tag = f"smoke_val_rows{SMOKE_EVAL_ROWS}_fb" if smoke else "val_fb"
    return Path(root) / "results" / f"lr_sweep_comparison_{tag}.csv"


def _show(t: pd.DataFrame) -> str:
    show = t.copy()
    for c in ("val_WER", "val_CER", "wer_diff_vs_current", "ci_low", "ci_high", "improvement",
              "noise_floor"):
        show[c] = [f"{x:+.4f}" if c in ("wer_diff_vs_current", "ci_low", "ci_high", "improvement")
                   and x is not None and pd.notna(x) else (f"{x:.4f}" if x is not None and pd.notna(x)
                                                           else "-") for x in show[c]]
    show["95%CI"] = [f"[{lo}, {hi}]" if lo != "-" else "-" for lo, hi in zip(show["ci_low"], show["ci_high"])]
    show["learning_rate"] = [f"{x:g}" if x is not None and pd.notna(x) else "-" for x in show["learning_rate"]]
    for c in ("best_step", "loop_rows", "n_rows", "n_clusters"):
        show[c] = [str(int(x)) if x is not None and pd.notna(x) else "-" for x in show[c]]
    cols = ["run", "learning_rate", "warmup_ratio", "seed", "best_step", "best_epoch",
            "best_eval_loss", "val_WER", "val_CER", "loop_rows", "n_rows", "wer_diff_vs_current",
            "95%CI", "cluster_unit", "n_clusters", "improvement", "beats_noise_floor", "wins", "status"]
    return show[cols].fillna("-").to_string(index=False)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare the LR / seed sweep runs with the current model "
                                            "on val and apply the decision rule.")
    p.add_argument("--smoke", action="store_true",
                   help="Use the smoke-* runs of run_lr_sweep.py -SmokeTest (mechanics check only).")
    p.add_argument("--cluster", choices=("conversation", "speaker"), default="conversation",
                   help="Bootstrap cluster unit for val rows (default conversation, joined from the "
                        "local ASR-PART3-Train snapshot; falls back to speaker if it is missing).")
    p.add_argument("--snapshot_dir", default=None,
                   help="Local HF dataset snapshot with ASR-PART3-Train (default: the HF cache).")
    p.add_argument("--n_boot", type=int, default=DEFAULT_N_BOOT,
                   help=f"Bootstrap resamples (default {DEFAULT_N_BOOT}).")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"Bootstrap seed (default {DEFAULT_SEED}).")
    p.add_argument("--out", default=None,
                   help="Output CSV (default results/lr_sweep_comparison_val_fb.csv; never overwritten).")
    args = p.parse_args(argv)
    if args.n_boot < 1:
        p.error("--n_boot must be >= 1")
    return args


def main(argv: Optional[List[str]] = None, root: Path = PROJECT_ROOT) -> int:
    args = parse_args(argv)
    root = Path(root)
    out = Path(args.out) if args.out else default_out(args.smoke, root)
    if args.smoke:
        print("SMOKE MODE: sweep rows are 20-step models on the first "
              f"{SMOKE_EVAL_ROWS} val rows. The decision below only checks the mechanics.")
    try:
        table, inputs = build_table(args.smoke, root, args.cluster, args.snapshot_dir,
                                    args.n_boot, args.seed)
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 2
    winner, reasons, table = decide(table)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print("=" * 70)
    print("LR / SEED SWEEP: VAL COMPARISON (fallback decoding; diff = run - current, "
          f"paired cluster bootstrap, n_boot={args.n_boot}, seed={args.seed}, 95% CI)")
    print("=" * 70)
    print(_show(table))
    print()
    print("Decision rule: a run wins only if its 95% CI of the WER diff is entirely below 0 AND its")
    print("improvement is larger than the noise floor |WER(seed run) - WER(current)|.")
    for r in reasons:
        print(f"  - {r}")
    print()
    print(recommendation_text(winner, reasons))
    print()

    if (table["status"] == "pending").any():
        print("Some runs are pending: nothing written (the output file name stays free).")
        return 0
    if any(out.resolve() == Path(f).resolve() for f in inputs):
        print(f"ERROR: --out {out} is one of the input files.")
        return 2
    if out.exists():
        print(f"ERROR: {out} exists; it is never overwritten. Pass a different --out.")
        return 2
    out.parent.mkdir(parents=True, exist_ok=True)
    table.assign(decision=winner or "keep current").to_csv(out, index=False)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
