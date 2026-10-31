"""
run_lr_sweep.py

Overnight learning-rate / seed sweep for the held-out Whisper-small model. For each run, in
order: train (scripts/train_whisper_speaker_split.py) and then evaluate on the run's own
validation split with temperature fallback (scripts/evaluate_whisper_speaker_split.py
--split val). Never runs a test-split evaluation.

  1. whisper-small-mnsc-sg-15k-heldout-seed43      LR 2e-5, no warmup,  seed 43 (noise floor)
  2. whisper-small-mnsc-sg-15k-heldout-lr1e-5-wu10 LR 1e-5, 10 % warmup, seed 42
  3. whisper-small-mnsc-sg-15k-heldout-lr3e-5-wu10 LR 3e-5, 10 % warmup, seed 42

All runs use data/part3_15k_heldout and the script defaults for everything else (3 epochs,
early stopping, batch 2 x 4). Safe to re-run: a run whose models/<run>/ is complete is not
trained again, an interrupted run resumes from its last checkpoint, and an eval whose
summary CSV exists is skipped. A failed run is logged and the next run starts.
Logs: logs/sweep/<run>.log (timestamped) and logs/sweep/sweep_status.log.

Usage (from the project root, in PowerShell):
    python scripts/run_lr_sweep.py -DryRun       # print the commands, run nothing
    python scripts/run_lr_sweep.py               # the real sweep (about 6.8 h)
    python scripts/run_lr_sweep.py -SmokeTest    # smoke-* runs: 20 steps, 64 val rows
Then: python scripts/compare_lr_sweep.py
"""

import argparse
import ctypes
import datetime
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

BASELINE_RUN = "whisper-small-mnsc-sg-15k-heldout"   # the current shipped model
DATA_NAME = "part3_15k_heldout"
SMOKE_PREFIX = "smoke-"
SMOKE_TRAIN_ARGS = ["--max_steps", "20", "--eval_steps", "10", "--max_eval_samples", "16"]
SMOKE_EVAL_ROWS = 64
MIN_FREE_GB = 20            # peak need about 16.6 GB (see README.md)

# Seed run first: without the noise floor the comparison cannot decide anything.
SWEEP_RUNS = [
    {"name": f"{BASELINE_RUN}-seed43", "learning_rate": 2e-5, "warmup_ratio": 0.0, "seed": 43,
     "purpose": "noise floor: current config, new seed"},
    {"name": f"{BASELINE_RUN}-lr1e-5-wu10", "learning_rate": 1e-5, "warmup_ratio": 0.1, "seed": 42,
     "purpose": "lower LR + 10 % warmup"},
    {"name": f"{BASELINE_RUN}-lr3e-5-wu10", "learning_rate": 3e-5, "warmup_ratio": 0.1, "seed": 42,
     "purpose": "higher LR + 10 % warmup"},
]
BASELINE_HPARAMS = {"learning_rate": 2e-5, "warmup_ratio": 0.0, "seed": 42}

# Estimates for the plan header (measured on the RTX 3070 Laptop)
TRAIN_HOURS_PER_RUN = 2.05   # checkpoints/<baseline>/resource_usage.json full_run_estimate_hours
EVAL_MINUTES_PER_RUN = 12    # full 1,411-row val eval with fallback (measured 0.47 s/row + load)
DISK_GB_FINAL = 11.1         # 3 x (2.77 GB best checkpoint + 0.93 GB model)
DISK_GB_PEAK = 16.6

MODEL_FILES = ["config.json", "generation_config.json", "model.safetensors",
               "tokenizer.json", "tokenizer_config.json"]
PROCESSOR_FILES = ["processor_config.json", "preprocessor_config.json"]   # one of them


# ============================================================
# PURE HELPERS
# ============================================================

def run_name(run: dict, smoke: bool) -> str:
    return (SMOKE_PREFIX if smoke else "") + run["name"]


def train_command(run: dict, smoke: bool) -> List[str]:
    cmd = [sys.executable, "-u", str(SCRIPT_DIR / "train_whisper_speaker_split.py"),
           "--data_name", DATA_NAME, "--run_name", run_name(run, smoke),
           "--learning_rate", f"{run['learning_rate']:g}",
           "--warmup_ratio", f"{run['warmup_ratio']:g}",
           "--seed", str(run["seed"])]
    return cmd + (list(SMOKE_TRAIN_ARGS) if smoke else [])


def eval_command(run: dict, smoke: bool) -> List[str]:
    cmd = [sys.executable, "-u", str(SCRIPT_DIR / "evaluate_whisper_speaker_split.py"),
           "--run_name", run_name(run, smoke), "--split", "val", "--data_name", DATA_NAME]
    return cmd + (["--max_rows", str(SMOKE_EVAL_ROWS)] if smoke else [])


def val_eval_suffix(smoke: bool) -> str:
    """Output suffix the eval script gives these commands (fallback is its default)."""
    return f"_val_rows{SMOKE_EVAL_ROWS}_fb" if smoke else "_val_fb"


def val_summary_path(name: str, smoke: bool, root: Path = PROJECT_ROOT) -> Path:
    return Path(root) / "results" / f"summary_{name}{val_eval_suffix(smoke)}.csv"


def val_eval_path(name: str, smoke: bool, root: Path = PROJECT_ROOT) -> Path:
    return Path(root) / "results" / f"eval_{name}{val_eval_suffix(smoke)}.csv"


def model_complete(model_dir: Path) -> bool:
    """True when models/<run>/ holds everything the training script saves at the end."""
    d = Path(model_dir)
    return (all((d / f).is_file() for f in MODEL_FILES)
            and any((d / f).is_file() for f in PROCESSOR_FILES))


def has_checkpoint(ckpt_dir: Path) -> bool:
    d = Path(ckpt_dir)
    return d.is_dir() and any(p.is_dir() and p.name.startswith("checkpoint-") for p in d.iterdir())


def _ts() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _fmt_secs(s: Optional[float]) -> str:
    if s is None:
        return "-"
    s = int(round(s))
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


# ============================================================
# RUNNING
# ============================================================

def log_line(log_path: Optional[Path], text: str) -> None:
    """Timestamped line to the console and (if given) the log file."""
    line = f"[{_ts()}] {text}"
    print(line, flush=True)
    if log_path is not None:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def run_logged(cmd: List[str], log_path: Path, cwd: Path) -> Tuple[int, float]:
    """Run cmd, streaming its stdout+stderr line by line with timestamps to the console and
    log_path (appended). Returns (exit code, seconds). Ctrl+C stops the child and re-raises."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    t0 = time.time()
    with open(log_path, "a", encoding="utf-8") as fh:
        def emit(text: str) -> None:
            line = f"[{_ts()}] {text}"
            print(line, flush=True)
            fh.write(line + "\n")
            fh.flush()

        emit(f"COMMAND: {subprocess.list2cmdline(cmd)}")
        emit(f"CWD: {cwd}")
        proc = subprocess.Popen(cmd, cwd=str(cwd), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                errors="replace", env=env)
        try:
            for line in proc.stdout:
                line = line.rstrip("\n")
                if line.strip():
                    emit(line)
            rc = proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
            emit("INTERRUPTED (Ctrl+C): child process stopped. Re-run the same command to resume.")
            raise
        dt = time.time() - t0
        emit(f"EXIT CODE {rc} after {_fmt_secs(dt)}")
    return rc, dt


Runner = Callable[[List[str], Path, Path], Tuple[int, float]]


def run_sweep(smoke: bool, dry_run: bool, root: Path = PROJECT_ROOT,
              runner: Runner = run_logged, rows: Optional[List[dict]] = None) -> List[dict]:
    """Train + val-eval every run in SWEEP_RUNS order. Appends one status dict per run to
    `rows` (also returned), so a caller still has the finished rows after Ctrl+C."""
    root = Path(root)
    rows = [] if rows is None else rows
    for run in SWEEP_RUNS:
        name = run_name(run, smoke)
        model_dir, ckpt_dir = root / "models" / name, root / "checkpoints" / name
        summary = val_summary_path(name, smoke, root)
        log_path = root / "logs" / "sweep" / f"{name}.log"
        row = {"run": name, "lr": f"{run['learning_rate']:g}", "warmup": f"{run['warmup_ratio']:g}",
               "seed": run["seed"], "train": "-", "train_time": None, "eval": "-",
               "eval_time": None, "summary": "-"}
        rows.append(row)
        tcmd, ecmd = train_command(run, smoke), eval_command(run, smoke)
        print()
        print("=" * 70)
        print(f"RUN {name}  ({run['purpose']})")
        print("=" * 70)

        # ---- train ----
        if model_complete(model_dir):
            row["train"] = "skipped (complete)"
            msg = f"TRAIN skipped: {model_dir} is complete (never retrained by the runner)."
        else:
            row["train"] = "would resume" if has_checkpoint(ckpt_dir) else "would start fresh"
            msg = (f"TRAIN {'resuming from checkpoint in ' + str(ckpt_dir) if has_checkpoint(ckpt_dir) else 'fresh'}"
                   f": {subprocess.list2cmdline(tcmd)}")
        if dry_run:
            print(f"[dry run] {msg}")
        elif row["train"] == "skipped (complete)":
            log_line(log_path, msg)
        else:
            log_line(log_path, msg)
            rc, dt = runner(tcmd, log_path, root)
            row["train_time"] = dt
            if rc != 0:
                row["train"] = f"failed (rc {rc})"
            elif not model_complete(model_dir):
                row["train"] = "failed (model incomplete)"
            else:
                row["train"] = "done"
            if row["train"] != "done":
                log_line(log_path, f"TRAIN {row['train']}; see this log. Continuing with the next run.")

        # ---- val eval ----
        if summary.exists():
            row["eval"] = "skipped (summary exists)"
            row["summary"] = summary.name
            msg = f"EVAL skipped: {summary} exists."
        elif not dry_run and not model_complete(model_dir):
            row["eval"] = "skipped (no model)"
            msg = f"EVAL skipped: no complete model in {model_dir}."
        else:
            row["eval"] = "would run"
            msg = f"EVAL: {subprocess.list2cmdline(ecmd)}"
        if dry_run:
            print(f"[dry run] {msg}")
            continue
        log_line(log_path, msg)
        if row["eval"] != "would run":
            continue
        rc, dt = runner(ecmd, log_path, root)
        row["eval_time"] = dt
        if rc == 0 and summary.exists():
            row["eval"] = "done"
            row["summary"] = summary.name
        else:
            row["eval"] = f"failed (rc {rc})"
            hint = ""
            if val_eval_path(name, smoke, root).exists() and not summary.exists():
                hint = (f" An eval CSV without a summary blocks the eval script: delete only "
                        f"{val_eval_path(name, smoke, root)} and re-run.")
            log_line(log_path, f"EVAL {row['eval']}; see this log.{hint}")
    return rows


def format_status_table(rows: List[dict]) -> str:
    cols = ["run", "lr", "warmup", "seed", "train", "train_time", "eval", "eval_time", "summary"]
    data = [[str(r[c]) if c not in ("train_time", "eval_time") else _fmt_secs(r[c]) for c in cols]
            for r in rows]
    widths = [max(len(c), *(len(d[i]) for d in data)) if data else len(c) for i, c in enumerate(cols)]
    lines = ["  ".join(c.ljust(w) for c, w in zip(cols, widths)),
             "  ".join("-" * w for w in widths)]
    lines += ["  ".join(v.ljust(w) for v, w in zip(d, widths)) for d in data]
    return "\n".join(lines)


def any_failed(rows: List[dict]) -> bool:
    return any(r["train"].startswith("failed") or r["eval"].startswith("failed") for r in rows)


# ============================================================
# SLEEP GUARD (Windows)
# ============================================================

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


def keep_awake(on: bool) -> bool:
    """Ask Windows not to idle-sleep while the sweep runs (does not stop lid-close sleep).
    No-op elsewhere. Returns True if the call succeeded."""
    if os.name != "nt":
        return False
    try:
        flags = ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0)
        return bool(ctypes.windll.kernel32.SetThreadExecutionState(flags))
    except Exception:
        return False


# ============================================================
# MAIN
# ============================================================

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Overnight LR / seed sweep: train + val eval (fallback) "
                                            "for each run; never touches the test splits.")
    p.add_argument("--dry_run", "-DryRun", action="store_true",
                   help="Print the commands and skip decisions; run and create nothing.")
    p.add_argument("--smoke", "-SmokeTest", action="store_true",
                   help=f"Smoke configuration: run names prefixed '{SMOKE_PREFIX}', training "
                        f"{' '.join(SMOKE_TRAIN_ARGS)}, val eval --max_rows {SMOKE_EVAL_ROWS}.")
    return p.parse_args(argv)


def plan_header(smoke: bool) -> str:
    lines = ["=" * 70, "LR / SEED SWEEP" + ("  (SMOKE TEST)" if smoke else ""), "=" * 70,
             f"Data          : data/{DATA_NAME} (unchanged; no rebuild)",
             f"Compared with : {BASELINE_RUN} (LR 2e-5, no warmup, seed 42)"]
    for i, r in enumerate(SWEEP_RUNS, 1):
        lines.append(f"  {i}. {run_name(r, smoke):55s} LR {r['learning_rate']:g}  warmup "
                     f"{r['warmup_ratio']:g}  seed {r['seed']}  ({r['purpose']})")
    if smoke:
        lines.append(f"Smoke: training {' '.join(SMOKE_TRAIN_ARGS)}; val eval {SMOKE_EVAL_ROWS} rows.")
    else:
        total_h = len(SWEEP_RUNS) * (TRAIN_HOURS_PER_RUN + EVAL_MINUTES_PER_RUN / 60)
        lines.append(f"Expected time : about {total_h:.1f} h ({len(SWEEP_RUNS)} x {TRAIN_HOURS_PER_RUN} h "
                     f"training + {len(SWEEP_RUNS)} x ~{EVAL_MINUTES_PER_RUN} min val eval)")
        lines.append(f"Expected disk : about {DISK_GB_FINAL} GB at the end, {DISK_GB_PEAK} GB peak")
    lines.append("Keep the laptop plugged in, sleep set to Never, and do not use the GPU meanwhile.")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None, root: Path = PROJECT_ROOT,
         runner: Runner = run_logged) -> int:
    args = parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except Exception:
            pass
    root = Path(root)
    print(plan_header(args.smoke))
    if args.dry_run:
        print("\nDRY RUN: nothing is run or created.")
    else:
        free_gb = shutil.disk_usage(root).free / 1e9
        print(f"Free disk     : {free_gb:.1f} GB")
        if free_gb < MIN_FREE_GB:
            print(f"ERROR: less than {MIN_FREE_GB} GB free on the project drive; free some space first.")
            return 2
        print("Sleep guard   :", "on (idle sleep blocked while running)" if keep_awake(True)
              else "not available (set Windows sleep to Never)")

    rows: List[dict] = []
    interrupted = False
    started = _ts()
    try:
        run_sweep(args.smoke, args.dry_run, root, runner, rows)
    except KeyboardInterrupt:
        interrupted = True
    finally:
        if not args.dry_run:
            keep_awake(False)

    table = format_status_table(rows)
    print()
    print("=" * 70)
    print("SWEEP STATUS" + (" (dry run)" if args.dry_run else "") + (" (INTERRUPTED)" if interrupted else ""))
    print("=" * 70)
    print(table)
    if not args.dry_run:
        status_log = root / "logs" / "sweep" / f"{SMOKE_PREFIX if args.smoke else ''}sweep_status.log"
        status_log.parent.mkdir(parents=True, exist_ok=True)
        with open(status_log, "a", encoding="utf-8") as fh:
            fh.write(f"Sweep started {started}, ended {_ts()}"
                     f"{' (INTERRUPTED)' if interrupted else ''}\n{table}\n\n")
        print(f"\nStatus appended to {status_log}")
    print("\nNext step (after all runs are done):")
    print("  python scripts\\compare_lr_sweep.py" + (" --smoke" if args.smoke else ""))
    if interrupted:
        print("Interrupted. Re-run the same command to resume; finished parts are skipped.")
        return 130
    return 1 if any_failed(rows) else 0


if __name__ == "__main__":
    sys.exit(main())
