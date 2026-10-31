"""
train_whisper_speaker_split.py

Fine-tune Whisper-small on MNSC Extended data using preprocessed data
from preprocess_mnsc_data.py.

Audio is decoded lazily, one clip per __getitem__, from the sharded FLAC
dataset in data/<DATA_NAME>/, so RAM use does not grow with dataset size.

Usage (from the project root):
    1. python scripts/preprocess_mnsc_data.py                 # -> data/part3_15k/
    2. python scripts/train_whisper_speaker_split.py          # RUN_NAME whisper-small-mnsc-sg-15k

    # Smoke run (20 optimizer steps, tiny eval):
    python scripts/train_whisper_speaker_split.py --run_name smoke-whisper-mnsc \
        --max_steps 20 --eval_steps 10 --max_eval_samples 16

    # Old 4k files (data/preprocessed_*.parquet), eager in-memory:
    python scripts/train_whisper_speaker_split.py --data_name legacy --run_name <new name>

    # Learning-rate / seed sweep run (run_lr_sweep.py builds these commands):
    python scripts/train_whisper_speaker_split.py --data_name part3_15k_heldout \
        --run_name whisper-small-mnsc-sg-15k-heldout-lr1e-5-wu10 \
        --learning_rate 1e-5 --warmup_ratio 0.1 --seed 42
"""

# ============================================================
# 1. IMPORTS
# ============================================================

import argparse
import json
import math
import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    TrainerState,
    TrainerControl,
)

# Import shared preprocessing (same scripts/ folder)
from preprocess_mnsc_data import (
    load_preprocessed_data,
    load_preprocessed_index,
    read_clip_audio,
)


# ============================================================
# 2. CONFIGURATION
# ============================================================

# Output directories (resolved relative to this script's location
# so the script works regardless of the shell's cwd)
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent              # scripts/ holds code only

# Default dataset (built by preprocess_mnsc_data.py). 'legacy' = old 4k files.
DEFAULT_DATA_NAME = "part3_15k"

# One descriptive name per experiment: model_data_size_split_key-hyperparams.
# Default is derived from the dataset size (15,000 clips ->
# whisper-small-mnsc-sg-15k); override with --run_name. Pass the same name
# to evaluate_whisper_speaker_split.py --run_name.
RUN_NAME_PREFIX = "whisper-small-mnsc-sg"
LEGACY_RUN_NAME = "whisper-small-mnsc-sg-4k"

# Training hyperparameters (optimized for RTX 3070 Laptop 8GB VRAM)
BATCH_SIZE = 2              # Clips per step; limited by 8 GB VRAM
GRADIENT_ACCUMULATION = 4   # Effective batch = 2 x 4 = 8
LEARNING_RATE = 2e-5        # --learning_rate default
DEFAULT_WARMUP_RATIO = 0.0  # --warmup_ratio default: no warmup (HF default)
DEFAULT_SEED = 42           # --seed default: HF TrainingArguments default; all existing models used it
NUM_EPOCHS = 3              # --epochs default
LOGGING_STEPS = 25


# ============================================================
# 3. PYTORCH DATASET
# ============================================================

class WhisperMNSCDataset(Dataset):
    """
    PyTorch Dataset over a list of clip items. Items from
    load_preprocessed_index() carry only shard/row offsets and the audio is
    decoded lazily here; legacy items that already contain "audio" are used
    as-is. Whisper feature extraction happens on-the-fly.
    """

    def __init__(
        self,
        records: List[dict],
        processor: Any
    ):
        self.records = records
        self.processor = processor
        self._cache: dict = {}

    def __len__(self) -> int:
        return len(self.records)

    def __getstate__(self):
        # Open ParquetFile handles are not picklable (DataLoader workers)
        state = self.__dict__.copy()
        state["_cache"] = {}
        return state

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]

        audio = record.get("audio")
        if audio is None:
            audio = read_clip_audio(record, self._cache)

        # Convert audio waveform to Whisper log-Mel features
        input_features = self.processor.feature_extractor(
            audio,
            sampling_rate=record["sampling_rate"]
        ).input_features[0]

        # Convert transcript to token IDs
        labels = self.processor.tokenizer(
            record["text"]
        ).input_ids

        return {
            "input_features": input_features,
            "labels": labels,
        }


# ============================================================
# 4. DATA COLLATOR
# ============================================================

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Pads variable-length transcripts so samples can be
    batched together.
    """

    processor: Any
    decoder_start_token_id: int

    def __call__(
        self,
        features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> dict:

        # Audio input features
        input_features = [
            {"input_features": f["input_features"]}
            for f in features
        ]
        batch = self.processor.feature_extractor.pad(
            input_features,
            return_tensors="pt"
        )

        # Label (transcript token) ids
        label_features = [
            {"input_ids": f["labels"]}
            for f in features
        ]
        labels_batch = self.processor.tokenizer.pad(
            label_features,
            return_tensors="pt"
        )

        # Replace padding token id with -100 so loss ignores it
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1),
            -100
        )

        # Strip BOS token if already prepended by the tokenizer
        if (
            labels[:, 0] == self.decoder_start_token_id
        ).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


# ============================================================
# 5. CALLBACKS
# ============================================================

# Stops if eval_loss doesn't improve for 2 consecutive evaluations
class EarlyStoppingCallbackWithPatience(TrainerCallback):
    """Early stopping callback with configurable patience."""

    def __init__(self, patience=3, min_delta=0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.wait_count = 0
        self.stopped_epoch = 0

    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, metrics=None, **kwargs):
        if metrics is None:
            return

        current_loss = metrics.get("eval_loss")
        if current_loss is None:
            return

        if self.best_loss is None:
            self.best_loss = current_loss
            return

        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.wait_count = 0
        else:
            self.wait_count += 1
            if self.wait_count >= self.patience:
                control.should_training_stop = True
                print(f"\nEarly stopping triggered after {state.epoch:.2f} epochs.")
                print(f"No improvement for {self.patience} evaluations.")
                print(f"Best eval_loss: {self.best_loss:.4f}")


def _rss_mb() -> Tuple[float, Optional[float]]:
    """(RSS of this process + children in MB, Windows peak working set in MB)."""
    import psutil
    proc = psutil.Process()
    rss = proc.memory_info().rss
    for child in proc.children(recursive=True):
        try:
            rss += child.memory_info().rss
        except Exception:
            pass
    peak = getattr(proc.memory_info(), "peak_wset", None)
    return rss / 2**20, (peak / 2**20 if peak else None)


class ResourceMonitorCallback(TrainerCallback):
    """
    Records peak RAM / VRAM, wall time per optimizer step and eval runtime,
    writes them to output_path (JSON) and prints a full-run time estimate.
    """

    def __init__(self, output_path: Path, steps_per_epoch: int, epochs: int,
                 eval_steps: int, n_val_full: int, n_val_used: int,
                 hyperparameters: Optional[dict] = None):
        self.output_path = Path(output_path)
        self.hyperparameters = hyperparameters
        self.eval_history: List[dict] = []
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.eval_steps = eval_steps
        self.n_val_full = n_val_full
        self.n_val_used = max(1, n_val_used)
        self.step_times: List[float] = []
        self.interrupted_times: List[float] = []   # step + eval + checkpoint save
        self.eval_runtimes: List[float] = []
        self.peak_rss_mb = 0.0
        self.peak_wset_mb: Optional[float] = None
        self._t0: Optional[float] = None

    def _sample(self):
        rss, peak = _rss_mb()
        self.peak_rss_mb = max(self.peak_rss_mb, rss)
        if peak is not None:
            self.peak_wset_mb = max(self.peak_wset_mb or 0.0, peak)

    # Wall time between consecutive optimizer-step ends (includes data loading,
    # FLAC decode and feature extraction); intervals containing an
    # evaluation or checkpoint save are excluded and accounted separately.
    def on_train_begin(self, args, state, control, **kwargs):
        self._t0 = time.perf_counter()
        self._interrupted = False

    def on_step_end(self, args, state, control, **kwargs):
        now = time.perf_counter()
        if self._t0 is not None:
            if getattr(self, "_interrupted", False):
                self.interrupted_times.append(now - self._t0)
            else:
                self.step_times.append(now - self._t0)
        self._t0 = now
        self._interrupted = False
        self._sample()

    def on_save(self, args, state, control, **kwargs):
        self._interrupted = True

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics and "eval_runtime" in metrics:
            self.eval_runtimes.append(float(metrics["eval_runtime"]))
        if metrics and "eval_loss" in metrics:
            self.eval_history.append({"step": state.global_step, "epoch": state.epoch,
                                      "eval_loss": float(metrics["eval_loss"])})
        self._interrupted = True
        self._sample()
        self.write()

    def on_train_end(self, args, state, control, **kwargs):
        self._sample()
        report = self.write()
        print()
        print("=" * 70)
        print("RESOURCE USAGE")
        print("=" * 70)
        for k, v in report.items():
            print(f"{k:32s}: {v}")

    def report(self) -> dict:
        cuda = torch.cuda.is_available()
        # Skip the first steps (CUDA warm-up) when enough are available
        times = self.step_times[2:] if len(self.step_times) > 4 else self.step_times
        med = statistics.median(times) if times else None
        total_steps = self.steps_per_epoch * self.epochs
        n_evals = math.ceil(total_steps / self.eval_steps) if self.eval_steps else 0
        eval_med = statistics.median(self.eval_runtimes) if self.eval_runtimes else None
        eval_full = eval_med * self.n_val_full / self.n_val_used if eval_med else None
        # checkpoint save (+ logging) overhead per eval, measured directly
        save_overhead = None
        if self.interrupted_times and med is not None:
            save_overhead = max(0.0, statistics.median(self.interrupted_times) - med - (eval_med or 0.0))
        est = None
        if med is not None:
            est = med * total_steps + n_evals * ((eval_full or 0.0) + (save_overhead or 0.0))
        rep = {
            "optimizer_steps_measured": len(self.step_times),
            "median_s_per_step": round(med, 3) if med is not None else None,
            "mean_s_per_step": round(statistics.mean(times), 3) if times else None,
            "peak_rss_mb_process_plus_children": round(self.peak_rss_mb, 1),
            "peak_working_set_mb_main_process": (round(self.peak_wset_mb, 1)
                                                 if self.peak_wset_mb else None),
            "peak_vram_allocated_mb": (round(torch.cuda.max_memory_allocated() / 2**20, 1)
                                       if cuda else None),
            "peak_vram_reserved_mb": (round(torch.cuda.max_memory_reserved() / 2**20, 1)
                                      if cuda else None),
            "eval_runtimes_s": [round(x, 2) for x in self.eval_runtimes],
            "eval_samples_used": self.n_val_used,
            "full_val_samples": self.n_val_full,
            "full_run_steps_per_epoch": self.steps_per_epoch,
            "full_run_epochs": self.epochs,
            "full_run_eval_steps": self.eval_steps,
            "full_run_n_evals": n_evals,
            "full_eval_runtime_est_s": round(eval_full, 1) if eval_full else None,
            "checkpoint_save_overhead_s": (round(save_overhead, 1)
                                           if save_overhead is not None else None),
            "full_run_estimate_hours": round(est / 3600, 2) if est is not None else None,
        }
        if self.hyperparameters is not None:
            rep["hyperparameters"] = self.hyperparameters
        rep["eval_history"] = list(self.eval_history)
        return rep

    def write(self) -> dict:
        rep = self.report()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as fh:
            json.dump(rep, fh, indent=2)
        return rep


# ============================================================
# 6. HELPERS
# ============================================================

def derive_run_name(info: dict, max_steps: int) -> str:
    """whisper-small-mnsc-sg-<15k|500>, plus '-heldout' for held-out builds or
    '-conv' for --split_by_conversation-only builds (so neither ever resumes the
    speaker-split checkpoint of the same size) and '-smoke' for --max_steps runs."""
    n = int(info.get("collection", {}).get("total_collected", 0))
    tag = f"{round(n / 1000)}k" if n >= 1000 else str(n)
    name = f"{RUN_NAME_PREFIX}-{tag}"
    if info.get("heldout_test"):
        name += "-heldout"
    elif info.get("split", {}).get("split_strategy") == "conversation_component":
        name += "-conv"
    if max_steps and max_steps > 0:
        name += "-smoke"
    return name


def compute_eval_steps(n_train: int, eval_steps_arg: int) -> Tuple[int, int]:
    """(optimizer steps per epoch, eval/save interval). 0 -> half an epoch."""
    steps_per_epoch = max(1, math.ceil(n_train / (BATCH_SIZE * GRADIENT_ACCUMULATION)))
    eval_steps = eval_steps_arg if eval_steps_arg > 0 else max(1, steps_per_epoch // 2)
    return steps_per_epoch, eval_steps


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune Whisper-small on preprocessed MNSC data.")
    p.add_argument("--data_name", default=DEFAULT_DATA_NAME,
                   help="Dataset folder under data/ (default part3_15k); "
                        "'legacy' = old data/preprocessed_*.parquet (eager).")
    p.add_argument("--run_name", default=None,
                   help="Experiment name (default whisper-small-mnsc-sg-<size>, "
                        "e.g. whisper-small-mnsc-sg-15k).")
    p.add_argument("--max_steps", type=int, default=-1,
                   help="Stop after N optimizer steps (smoke runs). -1 = full epochs.")
    p.add_argument("--eval_steps", type=int, default=0,
                   help="Evaluate/save every N steps (default 0 = half an epoch).")
    p.add_argument("--max_eval_samples", type=int, default=0,
                   help="Use only the first N validation clips (0 = all).")
    p.add_argument("--num_workers", type=int, default=0,
                   help="DataLoader workers (default 0; Windows-safe).")
    p.add_argument("--epochs", type=float, default=NUM_EPOCHS,
                   help=f"Training epochs (default {NUM_EPOCHS}).")
    p.add_argument("--overwrite_model", action="store_true",
                   help="Allow replacing an existing models/<RUN_NAME>/ when no checkpoint exists.")
    p.add_argument("--learning_rate", type=float, default=LEARNING_RATE,
                   help=f"Peak learning rate of the linear schedule (default {LEARNING_RATE:g}).")
    p.add_argument("--warmup_ratio", type=float, default=DEFAULT_WARMUP_RATIO,
                   help="Fraction of total optimizer steps for linear warmup, in [0, 1) "
                        f"(default {DEFAULT_WARMUP_RATIO:g} = no warmup). Passed as "
                        "TrainingArguments.warmup_steps (transformers 5 has no warmup_ratio; "
                        "a value < 1 there is a ratio).")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED,
                   help=f"Training seed: data order and dropout (default {DEFAULT_SEED}, "
                        "the HF default all existing models used).")
    args = p.parse_args(argv)
    if not args.learning_rate > 0:
        p.error("--learning_rate must be > 0")
    if not 0.0 <= args.warmup_ratio < 1.0:
        p.error("--warmup_ratio must be in [0, 1)")
    return args


def training_args_kwargs(args: argparse.Namespace, output_dir: Path, eval_steps: int) -> dict:
    """Keyword arguments of Seq2SeqTrainingArguments. With default flags this is exactly
    what the script passed before --learning_rate / --warmup_ratio / --seed existed,
    plus seed=42 and warmup_steps=0, which are the HF defaults."""
    return dict(

        output_dir=str(output_dir),

        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,

        # Effective batch = 2 x 4 = 8
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,

        learning_rate=args.learning_rate,

        # A float < 1 is a ratio of total steps in transformers 5; 0 = no warmup
        warmup_steps=args.warmup_ratio if args.warmup_ratio > 0 else 0,

        seed=args.seed,

        num_train_epochs=args.epochs,
        max_steps=args.max_steps,

        # Mixed precision training (reduces VRAM, speeds up training)
        fp16=True,

        # Gradient checkpointing (trades compute for memory)
        gradient_checkpointing=True,

        # Evaluate + save every ~half epoch so early stopping and
        # load_best_model_at_end have several points to choose from
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=eval_steps,
        save_steps=eval_steps,

        logging_steps=min(LOGGING_STEPS, max(1, eval_steps // 2)),

        # Windows-safe: 0 = load in the main process
        dataloader_num_workers=args.num_workers,
        dataloader_persistent_workers=args.num_workers > 0,

        # Skip generate() during training -- evaluate on fixed test set later
        predict_with_generate=False,

        report_to="none",

        # Required when using a custom PyTorch Dataset
        remove_unused_columns=False,

        # Keep only the last checkpoint to save disk space
        save_total_limit=1,

        # Load the best checkpoint (lowest eval loss) at the end
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )


def build_training_args(args: argparse.Namespace, output_dir: Path,
                        eval_steps: int) -> Seq2SeqTrainingArguments:
    return Seq2SeqTrainingArguments(**training_args_kwargs(args, output_dir, eval_steps))


def total_training_steps(steps_per_epoch: int, epochs: float, max_steps: int) -> int:
    """Optimizer steps the LR schedule spans (what HF uses for warmup and decay)."""
    return max_steps if max_steps > 0 else math.ceil(epochs * steps_per_epoch)


def hyperparameters_record(args: argparse.Namespace, training_args, total_steps: int,
                           run_name: str) -> dict:
    """Values actually used, read back from the TrainingArguments where possible."""
    return {
        "run_name": run_name,
        "data_name": args.data_name,
        "learning_rate": training_args.learning_rate,
        "warmup_ratio": args.warmup_ratio,
        "warmup_steps_arg": training_args.warmup_steps,
        "warmup_steps_resolved": training_args.get_warmup_steps(total_steps),
        "seed": training_args.seed,
        "epochs": args.epochs,
        "max_steps": training_args.max_steps,
        "total_steps": total_steps,
        "batch_size": training_args.per_device_train_batch_size,
        "gradient_accumulation": training_args.gradient_accumulation_steps,
        "lr_scheduler_type": str(getattr(training_args.lr_scheduler_type, "value",
                                         training_args.lr_scheduler_type)),
        "weight_decay": training_args.weight_decay,
        "optim": str(getattr(training_args.optim, "value", training_args.optim)),
    }


RESUME_KEYS = ("learning_rate", "warmup_ratio", "seed", "data_name")


def check_resume_hyperparameters(path: Path, record: dict) -> Optional[str]:
    """Error message if a previous run of this checkpoint folder (resource_usage.json)
    used different learning_rate / warmup_ratio / seed / data_name, else None.
    Files without a 'hyperparameters' block (older runs) are not checked."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            old = json.load(fh).get("hyperparameters")
    except (OSError, ValueError):
        return None
    if not isinstance(old, dict):
        return None
    diffs = [f"{k}: checkpoint {old.get(k)!r} vs now {record.get(k)!r}"
             for k in RESUME_KEYS if k in old and old.get(k) != record.get(k)]
    if not diffs:
        return None
    return (f"ERROR: the checkpoint in {path.parent} was trained with different settings "
            f"({'; '.join(diffs)}). Resuming would mix them. Use the same flags, or another "
            "--run_name.")


# ============================================================
# 7. MAIN
# ============================================================

def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    # --------------------------------------------------------
    # Load preprocessed data (lazy index; legacy = eager)
    # --------------------------------------------------------
    print("=" * 70)
    print("LOADING PREPROCESSED DATA")
    print("=" * 70)

    if args.data_name == "legacy":
        train_records, val_records, info = load_preprocessed_data()
        default_run = LEGACY_RUN_NAME
    else:
        train_records, val_records, info = load_preprocessed_index(args.data_name)
        default_run = derive_run_name(info, args.max_steps)

    run_name = args.run_name or default_run
    output_dir = PROJECT_ROOT / "checkpoints" / run_name   # resumable training state
    final_model_dir = PROJECT_ROOT / "models" / run_name   # best model, share this

    n_val_full = len(val_records)
    if args.max_eval_samples > 0:
        val_records = val_records[:args.max_eval_samples]

    print(f"Dataset        : {args.data_name}")
    print(f"Run name       : {run_name}")
    print(f"Train samples  : {len(train_records):,}")
    print(f"Val samples    : {len(val_records):,} (of {n_val_full:,})")
    split_strategy = info.get("split", {}).get("split_strategy", "unknown")
    print(f"Split strategy : {split_strategy}")
    print()

    # --------------------------------------------------------
    # Protect existing final models (e.g. the 4k baseline)
    # --------------------------------------------------------
    has_checkpoint = (
        output_dir.is_dir()
        and any(f.startswith("checkpoint-") for f in os.listdir(output_dir))
    )
    if (final_model_dir / "model.safetensors").exists() and not has_checkpoint \
            and not args.overwrite_model:
        print(f"ERROR: {final_model_dir} already contains a trained model and there is no "
              f"checkpoint in {output_dir} to resume.\n"
              "Choose another --run_name, or pass --overwrite_model to replace it.")
        return 2

    # --------------------------------------------------------
    # --------------------------------------------------------
    print("=" * 70)
    print("LOADING WHISPER PROCESSOR")
    print("=" * 70)

    processor = WhisperProcessor.from_pretrained(
        "openai/whisper-small",
        language="en",
        task="transcribe"
    )
    print("Processor loaded.")

    # --------------------------------------------------------
    # First sample
    # --------------------------------------------------------
    print()
    print("=" * 70)
    print("FIRST TRAINING SAMPLE")
    print("=" * 70)

    first = train_records[0]
    first_audio = first.get("audio")
    if first_audio is None:
        first_audio = read_clip_audio(first)
    print(f"Speaker ID   : {first['speaker_id']}")
    print(f"Sample rate  : {first['sampling_rate']} Hz")
    print(f"Audio samples: {len(first_audio)}")
    print(f"Duration     : {len(first_audio) / first['sampling_rate']:.2f} s")
    print(f"Transcript   : {first['text']}")
    del first_audio

    train_dataset = WhisperMNSCDataset(train_records, processor)
    val_dataset = WhisperMNSCDataset(val_records, processor)

    print()
    print("=" * 70)
    print("PYTORCH DATASETS")
    print("=" * 70)
    print(f"Training   : {len(train_dataset):,}")
    print(f"Validation : {len(val_dataset):,}")

    # --------------------------------------------------------
    # --------------------------------------------------------
    print()
    print("=" * 70)
    print("LOADING WHISPER-SMALL")
    print("=" * 70)

    model = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-small"
    )

    # Disable forced decoder IDs -- we set language/task explicitly
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.generation_config.language = "en"
    model.generation_config.task = "transcribe"

    print("Whisper-small loaded.")

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id
    )
    print()
    print("Data collator created.")

    # --------------------------------------------------------
    # --------------------------------------------------------
    print()
    print("=" * 70)
    print("TRAINING ARGUMENTS")
    print("=" * 70)

    steps_per_epoch, eval_steps = compute_eval_steps(len(train_dataset), args.eval_steps)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = build_training_args(args, output_dir, eval_steps)
    total_steps = total_training_steps(steps_per_epoch, args.epochs, args.max_steps)
    hparams = hyperparameters_record(args, training_args, total_steps, run_name)

    print(f"Output dir      : {output_dir}")
    print(f"Batch size      : {training_args.per_device_train_batch_size} "
          f"(effective {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps})")
    print(f"Learning rate   : {training_args.learning_rate:g}")
    print(f"LR schedule     : {hparams['lr_scheduler_type']}")
    print(f"Warmup          : {args.warmup_ratio:g} of {total_steps} steps = "
          f"{hparams['warmup_steps_resolved']} steps")
    print(f"Seed            : {training_args.seed}")
    print(f"Epochs          : {args.epochs}")
    print(f"Steps per epoch : {steps_per_epoch}")
    print(f"Eval/save every : {eval_steps} steps")
    print(f"Max steps       : {args.max_steps}")
    print(f"Num workers     : {args.num_workers}")

    if has_checkpoint:
        mismatch = check_resume_hyperparameters(output_dir / "resource_usage.json", hparams)
        if mismatch:
            print(mismatch)
            return 2

    # --------------------------------------------------------
    # --------------------------------------------------------
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    trainer.add_callback(EarlyStoppingCallbackWithPatience(patience=2))
    trainer.add_callback(ResourceMonitorCallback(
        output_path=output_dir / "resource_usage.json",
        steps_per_epoch=steps_per_epoch,
        epochs=math.ceil(args.epochs),
        # Estimate for the FULL run: smoke runs use the default half-epoch cadence
        eval_steps=(args.eval_steps if args.eval_steps > 0 and args.max_steps <= 0
                    else compute_eval_steps(len(train_dataset), 0)[1]),
        n_val_full=n_val_full,
        n_val_used=len(val_dataset),
        hyperparameters=hparams,
    ))

    print()
    print("=" * 70)
    print("TRAINER READY")
    print("=" * 70)

    # --------------------------------------------------------
    # --------------------------------------------------------
    print()
    print("=" * 70)
    print("STARTING FINE-TUNING")
    print("=" * 70)

    if has_checkpoint:
        print(f"Resuming from checkpoint in {output_dir}")
        trainer.train(resume_from_checkpoint=True)
    else:
        print("No checkpoint found. Starting fresh training.")
        trainer.train()

    # --------------------------------------------------------
    # --------------------------------------------------------
    print()
    print("=" * 70)
    print("SAVING FINAL MODEL")
    print("=" * 70)

    final_model_dir.mkdir(parents=True, exist_ok=True)

    trainer.save_model(str(final_model_dir))
    processor.save_pretrained(str(final_model_dir))

    print()
    print("Training complete.")
    print(f"Model saved to: {final_model_dir}")
    print()
    print("Next step: evaluate on the fixed 4,000-sample test set")
    print("(MNSC PART1 3000 + PART3 1000) and compare with the 4k baseline:")
    print(f"  python scripts/evaluate_whisper_speaker_split.py --run_name {run_name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
