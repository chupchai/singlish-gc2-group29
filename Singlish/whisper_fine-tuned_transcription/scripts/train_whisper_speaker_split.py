"""
train_whisper_speaker_split.py

Fine-tune Whisper-small on MNSC Extended data using preprocessed data
from preprocess_mnsc_data.py.

Usage:
    1. Run preprocess_mnsc_data.py first to generate the parquet files.
    2. python train_whisper_speaker_split.py
"""

# ============================================================
# 1. IMPORTS
# ============================================================

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Union

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
from preprocess_mnsc_data import load_preprocessed_data


# ============================================================
# 2. CONFIGURATION
# ============================================================

# Output directories (resolved relative to this script's location
# so the script works regardless of the shell's cwd)
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent              # scripts/ holds code only

# One descriptive name per experiment: model_data_size_split_key-hyperparams.
# Change this for each new experiment so runs never resume each other's
# checkpoints. Keep it in sync with RUN_NAME in evaluate_whisper_speaker_split.py.
RUN_NAME = "whisper-small-mnsc-sg-4k"

OUTPUT_DIR      = PROJECT_ROOT / "checkpoints" / RUN_NAME  # resumable training state
FINAL_MODEL_DIR = PROJECT_ROOT / "models" / RUN_NAME       # best model, share this

# Training hyperparameters (optimized for RTX 3070 Laptop 8GB VRAM)
BATCH_SIZE = 2              # Double batch size for better GPU utilization
GRADIENT_ACCUMULATION = 4   # Effective batch = 2 × 4 = 8 (same as before)
LEARNING_RATE = 2e-5        # Higher LR for faster convergence
NUM_EPOCHS = 3              # Keep same
LOGGING_STEPS = 25


# ============================================================
# 3. LOAD PREPROCESSED DATA
# ============================================================

print("=" * 70)
print("LOADING PREPROCESSED DATA")
print("=" * 70)

train_records, val_records, info = load_preprocessed_data()

print(f"Train samples  : {len(train_records):,}")
print(f"Val samples    : {len(val_records):,}")
split_strategy = info.get("split", {}).get("split_strategy", "unknown")
print(f"Split strategy : {split_strategy}")
print()


# ============================================================
# 4. LOAD WHISPER PROCESSOR
# ============================================================

print()
print("=" * 70)
print("LOADING WHISPER PROCESSOR")
print("=" * 70)

processor = WhisperProcessor.from_pretrained(
    "openai/whisper-small",
    language="en",
    task="transcribe"
)

print("Processor loaded.")


# ============================================================
# 5. CHECK FIRST SAMPLE
# ============================================================

print()
print("=" * 70)
print("FIRST TRAINING SAMPLE")
print("=" * 70)

first = train_records[0]
print(f"Speaker ID   : {first['speaker_id']}")
print(f"Sample rate  : {first['sampling_rate']} Hz")
print(f"Audio samples: {len(first['audio'])}")
print(
    f"Duration     : "
    f"{len(first['audio']) / first['sampling_rate']:.2f} s"
)
print(f"Transcript   : {first['text']}")


# ============================================================
# 6. PYTORCH DATASET
# ============================================================

class WhisperMNSCDataset(Dataset):
    """
    PyTorch Dataset that wraps the in-memory records list
    and performs Whisper feature extraction on-the-fly.
    """

    def __init__(
        self,
        records: List[dict],
        processor: Any
    ):
        self.records = records
        self.processor = processor

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]

        # Convert audio waveform to Whisper log-Mel features
        input_features = self.processor.feature_extractor(
            record["audio"],
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


train_dataset = WhisperMNSCDataset(train_records, processor)
val_dataset = WhisperMNSCDataset(val_records, processor)

print()
print("=" * 70)
print("PYTORCH DATASETS")
print("=" * 70)
print(f"Training   : {len(train_dataset):,}")
print(f"Validation : {len(val_dataset):,}")


# ============================================================
# 7. LOAD WHISPER-SMALL
# ============================================================

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


# ============================================================
# 8. DATA COLLATOR
# ============================================================

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Pads variable-length transcripts so samples can be
    batched together. Identical to the collator used in
    previous MNSC training scripts.
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


data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id
)

print()
print("Data collator created.")


# ============================================================
# 9. TRAINING ARGUMENTS
# ============================================================

print()
print("=" * 70)
print("TRAINING ARGUMENTS")
print("=" * 70)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

training_args = Seq2SeqTrainingArguments(

    output_dir=str(OUTPUT_DIR),

    # RTX 3070 Laptop 8 GB: keep batch_size=1 to fit in VRAM
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,

    # Effective batch = 1 x 8 = 8
    gradient_accumulation_steps=GRADIENT_ACCUMULATION,

    learning_rate=LEARNING_RATE,

    num_train_epochs=NUM_EPOCHS,

    # Mixed precision training (reduces VRAM, speeds up training)
    fp16=True,

    # Gradient checkpointing (trades compute for memory)
    gradient_checkpointing=True,

    eval_strategy="epoch",
    save_strategy="epoch",

    logging_steps=LOGGING_STEPS,

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

print(f"Output dir      : {OUTPUT_DIR}")
print(f"Batch size      : {BATCH_SIZE} (effective {BATCH_SIZE * GRADIENT_ACCUMULATION})")
print(f"Learning rate   : {LEARNING_RATE}")
print(f"Epochs          : {NUM_EPOCHS}")


# ============================================================
# 10. TRAINER
# ============================================================

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
)

# Add early stopping callback
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

early_stopping = EarlyStoppingCallbackWithPatience(patience=2)

trainer.add_callback(early_stopping)

print()
print("=" * 70)
print("TRAINER READY")
print("=" * 70)


# ============================================================
# 11. FINE-TUNE
# ============================================================

print()
print("=" * 70)
print("STARTING FINE-TUNING")
print("=" * 70)

# Check if there's an existing checkpoint to resume from
import os
has_checkpoint = (
    os.path.isdir(OUTPUT_DIR) and 
    any(f.startswith("checkpoint-") for f in os.listdir(OUTPUT_DIR))
    if os.path.isdir(OUTPUT_DIR) else False
)

if has_checkpoint:
    print(f"Resuming from checkpoint in {OUTPUT_DIR}")
    trainer.train(resume_from_checkpoint=True)
else:
    print("No checkpoint found. Starting fresh training.")
    trainer.train()


# ============================================================
# 12. SAVE FINAL MODEL
# ============================================================

print()
print("=" * 70)
print("SAVING FINAL MODEL")
print("=" * 70)

FINAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)

trainer.save_model(str(FINAL_MODEL_DIR))
processor.save_pretrained(str(FINAL_MODEL_DIR))

print()
print("Training complete.")
print(f"Model saved to: {FINAL_MODEL_DIR}")
print()
print("Next step: run the evaluation script on the fixed 4,000-sample")
print("test set (MNSC PART1 3000 + PART3 1000) to compare WER/CER")
print("against the previous baseline and 5000-sample fine-tuned models.")