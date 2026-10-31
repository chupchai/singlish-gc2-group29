"""
Evaluation script for speaker-split fine-tuned Whisper model.

Purpose: evaluate the speaker-split fine-tuned Whisper model for comparison with previous runs.
Model: whisper-small-mnsc-speaker-split-final (trained with speaker-aware split)
Test set: 4000 samples (3000 MNSC PART1 + 1000 MNSC PART3), same fixed manifest as all prior experiments
Output: per-sample CSV and summary CSV
"""

import io
import os
import re
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import soundfile as sf
import torch
from huggingface_hub import snapshot_download

from datasets import Dataset
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from jiwer import wer, cer

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent              # scripts/ holds code only

# ============================================================
# CONFIGURATION
# ============================================================

# Path to the fine-tuned model (output from train_whisper_speaker_split.py)
# Must match RUN_NAME in train_whisper_speaker_split.py
RUN_NAME = "whisper-small-mnsc-sg-4k"
MODEL_PATH = PROJECT_ROOT / "models" / RUN_NAME

# Manifest (input) lives in data/; evaluation outputs go to results/
MANIFEST_PATH = PROJECT_ROOT / "data" / "mnsc_extended_evaluation_manifest.csv"
RESULTS_DIR   = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH   = RESULTS_DIR / "whisper_small_mnsc_extended_evaluation.csv"
SUMMARY_PATH  = RESULTS_DIR / "whisper_small_mnsc_extended_summary.csv"

# Where the MNSC test parquet files are located
HF_DATASET_ID = "AudioLLMs/Multitask-National-Speech-Corpus-v1-extend"


# ============================================================
# TRANSCRIPT NORMALIZATION
# ============================================================

def normalize_transcript(text):
    if not text:
        return ""

    # Remove speaker labels
    text = re.sub(
        r"<Speaker\d+>:\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    # Remove square-bracket annotation markers
    # while preserving their contents
    while re.search(r"\[[^\[\]]+\]", text):
        text = re.sub(
            r"\[([^\[\]]+)\]",
            r"\1",
            text
        )

    # Remove curly-bracket markers
    # while preserving their contents
    text = re.sub(
        r"\{([^{}\[\]]+)\}",
        r"\1",
        text
    )

    # Remove parentheses while preserving
    # spoken content
    text = re.sub(
        r"\(([^()]*)\)",
        r"\1",
        text
    )

    # Remove hash markers while preserving content
    text = re.sub(
        r"#([^#]+)#",
        r"\1",
        text
    )

    # Normalize whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    # Lowercase
    text = text.lower()

    return text


def normalize_for_evaluation(text):
    """
    Additional normalization used specifically for WER/CER.

    Keeps the same transcript content but removes punctuation,
    except apostrophes.
    """

    text = normalize_transcript(text)

    # Remove punctuation except apostrophes
    text = re.sub(
        r"[^\w\s']",
        "",
        text
    )

    # Normalize whitespace again
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ============================================================
# LOAD MANIFEST
# ============================================================

print("=" * 70)
print("LOADING FIXED EVALUATION MANIFEST")
print("=" * 70)

manifest = pd.read_csv(MANIFEST_PATH)

print("Rows:", len(manifest))
print()
print(manifest["config"].value_counts())

# Important sanity checks
assert len(manifest) == 4000
assert manifest["sample_id"].duplicated().sum() == 0
assert manifest["config"].isin(
    ["ASR-PART1-Test", "ASR-PART3-Test"]
).all()

print("\nManifest verification: PASS")


# ============================================================
# LOAD TEST PARQUET DATA
# ============================================================

def load_parquet_rows(parquet_path):
    """
    Load all rows from a MNSC parquet file.
    """

    table = pq.read_table(parquet_path)

    rows = table.to_pylist()

    return rows


print("\n" + "=" * 70)
print("LOADING MNSC TEST DATA")
print("=" * 70)

# Downloads only the two test splits on first run; afterwards reuses the
# local HuggingFace cache (~/.cache/huggingface/hub) without re-downloading.
print("Resolving MNSC test splits (downloads on first run, then cached)...")
dataset_snapshot_dir = Path(
    snapshot_download(
        repo_id=HF_DATASET_ID,
        repo_type="dataset",
        allow_patterns=["ASR-PART1-Test/*", "ASR-PART3-Test/*"],
    )
)
print("Snapshot dir:", dataset_snapshot_dir)

parquet_files = {
    "ASR-PART1-Test": sorted(
        (dataset_snapshot_dir / "ASR-PART1-Test").glob("*.parquet")
    ),
    "ASR-PART3-Test": sorted(
        (dataset_snapshot_dir / "ASR-PART3-Test").glob("*.parquet")
    ),
}


# ============================================================
# BUILD LOOKUP
# ============================================================

print("\nBuilding row lookup...")

lookup = {}

for config, paths in parquet_files.items():

    global_row_index = 0

    for parquet_path in paths:

        print(
            f"Reading {config}: "
            f"{os.path.basename(parquet_path)}"
        )

        rows = load_parquet_rows(parquet_path)

        print("Rows:", len(rows))

        for row in rows:

            lookup[(config, global_row_index)] = row

            global_row_index += 1


print("\nTotal lookup rows:", len(lookup))


# ============================================================
# VERIFY MANIFEST AGAINST PARQUET
# ============================================================

print("\nVerifying manifest rows exist...")

for _, manifest_row in manifest.iterrows():

    key = (
        manifest_row["config"],
        int(manifest_row["row_index"])
    )

    assert key in lookup, (
        f"Missing row: {key}"
    )

print("Manifest parquet verification: PASS")


# ============================================================
# LOAD MODEL
# ============================================================

print("\n" + "=" * 70)
print("LOADING FINE-TUNED MODEL")
print("=" * 70)

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", device)

processor = WhisperProcessor.from_pretrained(
    str(MODEL_PATH)
)

model = WhisperForConditionalGeneration.from_pretrained(
    str(MODEL_PATH)
)

# fp16 on GPU roughly halves memory and speeds up decoding.
# Cast after loading so it works on every transformers version.
model_dtype = torch.float16 if device == "cuda" else torch.float32
model.to(device, dtype=model_dtype)
model.eval()

# Samples decoded together per generate() call.
# 8 fits comfortably in 8 GB VRAM for whisper-small in fp16;
# lower this if you hit CUDA out-of-memory errors.
EVAL_BATCH_SIZE = 8

# Force English transcription
model.generation_config.language = "en"
model.generation_config.task = "transcribe"

print("Model loaded successfully.")


# ============================================================
# EVALUATION
# ============================================================

results = []

print("\n" + "=" * 70)
print("STARTING EVALUATION")
print("=" * 70)
print(f"Batch size: {EVAL_BATCH_SIZE} | dtype: {model_dtype}")

manifest_rows = manifest.to_dict("records")
total = len(manifest_rows)
next_progress = 50

for start in range(0, total, EVAL_BATCH_SIZE):

    batch_rows = manifest_rows[start:start + EVAL_BATCH_SIZE]

    # --------------------------------------------------------
    # Get audio + reference transcripts for the batch
    # --------------------------------------------------------

    audios = []
    references = []

    for manifest_row in batch_rows:

        row = lookup[(manifest_row["config"], int(manifest_row["row_index"]))]

        audio, sample_rate = sf.read(
            io.BytesIO(row["context"]["bytes"]),
            dtype="float32"
        )

        audios.append(audio)
        references.append(normalize_for_evaluation(row["answer"]))

    # --------------------------------------------------------
    # Whisper input (pads/trims every clip to 30 s, so the
    # batch is always a uniform tensor)
    # --------------------------------------------------------

    inputs = processor(
        audios,
        sampling_rate=sample_rate,
        return_tensors="pt"
    )

    # Match the model's dtype (fp16 on GPU) to avoid a dtype mismatch
    input_features = inputs.input_features.to(device, dtype=model_dtype)

    # --------------------------------------------------------
    # Generate predictions for the whole batch
    # --------------------------------------------------------

    with torch.no_grad():

        predicted_ids = model.generate(
            input_features,
            language="en",
            task="transcribe"
        )

    predictions = processor.batch_decode(
        predicted_ids,
        skip_special_tokens=True
    )

    # --------------------------------------------------------
    # Per-sample WER / CER + store results
    # --------------------------------------------------------

    for manifest_row, reference, prediction in zip(batch_rows, references, predictions):

        prediction_eval = normalize_for_evaluation(prediction)

        results.append({
            "sample_id": manifest_row["sample_id"],
            "dataset": manifest_row["dataset"],
            "config": manifest_row["config"],
            "row_index": int(manifest_row["row_index"]),
            "reference": reference,
            "prediction": prediction_eval,
            "WER": wer(reference, prediction_eval),
            "CER": cer(reference, prediction_eval)
        })

    # Progress
    done = start + len(batch_rows)
    if done >= next_progress or done == total:
        print(f"Processed {done}/{total}")
        next_progress = (done // 50 + 1) * 50



# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nEvaluation results saved to:")
print(OUTPUT_PATH)


# ============================================================
# CALCULATE SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("EVALUATION SUMMARY")
print("=" * 70)


def corpus_metric(group_df, metric):

    references = group_df["reference"].tolist()
    predictions = group_df["prediction"].tolist()

    if metric == "WER":
        return wer(
            references,
            predictions
        )

    elif metric == "CER":
        return cer(
            references,
            predictions
        )


summary = []


# ------------------------------------------------------------
# Overall
# ------------------------------------------------------------

summary.append({
    "group": "Overall",
    "samples": len(results_df),
    "mean_WER": results_df["WER"].mean(),
    "mean_CER": results_df["CER"].mean(),
    "corpus_WER": corpus_metric(results_df, "WER"),
    "corpus_CER": corpus_metric(results_df, "CER")
})


# ------------------------------------------------------------
# Per configuration
# ------------------------------------------------------------

for config in [
    "ASR-PART1-Test",
    "ASR-PART3-Test"
]:

    group = results_df[
        results_df["config"] == config
    ]

    summary.append({
        "group": config,
        "samples": len(group),
        "mean_WER": group["WER"].mean(),
        "mean_CER": group["CER"].mean(),
        "corpus_WER": corpus_metric(group, "WER"),
        "corpus_CER": corpus_metric(group, "CER")
    })


summary_df = pd.DataFrame(summary)

print(
    summary_df.to_string(
        index=False
    )
)

summary_df.to_csv(
    SUMMARY_PATH,
    index=False
)

print("\nSummary saved to:")
print(SUMMARY_PATH)