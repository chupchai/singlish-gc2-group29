# 1. IMPORTS
import os
import re
import pandas as pd
from datasets import load_dataset
from pathlib import Path

# 2. CONFIGURATION
# Target configs and max rows
CONFIGS = {
    "ASR-PART1-Test": 3000,
    "ASR-PART3-Test": 1000,
}

# Output path (resolved relative to this script's location)
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent              # scripts/ holds code only
DATA_DIR     = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_CSV   = DATA_DIR / "mnsc_extended_evaluation_manifest.csv"


# 3. HELPER FUNCTIONS
def normalize_reference(text):
    """
    Normalize text for evaluation.
    """
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    
    text = str(text)
    
    # 1. Remove speaker labels
    text = re.sub(r"<Speaker\d+>\s*:", " ", text, flags=re.IGNORECASE)
    
    # 2. Unwrap square brackets iteratively
    while re.search(r"\[[^\[\]]+\]", text):
        text = re.sub(r"\[([^\[\]]+)\]", r"\1", text)
    
    # 3. Unwrap curly braces
    text = re.sub(r"\{([^{}\[\]]+)\}", r"\1", text)
    
    # 4. Remove parenthetical annotations
    text = re.sub(r"\([^()]+\)", "", text)
    
    # 5. Unwrap hash markers
    text = re.sub(r"#([^#]+)#", r"\1", text)
    
    # 6. Lowercase
    text = text.lower()
    
    # 7. Remove all punctuation except apostrophes
    text = re.sub(r"[^\w\s']", " ", text)
    
    # 8. Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    
    return text


def count_labeled_speakers(answer_raw, row_dict):
    """
    Count distinct speaker labels in the answer.
    Uses two methods and returns the maximum.
    """
    count_from_answer = 0
    count_from_columns = 0
    
    # Method 1: Count <Speaker\d+> patterns in answer_raw
    if answer_raw:
        matches = re.findall(r"<Speaker(\d+)>", answer_raw, flags=re.IGNORECASE)
        if matches:
            count_from_answer = len(set(int(m) for m in matches))
    
    # Method 2: Check top-level columns speaker1, speaker2
    for col in ["speaker1", "speaker2"]:
        if col in row_dict and row_dict[col] is not None:
            count_from_columns += 1
    
    return max(count_from_answer, count_from_columns)


def has_singlish(text):
    """
    Check if normalized text contains Singlish markers.
    """
    if not text:
        return False
    
    # Singlish particles and Singaporean abbreviations
    pattern = r"\b(lah|leh|loh|meh|sia|lor|hor|hdb|pap|ndp|mrt|pwd|nea|mom|moe)\b"
    return bool(re.search(pattern, text, flags=re.IGNORECASE))


# 4. LOAD DATASETS
print("=" * 70)
print("LOADING DATASETS")
print("=" * 70)

dataset_data = {}

for config_name, max_rows in CONFIGS.items():
    print(f"\n--- Loading {config_name} ---")
    
    # Load dataset using datasets library
    print(f"Loading: AudioLLMs/Multitask-National-Speech-Corpus-v1-extend, config={config_name}")
    
    dataset = load_dataset(
        "AudioLLMs/Multitask-National-Speech-Corpus-v1-extend",
        config_name,
        split="train"
    )
    
    print(f"Dataset loaded: {len(dataset):,} rows")
    print(f"Features: {list(dataset.features.keys())}")
    
    # Convert to pandas to avoid audio decoding issues
    # Only keep the text/metadata columns we need
    df_temp = dataset.to_pandas()
    print(f"DataFrame columns: {list(df_temp.columns)}")
    
    # Take only the required rows
    df_subset = df_temp.head(max_rows)
    
    dataset_data[config_name] = {
        "df": df_subset,
        "total_rows": len(df_temp),
        "taken_rows": len(df_subset),
    }
    
    print(f"Taking {len(df_subset):,} rows for manifest")


# 5. COLLECT RECORDS
print("\n" + "=" * 70)
print("COLLECTING RECORDS")
print("=" * 70)

records = []

for config_name, data in dataset_data.items():
    df = data["df"]
    max_rows = data["taken_rows"]
    
    print(f"\n{config_name}: Processing {max_rows:,} rows")
    
    for idx, row_dict in df.iterrows():
        # Get reference_raw from 'answer' column
        reference_raw = row_dict.get("answer", "")
        
        # Normalize reference
        reference_eval = normalize_reference(reference_raw)
        
        # Count labeled speakers
        num_labeled_speakers = count_labeled_speakers(reference_raw, row_dict)
        
        # Check Singlish marker
        has_singlish_marker = has_singlish(reference_eval)
        
        # Build sample_id
        sample_id = f"{config_name}_{idx:06d}"
        
        record = {
            "sample_id": sample_id,
            "dataset": "MNSC-Extended",
            "config": config_name,
            "row_index": idx,
            "reference_raw": reference_raw,
            "reference_eval": reference_eval,
            "num_labeled_speakers": num_labeled_speakers,
            "has_singlish_marker": has_singlish_marker,
        }
        
        records.append(record)

print(f"\nTotal records collected: {len(records):,}")


# 6. BUILD DATAFRAME
print("\n" + "=" * 70)
print("BUILDING DATAFRAME")
print("=" * 70)

df = pd.DataFrame(records)

# Verify required columns
required_columns = [
    "sample_id",
    "dataset",
    "config",
    "row_index",
    "reference_raw",
    "reference_eval",
    "num_labeled_speakers",
    "has_singlish_marker",
]

missing = [col for col in required_columns if col not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

print(f"DataFrame shape: {df.shape}")
print(f"Columns: {list(df.columns)}")


# 7. SAVE CSV
print("\n" + "=" * 70)
print("SAVING CSV")
print("=" * 70)

df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved to: {OUTPUT_CSV}")


# 8. SUMMARY STATISTICS
print("\n" + "=" * 70)
print("SUMMARY STATISTICS")
print("=" * 70)

print(f"\nTotal rows: {len(df):,}")

# Rows by config
print("\nRows by config:")
for config in CONFIGS.keys():
    count = len(df[df["config"] == config])
    print(f"  {config}: {count:,}")

# Singlish rate
singlish_count = df["has_singlish_marker"].sum()
singlish_rate = singlish_count / len(df) * 100
print(f"\nSinglish markers found: {singlish_count:,} ({singlish_rate:.2f}%)")

# Multi-speaker rate
multi_speaker_count = (df["num_labeled_speakers"] > 1).sum()
multi_speaker_rate = multi_speaker_count / len(df) * 100
print(f"Multi-speaker samples: {multi_speaker_count:,} ({multi_speaker_rate:.2f}%)")

# Average speakers
avg_speakers = df["num_labeled_speakers"].mean()
print(f"Average speakers per sample: {avg_speakers:.2f}")

# First 5 rows
print("\nFirst 5 rows:")
print(df.head().to_string())

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)