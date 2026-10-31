"""
verify_mnsc_dataset.py

Sanity-check a dataset built by preprocess_mnsc_data.py.

Usage (from the project root):
    python scripts/verify_mnsc_dataset.py --data_name part3_15k

Checks: listed shard files exist; row counts match the split info; train/val
speaker sets are disjoint and equal to the JSON lists; per-speaker clip count
<= max_clips_per_speaker; every source_config is a *-Train config; a random
sample of clips decodes to 16 kHz mono float32 of the stored length.
Prints PASS and exits 0, or prints each failure and exits 1.
"""

import argparse
import random
import sys
from collections import Counter
from pathlib import Path
from typing import List, Optional

import numpy as np
import pyarrow.parquet as pq

from preprocess_mnsc_data import (
    DATA_DIR, INFO_NAME, TARGET_SR, dataset_dir, load_preprocessed_index,
    read_clip_audio, speaker_distribution,
)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Verify a preprocessed MNSC dataset.")
    p.add_argument("--data_name", default="part3_15k")
    p.add_argument("--data_root", default=str(DATA_DIR))
    p.add_argument("--n_decode", type=int, default=20,
                   help="Number of random clips to decode (default 20).")
    args = p.parse_args(argv)

    out_dir = dataset_dir(args.data_name, Path(args.data_root))
    failures: List[str] = []

    train, val, info = load_preprocessed_index(args.data_name, Path(args.data_root))
    split = info["split"]
    layout = info["layout"]

    for s in ("train", "val"):
        for name in layout[f"{s}_shards"]:
            if not (out_dir / s / name).exists():
                failures.append(f"missing shard {s}/{name}")
    if len(train) != split["train_samples"]:
        failures.append(f"train rows {len(train)} != split.train_samples {split['train_samples']}")
    if len(val) != split["val_samples"]:
        failures.append(f"val rows {len(val)} != split.val_samples {split['val_samples']}")

    tr_spk = {i["speaker_id"] for i in train} - {"unknown"}
    va_spk = {i["speaker_id"] for i in val}
    if tr_spk & va_spk:
        failures.append(f"{len(tr_spk & va_spk)} speakers in both train and val")
    if split.get("split_strategy") in ("speaker_level", "conversation_component"):
        if tr_spk != set(split["train_speakers"]):
            failures.append("train speakers differ from split.train_speakers")
        if va_spk != set(split["val_speakers"]):
            failures.append("val speakers differ from split.val_speakers")

    cap = int(info.get("max_clips_per_speaker") or 0)
    counts = Counter(i["speaker_id"] for i in train + val)
    if cap and counts and max(counts.values()) > cap:
        failures.append(f"max clips per speaker {max(counts.values())} > cap {cap}")

    configs = Counter()
    for s in ("train", "val"):
        for name in layout[f"{s}_shards"]:
            col = pq.read_table(str(out_dir / s / name), columns=["source_config"]).column(0)
            configs.update(col.to_pylist())
    bad = [c for c in configs if not c.endswith("-Train")]
    if bad:
        failures.append(f"non-Train source configs present: {bad}")

    rng = random.Random(0)
    sample = rng.sample(train + val, min(args.n_decode, len(train) + len(val)))
    cache: dict = {}
    for item in sample:
        audio = read_clip_audio(item, cache)
        if item["sampling_rate"] != TARGET_SR:
            failures.append(f"{item['clip_id']}: sampling_rate {item['sampling_rate']}")
        if audio.ndim != 1 or audio.dtype != np.float32:
            failures.append(f"{item['clip_id']}: shape {audio.shape} dtype {audio.dtype}")
        if len(audio) != item["audio_length"]:
            failures.append(f"{item['clip_id']}: len {len(audio)} != audio_length {item['audio_length']}")
        if abs(item["duration_sec"] - len(audio) / TARGET_SR) >= 1e-3:
            failures.append(f"{item['clip_id']}: duration mismatch")

    dist = speaker_distribution(train + val, cap)
    hours_tr = sum(i["duration_sec"] for i in train) / 3600
    hours_va = sum(i["duration_sec"] for i in val) / 3600
    print(f"Dataset      : {out_dir}")
    print(f"Source configs: {dict(configs)}")
    print(f"Train        : {len(train):,} clips, {hours_tr:.2f} h, {len(tr_spk):,} speakers")
    print(f"Val          : {len(val):,} clips, {hours_va:.2f} h, {len(va_spk):,} speakers")
    print(f"Clips/speaker: n={dist.get('n_speakers')} min={dist.get('min')} "
          f"median={dist.get('median')} mean={dist.get('mean')} p90={dist.get('p90')} "
          f"max={dist.get('max')} at_cap({cap})={dist.get('n_at_cap')}")
    print(f"Decoded      : {len(sample)} random clips")
    print(f"Training speakers JSON: {out_dir / INFO_NAME} (key split.train_speakers)")

    if failures:
        for f in failures:
            print(f"FAIL: {f}")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
