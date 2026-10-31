# Performance test — 24 October 2026

The frozen **24 Oct ML (SVM65)** and **Ensemble (SVM65 + Whisper)** were tested on **9,000 fixed recordings**, with 3,000 in each group. Inputs use the models' training preparation: silence trimming, loudness normalization and PCM16 quantization. Singapore is selected when `p_singapore >= 0.50`. No retraining or threshold tuning was performed.

## Results

| Test group | ML | Ensemble |
|---|---:|---:|
| MNSC PART1 — Singapore recall | 97.20% | **97.97%** |
| MNSC PART2 — Singapore recall | 97.20% | **99.40%** |
| Common Voice — non-Singapore specificity | 97.73% | **99.03%** |
| **Overall accuracy — 9,000 clips** | **97.38%** | **98.80%** |

Recall measures correctly labelled Singapore recordings; specificity measures correctly labelled non-Singapore recordings. Overall accuracy combines 6,000 Singapore and 3,000 non-Singapore clips.

## Key findings

- **Ensemble performs best:** 108 errors versus ML's 236, or **128 fewer errors**.
- **Both improve overall over 23 Oct:** ML makes 40 fewer errors; Ensemble makes 14 fewer. Ensemble's PART1 result is slightly lower, with two additional misses.
- **The low raw PART2 score came from mismatched preparation.** ML recall rises from **53.60% to 97.20%** after normalization, removing 1,308 misses. Ensemble rises from **91.70% to 99.40%** with fully prepared acoustic and Whisper inputs.

The controlled ML check uses the same 3,000 PART2 recordings:

| Audio preparation | Singapore recall |
|---|---:|
| Original raw audio | 53.60% |
| Silence trimming only | 56.73% |
| Loudness normalization only | 97.20% |
| Trimming + normalization | 97.20% |

Normalization explains most of the recovery. All prepared PART2 clips remain within 3–6 seconds; none was cropped or discarded. Raw-input results are retained as a separate stress test.

## Limits and verification

Common Voice includes 446 recordings from training/validation speakers, including 137 clip IDs also found in training/validation. On the remaining **2,554 recordings**, specificity is **97.77% for ML** and **98.98% for Ensemble**. MNSC has no training/validation speaker overlap, but its speakers appear in the study's held-out test split. This is a fixed comparison, not a newly sampled independent benchmark.

All **45,000 historical class predictions** reproduced with zero differences. Model, metadata, manifest and feature-cache hashes were checked. Both acoustic features and Whisper embeddings for prepared PART2 were extracted from the same prepared waveforms.

## Files

- [summary.csv](summary.csv): all metrics, including 23 Oct baselines and speaker-independent subsets.
- [predictions.csv](predictions.csv): per-recording labels and probabilities.
- [metadata.json](metadata.json): frozen models, inputs, preprocessing and hashes.
- [supporting/](supporting/README.md): raw PART2 results, preprocessing diagnostics, overlap audits and reproduction instructions.
