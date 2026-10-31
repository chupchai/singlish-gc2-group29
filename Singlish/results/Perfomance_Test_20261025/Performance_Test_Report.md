# Performance test — 25 October 2026

Frozen 25 Oct models were tested on the same **9,000 fixed recordings** as the 24 Oct comparison, with 3,000 in each group. Singapore is selected when `p_singapore >= 0.50`. No retraining or threshold tuning was performed.

**Preprocessing matters:** this benchmark uses trimmed, normalized PCM16 inputs. The 25 Oct models were trained on trimmed float32 audio with source gain preserved. These results measure performance on the existing normalized benchmark; they do not evaluate the full training preparation of the 25 Oct models.

## Results on the normalized benchmark

| Test group | 25 Oct ML (SVM65) | 25 Oct Ensemble |
|---|---:|---:|
| MNSC PART1 — Singapore recall | 94.47% | 96.93% |
| MNSC PART2 — Singapore recall | 94.70% | 98.97% |
| Common Voice — non-Singapore specificity | 97.87% | 99.17% |
| **Overall accuracy — 9,000 clips** | **95.68%** | **98.36%** |

Overall accuracy combines 6,000 Singapore and 3,000 non-Singapore clips.

## Key findings

- **Ensemble is the strongest 25 Oct model:** 148 errors versus SVM65's 389.
- On the normalized benchmark, Ensemble scores **98.36% versus 98.80% for 24 Oct**, with 40 additional errors. Common Voice false positives improve from 29 to 25, while Singapore misses rise from 79 to 123.
- Among acoustic models, **SVM91 scores 95.91%**, followed by SVM65 at 95.68% and SVM39 at 92.16%.

## Raw PART2 stress test

On the same 3,000 PART2 recordings, retaining source loudness improves recall:

| 25 Oct model | Raw recall | Normalized recall |
|---|---:|---:|
| ML (SVM65) | 97.50% | 94.70% |
| Ensemble | 99.57% | 98.97% |

The raw stress test uses full, untrimmed PCM16 audio. It also differs from the 25 Oct training preparation, which trims audio and retains float32 samples.

## Limits and verification

Common Voice has 446 clips from training/validation speakers, including 137 matching training/validation clip IDs. On the remaining 2,554 clips, 25 Oct specificity is 97.81% for ML and 99.14% for Ensemble. MNSC has no training/validation speaker overlap, but its speakers appear in the study test split.

Both 24 Oct baselines reproduced all **18,000 archived class predictions** with zero differences; probabilities matched within 1e-10 tolerance. Model, metadata, feature-cache, manifest and split hashes were checked.

## Files

- [summary.csv](summary.csv): all six models, 24 Oct baselines and speaker-independent subsets.
- [predictions.csv](predictions.csv): per-recording labels and probabilities.
- [metadata.json](metadata.json): frozen model paths, input hashes and preprocessing.
- [supporting/](supporting/README.md): raw PART2 results, preprocessing checks, audits and reproduction instructions.
