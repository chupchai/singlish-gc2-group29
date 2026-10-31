# Supporting evidence and reproduction

The [main report](../Performance_Test_Report.md) contains the results and key findings. This folder holds the supporting evidence.

| Files | Purpose |
|---|---|
| `raw_part2_summary.csv`, `raw_part2_predictions.csv` | Historical unnormalized PART2 stress test. PART1 and Common Voice are unchanged and already present in the primary CSVs. |
| `part2_preprocessing_summary.csv`, `part2_preprocessing_predictions.csv` | Four controlled SVM65 preparation variants and per-clip probabilities. |
| `internal_test_by_part.csv`, `part2_calibration_feature_diagnostics.csv` | Notebook split, calibration and SVM91 checks. Prepared notebook PART2 SVM65 recall is 97.75%; raw uncalibrated SVM65 recall is 57.77%, and raw calibrated SVM91 recall is 63.73%. |
| `overlap_audit.json` | Training/validation speaker and clip-ID overlap. |
| `baseline_reproduction.csv` | All 45,000 historical class predictions reproduced without differences. |
| `run_metadata.json` | Scoring runtime, script hash and verification totals. |
| `preparation_verification.json` | Original prepared-cache extraction record: 3,000 PCM hashes and preparation checks, source Parquet hashes, extraction settings and code hashes. |

The original preparation record describes the historical extraction. Its script/code hashes identify that execution; the maintained preparation script now defines the 24 Oct normalization explicitly.

`mnsc_results.csv` and `common_voice_results.csv` were redundant filtered views of `../summary.csv`. The former `part2_prepared_comparison.csv` can be reconstructed from `raw_part2_summary.csv` and `../summary.csv`. These duplicate views were removed. No unique predictions were discarded.

## Reproduce scoring

Run from the repository root with the existing caches, model files and V4 study outputs:

```bash
HF_HUB_OFFLINE=1 mvp/.venv/bin/python results/Perfomance_Test_20261024/reproduce.py
```

The defaults use `tmp/release-comparison-portable-cache`, `mvp/models`, the sibling `E1_Feature_EngineeringV4` study, and the prepared cache recorded in `../metadata.json`. Override with `--cache`, `--models`, `--study-root` or `--part2-cache`. Use `--output` for a separate scoring destination. Primary outputs go in that destination; diagnostics go in its `supporting/` subfolder.

## Fresh PART2 preparation

Use the CUDA extraction environment recorded in `preparation_verification.json`: PyTorch 2.14.0+cu130, Transformers 4.57.6, NumPy 2.5.3, librosa 1.0.0 and pyarrow 25.0.1. Run from the repository root with the original Parquets, raw feature cache and pinned Whisper weights available:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 python3 \
  results/Perfomance_Test_20261024/prepare_part2.py \
  --data-root ../E1_Feature_Engineering/data/mnsc_part2_unseen_3000 \
  --output outputs/performance_test_20261024_prepared_fresh
```

The fixed protocol is mono 16 kHz; 25 ms RMS frames, 10 ms hops, a 40 dB relative silence threshold and 100 ms edge padding; target RMS −20 dBFS, peak limit 0.99; PCM16 FLAC quantization/decode. All 3,000 trimmed recordings are 3.000–5.998 seconds, so no cropping or exclusions are needed. Whisper uses revision `911407f4214e0e1d82085af863093ec0b66f9cd6`, CUDA bfloat16, batch size 64, and real-frame mean pooling. Acoustic and Whisper features use the same prepared waveforms.

The scorer enforces the recorded archive hash. Fresh extraction can produce a different archive hash even when arrays agree; compare clip IDs and arrays and record fresh provenance before treating it as the same extraction. Keep the recorded cache for exact historical scoring. Original manifests and cache-reconstruction instructions are in the [23 Oct report](../../Perfomance_Test_20261023/Performance_Test_Report.md#reproduce).
