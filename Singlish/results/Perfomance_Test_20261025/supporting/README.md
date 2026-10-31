# Supporting evidence and reproduction

The [main report](../Performance_Test_Report.md) contains the key results and preprocessing limitation.

| Files | Purpose |
|---|---|
| `part2_raw_prepared_comparison.csv` | Raw versus normalized PART2 recall and misses for all six models. |
| `raw_part2_predictions.csv` | Per-recording raw PART2 labels and probabilities for all six models. |
| `part2_preprocessing_summary.csv` | Four controlled 25 Oct SVM65 preparation variants. |
| `overlap_audit.json` | Training/validation speaker and clip-ID overlap. |
| `baseline_reproduction.csv` | Verification of 18,000 predictions from the two 24 Oct baselines. |
| `run_metadata.json` | Runtime, script hash, input metadata hash and verification totals. |

`mnsc_results.csv` and `common_voice_results.csv` were exact filtered views of `../summary.csv` and were removed. All unique results and predictions are retained.

## Reproduce

Run from the repository root using the existing local model artifacts and fixed feature caches:

```bash
HF_HUB_OFFLINE=1 mvp/.venv/bin/python results/Perfomance_Test_20261025/reproduce.py
```

The script reads frozen paths and hashes from `../metadata.json`. Defaults use the sibling `E1_Feature_EngineeringV5` study, model paths relative to the parent of this repository, `tmp/release-comparison-portable-cache`, and the prepared PART2 cache recorded in the metadata. Override with `--study-root`, `--artifact-root`, `--cache` or `--part2-cache`. Use `--output` for a separate destination. Primary CSVs and the report go in that destination; diagnostics go in its `supporting/` subfolder.

Both acoustic and Whisper inputs use matching prepared waveforms. The primary benchmark uses trimmed, normalized PCM16 audio, including the corrected PART2 cache from the [24 Oct test](../../Perfomance_Test_20261024/supporting/README.md). The supplemental raw PART2 test uses the original full, unnormalized PCM16 recordings. Neither protocol exactly matches the 25 Oct training preparation of trimmed float32 audio with source gain preserved.

The SVM65 ablation gives 97.50% recall on raw audio, 97.33% after trimming alone, 94.57% after normalization alone, and 94.70% after trimming plus normalization. These are controlled results for the same 3,000 recordings with the same frozen model and threshold.

No fitting, feature selection or threshold tuning occurs. The scorer verifies model, sidecar, feature-cache, manifest, split, notebook and baseline hashes. It reproduces both 24 Oct models' prepared-input labels and probabilities before writing results.
