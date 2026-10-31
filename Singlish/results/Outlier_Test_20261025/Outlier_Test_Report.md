# Outlier test — 24 vs 25 October Ensembles

Final calibrated E4 models; Singapore when p_singapore >= 0.50. Each model uses its matching preprocessing. Both acoustic and Whisper features are freshly extracted from that same prepared waveform with CPU float32 inference.

All 28 available WAVs are labelled non-Singapore as requested. The source manifest lists 30 clips. Missing: malay/malay_01.wav, malay/malay_02.wav.

| Language | Clips | 24 Oct correct non-SG | 24 Oct false SG | 25 Oct correct non-SG | 25 Oct false SG |
|---|---:|---:|---:|---:|---:|
| malay | 8 | 7/8 | 1 | 5/8 | 3 |
| mandarin | 10 | 5/10 | 5 | 4/10 | 6 |
| tamil | 10 | 3/10 | 7 | 3/10 | 7 |
| all | 28 | 15/28 | 13 | 12/28 | 16 |

Rejected model/clip evaluations: 0. Missing files and rejections are never counted as correct classifications.

Non-English Google FLEURS outliers: Malay, Mandarin and Tamil. Source language alone does not verify accent labels; this is an outlier stress test.

## Reproduce

```bash
HF_HUB_OFFLINE=1 mvp/.venv/bin/python results/Outlier_Test_20261025/reproduce.py
```

Per-clip probabilities are in `predictions.csv`; aggregate results are in `summary.csv`. `metadata.json` records input/model hashes, missing files, preprocessing and runtime.
