# Performance Test Report — 23 October 2026

Four release models, using the same fixed clips and decision rules as the two tests.

## MNSC: Singapore clips

Each set has 3,000 clips. Misses are Singapore clips labelled non-Singapore.

| Model | PART1 recall | Misses | PART2 recall | Misses |
|---|---:|---:|---:|---:|
| Latest SVM65 calibrated | 97.07% | 88 | 53.43% | 1,397 |
| Latest fusion65 calibrated | 98.03% | 59 | 93.23% | 203 |
| Previous SVM35* | 95.90% | 123 | 30.77% | 2,077 |
| Previous fusion35 | 97.57% | 73 | 42.77% | 1,717 |

## Common Voice: non-Singapore clips

3,000 clips from 2,469 speakers; seed `20261023`. False positives are clips labelled Singapore.

| Model | Specificity | False positives | FP rate |
|---|---:|---:|---:|
| Latest SVM65 calibrated | 97.03% | 89 | 2.97% |
| Latest fusion65 calibrated | 98.83% | 35 | 1.17% |
| Previous SVM35* | 95.90% | 123 | 4.10% |
| Previous fusion35 | 98.83% | 35 | 1.17% |

\* Previous SVM35 uses its deployed raw SVM boundary. At its calibrated 0.50 threshold,
PART1 recall is 94.40% (168 misses), PART2 recall is 26.60% (2,202 misses),
and Common Voice specificity is 97.00% (90 false positives).
Latest models use `p_singapore >= 0.50`.

PART1 and Common Voice use prepared 3–6 second normalized audio; PART2 uses full
original recordings. Evaluation decodes these unchanged, without MVP upload processing.
No training/validation speakers overlap either release. MNSC sets can share held-out
speakers. Common Voice is a random sample from an eligible prepared pool covering
American, British, South Asian, Filipino and Australian accents.

## Files

- `manifests/`: all 9,000 clip IDs, source locations, transcripts, speaker IDs and audio/PCM hashes.
- `metadata.json`: source revisions, model download URLs/hashes, seeds and runtime settings.
- `*_results.csv` and `predictions.csv`: summary results and every saved class prediction.

## Reproduce

Run from the repository root. Audio and model binaries stay outside Git.

```bash
pip install -r mvp/requirements.txt pandas==2.3.3 pyarrow==25.0.1 huggingface_hub==0.36.2
python results/Perfomance_Test_20261023/prepare_clips.py --dataset mnsc_part1_unseen_3000
python results/Perfomance_Test_20261023/prepare_clips.py --dataset mnsc_part2_unseen_3000
python results/Perfomance_Test_20261023/prepare_clips.py --dataset cv_unseen_3000 --cv-archive /path/to/common-voice-scripted-speech-26-0-english.tar.gz
python results/Perfomance_Test_20261023/reproduce.py
```

MNSC source shards download from the pinned Hugging Face revision. Obtain the
**Common Voice Scripted Speech 26.0 English** archive from
[Mozilla Data Collective](https://mozilladatacollective.com/).
Preparation selects the manifest entries and verifies source and prepared PCM hashes.
Existing MNSC source shards can be supplied with `--mnsc-source /path/to/source`.

If the three prepared Parquet datasets are already available, skip preparation:

```bash
python results/Perfomance_Test_20261023/reproduce.py --data-root /path/to/data
```

The script downloads missing release models and writes results under
`outputs/release_comparison_reproduced/`. Use GPU bfloat16 to match the tests;
other runtimes may differ slightly. Exact recorded versions are in `metadata.json`.
