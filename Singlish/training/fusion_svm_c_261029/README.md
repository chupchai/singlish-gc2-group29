# Train fusion_svm_c_261029.joblib

Start with this repository and the 40,000-clip dataset. Build acoustic and Whisper features.
Train the fusion SVM with 65 acoustic features and 512 Whisper values.
Calibrate its probabilities. Whisper stays frozen.
This folder includes all required notebooks and helpers, including the acoustic prerequisites.

## 1. Install the software

Use Python 3.14. Run these Bash commands from the repository root on Linux or WSL.

```bash
python3.14 -m venv training/fusion_svm_c_261029/.venv
source training/fusion_svm_c_261029/.venv/bin/activate
python -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r mvp/requirements.txt
python -m pip install pandas==2.3.3 pyarrow==25.0.1 matplotlib==3.11.2 jupyterlab ipykernel nbclient
python -m ipykernel install --user --name singlish-fusion-py314 --display-name "Singlish final fusion (Python 3.14)"
```

A GPU can reduce the time for Whisper feature extraction. For GPU use, install
the CUDA build of the same PyTorch version instead of the CPU build.
Use one environment for all six notebooks.

## 2. Prepare the dataset

Download the [shared dataset](https://drive.google.com/file/d/1EI7rEoAshSkGSpFY8MsJR0994x2y4Gwl/view?usp=sharing).
Extract it into `accent_classifier_data/data/`.
It contains audio, labels and speaker splits. It contains no trained classifier.

To download and prepare the original sources instead, follow
[Run from scratch](../../accent_classifier_data/README.md#run-from-scratch).
Use a separate environment for dataset preparation. Then reactivate the training environment.

Confirm that `data/` contains `cv_subset/`, `mnsc_subset/`, `saa_subset/` and
`speechocean_subset/`. Each source folder must contain `part-*.parquet` files.
Keep all 40,000 clips and the fixed speaker splits.
Follow the [dataset access conditions](../../accent_classifier_data/README.md#licenses).

Run these commands from the repository root:

```bash
mkdir -p accent_classifier_data/data/manifests
cp accent_classifier_data/manifests/clips_40000.csv.gz accent_classifier_data/data/manifests/
ln -s ../../accent_classifier_data/data training/fusion_svm_c_261029/data
mkdir -p training/fusion_svm_c_261029/results
python -m jupyterlab
```

## 3. Set the notebook paths

Use absolute paths on your PC. Set the paths in the table below.
Replace the original assignments in the included notebooks.

| Setting | Path |
|---|---|
| E1, original fusion and E2 `PROJECT_ROOT`; A1 and A2 `ROOT` | `<repo>/training/fusion_svm_c_261029` |
| A1 and A2 `UPSTREAM` | `<repo>/training/fusion_svm_c_261029/outputs` |
| E2 `DATA_ROOT`; A1 `DATA` | `<repo>/training/fusion_svm_c_261029/data` |
| Any reference to the original `/home/mh/nusproj/Singlish` root | `<repo>` |

Keep `USE_GOOGLE_DRIVE = False` where present.
In E2, add these settings immediately after `config = study.config(...)`.
Replace `<repo>` with your absolute repository path.

```python
config['e1'] = Path('<repo>/training/fusion_svm_c_261029/outputs/e1_feature_study_accent_40000')
config['fusion'] = Path('<repo>/training/fusion_svm_c_261029/outputs/svm65_fusion_comparison_40000')
config['reference'] = Path('<repo>/accent_classifier_data/manifests/clips_40000.csv.gz')
```

Keep `frozen_whisper_40k.py` unchanged. The E2 settings replace its local paths.
In A1, also update the `reference` path in the input-verification cell.
In A2, also update the repository path in the `protect(...)` loop.
Select **Singlish final fusion (Python 3.14)** as the kernel.
Save each notebook before execution. A1 and A2 read their own saved code.

## 4. Run the notebooks in order

Use **Restart Kernel and Run All** for each notebook.
Complete each notebook before the next one.

| Order | Notebook | Result |
|---|---|---|
| 1 | [E1_acoustic_feature_count_comparison.ipynb](E1_acoustic_feature_count_comparison.ipynb) | Build 39 features and train the first models. |
| 2 | [E1_acoustic_91_feature_count_comparison.ipynb](E1_acoustic_91_feature_count_comparison.ipynb) | Build 91 features. Select and train SVM65. |
| 3 | [E4_02_whisper_svm_concatenate.ipynb](notebooks/E4_02_whisper_svm_concatenate.ipynb) | Extract 512 Whisper values. Train the earlier fusion model. |
| 4 | [E2_01_whisper_classifier_beginner.ipynb](notebooks/E2_01_whisper_classifier_beginner.ipynb) | Train the Whisper control classifiers. |
| 5 | [E4_02_1_tuned_acoustic_svm_whisper_and_fusion.ipynb](notebooks/E4_02_1_tuned_acoustic_svm_whisper_and_fusion.ipynb) | Rank acoustic features on training clips. Save the A1 record. |
| 6 | [E4_02_5_frozen_full512_vs_fusion_calibrated.ipynb](notebooks/E4_02_5_frozen_full512_vs_fusion_calibrated.ipynb) | Train and save the final calibrated fusion classifier. |

Before order 3, set `WHISPER_REVISION` to
`'911407f4214e0e1d82085af863093ec0b66f9cd6'`.
Keep `config['rebuild_whisper'] = False` in order 4.
It uses the embeddings that order 3 has just created. No old cache is required.

### Before order 5: set the new-run fingerprints

A1 contains checksums for the published run. New timing records and feature caches
can have different checksums. Update only the working A1 notebook:

1. Run its settings/import cell to define `E2`, `E1`, `E4` and `sha256`.
2. Run the cell below to calculate your new input fingerprints.
3. Replace the values in `EXPECTED_E2_HASHES` with the printed values.
4. Replace the literal hashes for `speaker_splits.csv`, `acoustic_features_91.npz`
   and `whisper_provenance['cache_file_sha256']` in the input-verification cell.
5. Replace the literal acoustic `cache_id` with the printed cache ID.
6. Save the notebook. Then restart the kernel and run all cells.

```python
print({name: sha256(E2 / name) for name in EXPECTED_E2_HASHES})
for name in ['speaker_splits.csv', 'acoustic_features_91.npz']:
    print(name, sha256(E1 / name))
meta = json.loads((E4 / 'model_metadata.json').read_text())
w = E4 / 'feature_cache' / ('whisper_' + meta['whisper_provenance']['cache_id'] + '.npz')
print('Whisper cache SHA-256:', sha256(w))
with np.load(E1 / 'acoustic_features_91.npz', allow_pickle=False) as z:
    print('Acoustic cache ID:', z['cache_id'].item())
```

These values identify your new run. Keep the fixed dataset-manifest checksum,
encoder revision, clip checks, speaker checks and feature-order checks unchanged.
If a data or feature-count check fails, stop and check the dataset and environment.
Do not remove assertions to continue after a data mismatch.
Use empty output folders for a new run. A1 and A2 reject incompatible checkpoints.

Keep notebooks `E4_02_2`, `E4_02_3` and `E4_02_4` in `notebooks/`.
The final notebook checks their files. You do not need to run them.
Orders 1–5 create the features and intermediate models required by order 6.
The final classifier uses all 577 values, without PCA.
Five-fold speaker-grouped sigmoid calibration uses 33,881 training and validation clips.
Test clips do not enter fitting.

## 5. Export the MVP files

After order 6 completes, check for this file under this training folder:

```text
outputs/e4_02_5_frozen_full512_vs_fusion_calibrated_20261029/
  fusion_full512_acoustic65_calibrated.joblib
```

Run this cell in the final notebook. `OUT` and `provenance` are already defined.
Replace the repository path with your own path.

```python
import shutil

dest = Path('/absolute/path/to/Singlish/mvp/models')
source = OUT / 'fusion_full512_acoustic65_calibrated.joblib'
shutil.copy2(source, dest / 'fusion_svm_c_261029.joblib')
record = json.loads((OUT / 'run_record.json').read_text())
metadata = dict(
    acoustic_columns_in_91=record['selected_columns'],
    acoustic_feature_names=record['selected_feature_names'],
    whisper_width=512, positive_label=1,
    whisper_provenance=provenance,
    source_model=str(source), source_sha256=sha(source),
    calibration='sigmoid', ensemble=False, threshold=0.5,
)
(dest / 'fusion_svm_c_261029_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
```

This creates metadata for your new model. Keep it with the `.joblib` file.
Do not reuse the published feature selection or test scores for a different fit.

## 6. Check the export

From the repository root, use this training environment:

```bash
python -c "from mvp.models import EnsembleModel; m=EnsembleModel(); print('Model inputs:', m.pipeline.n_features_in_)"
```

Expected result: `Model inputs: 577`. This checks model loading, classes and feature metadata.
It does not load the acoustic MVP classifier, download Whisper or test audio predictions.

Keep the executed notebooks and new output records with your submission.
The dated MVP filename is a serving name. A new fit can have different model bytes or results.
