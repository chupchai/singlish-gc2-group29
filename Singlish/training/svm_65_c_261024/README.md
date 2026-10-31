# Train svm_65_c_261024.joblib

Start with this repository and the 40,000-clip dataset. Extract acoustic features.
Train an SVM with 65 selected features. Calibrate its probabilities.
This procedure uses only the notebooks and helper in this folder.

## 1. Install the software

Use Python 3.14. Run these Bash commands from the repository root on Linux or WSL.

```bash
python3.14 -m venv training/svm_65_c_261024/.venv
source training/svm_65_c_261024/.venv/bin/activate
python -m pip install numpy==2.5.3 scipy==1.18.1 scikit-learn==1.9.1 joblib==1.6.0 \
  librosa==1.0.0 numba==0.68.0 soundfile==0.14.0 soxr==1.1.0 \
  pandas==2.3.3 pyarrow==25.0.1 matplotlib==3.11.2 jupyterlab ipykernel
python -m ipykernel install --user --name singlish-acoustic-py314 --display-name "Singlish acoustic (Python 3.14)"
```

Use this environment for all three notebooks. A GPU is not required.

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

From the repository root, connect the dataset and start JupyterLab:

```bash
ln -s ../../accent_classifier_data/data training/svm_65_c_261024/data
python -m jupyterlab
```

## 3. Set the notebook paths

In each notebook, set `PROJECT_ROOT` to the absolute path of this training folder:

```python
PROJECT_ROOT = Path('/absolute/path/to/Singlish/training/svm_65_c_261024')
```

Replace the existing assignment. Keep `USE_GOOGLE_DRIVE = False` where present.
Keep the other paths relative to `PROJECT_ROOT`.
Select **Singlish acoustic (Python 3.14)** as the kernel. Save each notebook.

## 4. Run the notebooks in order

Use **Restart Kernel and Run All** for each notebook.
Complete each notebook before the next one.

| Order | Notebook | Result |
|---|---|---|
| 1 | [E1_acoustic_feature_count_comparison.ipynb](E1_acoustic_feature_count_comparison.ipynb) | Extract 39 features. Train and compare the first models. |
| 2 | [E1_acoustic_91_feature_count_comparison.ipynb](E1_acoustic_91_feature_count_comparison.ipynb) | Extend the cache to 91 features. Select and train SVM65. |
| 3 | [E1_svm_calibrated_probability.ipynb](E1_svm_calibrated_probability.ipynb) | Apply five-fold speaker-grouped sigmoid calibration. Save the classifier. |

The first two notebooks also train comparison models required by the calibration notebook.
The final SVM fits 33,881 training and validation clips. Test clips do not enter fitting.

Check for these files under this training folder:

```text
outputs/e1_feature_study_accent_40000/svm_calibrated_probability_testing/
  svm_65_calibrated.joblib
  svm_65_calibrated_metadata.json
```

If a data or feature-count check fails, stop and check the dataset and environment.
Do not remove the check to continue.

## 5. Export the MVP files

After notebook 3 completes, run this cell in that notebook.
Replace the repository path with your own path.

```python
import shutil

dest = Path('/absolute/path/to/Singlish/mvp/models')
shutil.copy2(RESULTS_DIR / 'svm_65_calibrated.joblib', dest / 'svm_65_c_261024.joblib')
shutil.copy2(RESULTS_DIR / 'svm_65_calibrated_metadata.json', dest / 'svm_65_c_261024_metadata.json')
```

Copy both files. The metadata supplies the selected feature order.

## 6. Check the export

From the repository root, use this training environment:

```bash
python -c "from mvp.models import AccentModel; m=AccentModel(); print('Model inputs:', m.pipeline.n_features_in_)"
```

Expected result: `Model inputs: 65`. This checks model loading, classes and feature metadata.
It does not run the fusion classifier or test audio predictions.

Keep the executed notebooks and new output records with your submission.
The dated MVP filename is a serving name. A new fit can have different model bytes or results.
