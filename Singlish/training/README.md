# Train the MVP classifiers

Choose one procedure. Each folder contains its own required notebooks, helper
files and detailed instructions. Each procedure starts with the dataset.

| Model file | Training procedure |
|---|---|
| `svm_65_c_261024.joblib` | [Train the acoustic classifier](svm_65_c_261024/README.md) |
| `fusion_svm_c_261029.joblib` | [Train the fusion classifier](fusion_svm_c_261029/README.md) |

The fusion procedure builds its own acoustic features and intermediate models.
You can run either procedure without running the other procedure.
To run the complete MVP, export both classifiers and their matching metadata.
