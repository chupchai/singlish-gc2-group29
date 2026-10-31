# Singlish POC

Proof of concept for Singapore English accent classification.

The repository contains simple Colab notebooks for each experiment.

Audio, feature caches, trained models and notebook outputs stay in each member's Google
Drive.

## Experiment and notebook map

### E1 — Supervised Learning

The draft compares classical models using handcrafted acoustic features. The i-vector notebook extends E1 with a traditional speech embedding.

| Notebook | Method | Purpose / Output |
|---|---|---|
| [SVM](notebooks/E1_01_svm_beginner.ipynb) | Train an SVM on handcrafted acoustic features. | Establish a supervised acoustic-feature baseline; report F1, balanced accuracy and a confusion matrix for unseen speakers. |
| [Random Forest](notebooks/E1_02_random_forest_beginner.ipynb) | Train a Random Forest on handcrafted acoustic features. | Check whether a tree-based model improves the E1 baseline; compare F1, balanced accuracy and confusion matrices. |
| [Logistic Regression](notebooks/E1_03_logistic_regression_beginner.ipynb) | Train a scaled logistic-regression classifier on acoustic features. | Check whether a simpler linear model is competitive |
| [i-vectors](notebooks/E1_04_ivector_beginner.ipynb) | Fit an MFCC-based UBM and total-variability model, then classify i-vectors. | Explore whether a traditional speech embedding captures accent information beyond E1's clip-level acoustic features. |

### E2 — Transfer Learning

The draft uses frozen Whisper embeddings. The x-vector and ECAPA-TDNN notebooks test other pretrained speech encoders.

| Notebook | Method | Purpose / Output |
|---|---|---|
| [Whisper classifier](notebooks/E2_01_whisper_classifier_beginner.ipynb) | Extract frozen Whisper encoder embeddings and train a logistic-regression head. | Test whether pretrained Whisper speech representations outperform handcrafted acoustic features without training a large network from scratch. |
| [x-vectors](notebooks/E2_03_xvector_beginner.ipynb) | Extract pretrained conventional TDNN speaker embeddings and train an accent classifier. | Test whether pretrained speaker embeddings transfer to Singapore-accent classification on unseen speakers. |
| [ECAPA-TDNN](notebooks/E2_04_ecapa_tdnn_beginner.ipynb) | Extract pretrained ECAPA-TDNN speaker embeddings and train an accent classifier. | Test whether a newer speaker embedding helps accent classification relative to x-vectors and Whisper. |

### E3 — Unsupervised learning

| Notebook | Method | Purpose / Output |
|---|---|---|
| [Whisper embedding clusters](notebooks/E3_01_unsupervised_learning_beginner.ipynb) | Apply PCA, UMAP and K-means to Whisper embeddings without accent labels during clustering. | Check whether Singapore and non-Singapore clips show natural structure; produce 2D cluster views and discuss overlap. |

### E4 — Ensemble of Models

The draft specifies weighted probability voting. The feature-fusion notebook tests an additional way to combine E1 and E2.

| Notebook | Method | Purpose / Output |
|---|---|---|
| [Weighted soft-voting ensemble](notebooks/E4_01_hybrid_ensemble_beginner.ipynb) | Select the best E1 model and E1/Whisper voting weight on validation speakers. | Test whether combining the strongest classical model with Whisper improves F1 and balanced accuracy; output the final Singapore-accent probability. |
| [Whisper + E1 feature fusion](notebooks/E4_02_whisper_e1_feature_fusion_beginner.ipynb) | Concatenate Whisper embeddings and E1 features for one classifier. | Test whether the two feature types add complementary information and improve F1 or balanced accuracy over the best single branch. |

Open a notebook in Colab and configure the Drive path and audio limit:

```python
PROJECT_ROOT = Path('/content/drive/MyDrive/Singlish')
MAX_CLIPS_PER_CLASS = 100  # Set None to use all audio.
```

The E1 and E2 notebooks choose up to this many clips from each class, spreading
the selection across speakers with a fixed random seed. Use the same value in
each notebook to compare the same recordings. For a formal comparison, also use
the exact same saved speaker split; a shared seed alone does not ensure this if
the selected clips differ. E3 reads the Whisper E2 embedding cache for the
matching limit, so run E2_01 through its cache cell first. E4_01 needs an E1
feature cache and the E2_01 embedding cache; E4_02 extracts and caches its own
features. If you add or remove audio that changes E2_01's selected file list,
set `REBUILD_CACHE = True` in that notebook for that run.

## Google Drive layout

Each team member uploads the prepared `Singlish` folder to `MyDrive`:

```text
Singlish/
├── data/
│   ├── README.md
│   ├── dataset_summary.json
│   ├── manifest.csv
│   └── audio/
│       ├── singapore/speaker_###/*.wav
│       └── non_singapore/speaker_###/*.wav
```

Upload the audio and the three files shown under `data/`. A notebook creates
its own output folder when it runs; for example, the E1 notebooks create
`Singlish/outputs/e1`.

Use the same audio for every experiment. E1 uses acoustic features, E2 creates
Whisper embeddings, E3 clusters the embeddings, and E4 combines the
supervised model probabilities. The beginner notebooks create their own
speaker-disjoint split with the same random seed.

'Singlish' folder shared at following Google drive link:
  https://drive.google.com/drive/folders/1rg_NL97iBC9lwSBO9bz7idZ2tkMJHWqi?usp=drive_link

## Simple team workflow

All three members may use `main`. Pull before editing, avoid editing the same
notebook simultaneously, clear large notebook outputs, and commit one coherent
change at a time.

Do not commit audio, models, credentials or Google Drive outputs.
