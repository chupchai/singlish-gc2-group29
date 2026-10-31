"""Saved ML/Ensemble contracts, calibrated predictions and Whisper pooling."""
import json
import shutil
from threading import Lock
from types import SimpleNamespace

import joblib
import numpy as np
import pytest
import torch

from mvp import models
from mvp.embeddings import WhisperEmbedding


def test_ensemble_concatenates_raw_features_before_saved_preprocessing(monkeypatch):
    model = models.EnsembleModel()
    vector = np.arange(91, dtype=np.float32)
    embedding = np.linspace(-1, 1, 512, dtype=np.float32)
    expected = np.concatenate([vector[model.columns], embedding]).reshape(1, 577)
    probabilities = model.pipeline.predict_proba(expected)[0]
    label = int(probabilities[1] >= 0.5)
    score = float(model.decision_estimator.decision_function(expected)[0])
    result = model.predict(vector, embedding)
    assert result['class_id'] == label
    assert result['decision_score'] == pytest.approx(score)
    assert result['probability'] == pytest.approx(probabilities[label])
    # Capture probability inputs so an accidental feature reorder cannot pass on labels alone.
    captured = []
    def predict_proba(values):
        captured.append(values)
        return probabilities.reshape(1, 2)
    monkeypatch.setattr(model.pipeline, 'predict_proba', predict_proba)
    model.predict(vector, embedding)
    np.testing.assert_array_equal(captured[0], expected)


@pytest.mark.parametrize('embedding', [np.zeros(32), np.full(512, np.nan)])
def test_ensemble_rejects_invalid_embeddings(embedding):
    with pytest.raises(ValueError, match='512-value'):
        models.EnsembleModel().predict(np.zeros(91), embedding)


def test_ensemble_rejects_feature_order(monkeypatch, tmp_path):
    path = tmp_path / 'fusion.joblib'
    shutil.copy2(models.ROOT / 'models/fusion_svm_c_261024.joblib', path)
    metadata = json.loads((models.ROOT / 'models/fusion_svm_c_261024_metadata.json').read_text())
    metadata['acoustic_feature_names'].reverse()
    path.with_name('fusion_metadata.json').write_text(json.dumps(metadata))
    monkeypatch.setenv('ENSEMBLE_MODEL_PATH', str(path))
    with pytest.raises(ValueError, match='feature names/order'):
        models.EnsembleModel()


def test_whisper_pooling_excludes_thirty_second_padding():
    model = WhisperEmbedding.__new__(WhisperEmbedding)
    hidden = torch.full((1, 1500, 512), 999.)
    # 3.001 seconds rounds up to 151 real encoder steps, matching training.
    hidden[0, :151] = torch.arange(151, dtype=torch.float32)[:, None]
    model.extractor = lambda *args, **kwargs: SimpleNamespace(input_features=torch.zeros(1, 80, 3000))
    model.encoder = lambda **kwargs: SimpleNamespace(last_hidden_state=hidden)
    actual = model(np.zeros(48001, dtype=np.float32))
    np.testing.assert_array_equal(actual, np.full(512, 75., dtype=np.float32))


@pytest.mark.parametrize('ml_class,ensemble_class,agreement', [(1, 1, 'agree'), (0, 1, 'disagree')])
def test_agreement_compares_predicted_classes(ml_class, ensemble_class, agreement):
    pair = models.AccentModels.__new__(models.AccentModels)
    pair.ml = SimpleNamespace(manifest={'model_version': 'ml'}, predict=lambda _: {'class_id': ml_class})
    pair.ensemble = SimpleNamespace(version='fusion', predict=lambda *_: {'class_id': ensemble_class})
    pair.embedding = lambda _: np.zeros(512)
    pair.lock = Lock()
    result = pair.predict(np.zeros(91), np.zeros(48000))
    assert result['agreement'] == agreement
    assert result['prediction'] == result['models']['ensemble']['prediction']


def test_model_rejects_feature_order(monkeypatch):
    artifact = {'model': joblib.load(models.ROOT / 'models/svm_65_c_261024.joblib'),
                'columns': list(range(91)), 'feature_names': models.FEATURE_NAMES.copy()}
    artifact['feature_names'] = list(reversed(artifact['feature_names']))
    monkeypatch.setattr(models.joblib, 'load', lambda path: artifact)
    with pytest.raises(ValueError, match='feature names/order'):
        models.AccentModel()


@pytest.mark.parametrize('calibrated', [True, False])
def test_model_override_needs_no_checksum(monkeypatch, tmp_path, calibrated):
    alternate = tmp_path / 'development-model.joblib'
    if calibrated:
        shutil.copy2(models.ROOT / 'models/svm_65_c_261024.joblib', alternate)
    else:
        # Exercise an uncalibrated export using the shipped model's fitted SVM.
        models.joblib.dump(models.AccentModel().decision_estimator, alternate)
    shutil.copy2(models.ROOT / 'models/svm_65_c_261024_metadata.json',
                 tmp_path / 'development-model_metadata.json')
    monkeypatch.setenv('MODEL_PATH', str(alternate))
    model = models.AccentModel()
    assert model.manifest['model_version'] == 'development-model'
    vector = np.arange(91, dtype=np.float32)
    values = vector[model.columns].reshape(1, -1)
    result = model.predict(vector)
    assert result['class_id'] == (int(model.pipeline.predict_proba(values)[0, 1] >= 0.5)
                                  if calibrated else model.decision_estimator.predict(values)[0])
    assert result['decision_score'] == pytest.approx(model.decision_estimator.decision_function(values)[0])
    assert (result['probability'] is None) == (not calibrated)
    monkeypatch.setenv('MODEL_VERSION', 'experiment-2')
    assert models.AccentModel().manifest['model_version'] == 'experiment-2'


@pytest.mark.parametrize('score,label,probabilities', [(1.2, 0, [0.89, 0.11]), (-0.4, 1, [0.11, 0.89]), (0., 1, [0.5, 0.5])])
@pytest.mark.parametrize('ensemble', [False, True])
def test_label_uses_calibrated_threshold_and_keeps_raw_score(monkeypatch, score, label, probabilities, ensemble):
    model = models.EnsembleModel() if ensemble else models.AccentModel()
    monkeypatch.setattr(model.pipeline, 'predict_proba', lambda x: np.asarray([probabilities]))
    monkeypatch.setattr(model.decision_estimator, 'decision_function', lambda x: np.asarray([score]))
    monkeypatch.setattr(model.decision_estimator, 'predict', lambda x: np.asarray([label]))
    vector = np.zeros(91, dtype=np.float32)
    result = model.predict(vector, np.zeros(512, dtype=np.float32)) if ensemble else model.predict(vector)
    assert result['class_id'] == label
    assert result['decision_score'] == score
    assert result['probability'] == probabilities[label]
    assert result['display_label'] == ('Singapore Accent' if label else 'Non-Singapore Accent')


def test_selected_columns_match_export_and_are_applied_in_order(monkeypatch):
    metadata = json.loads((models.ROOT / 'models/svm_65_c_261024_metadata.json').read_text())
    model = models.AccentModel()
    assert model.columns == metadata['columns']
    assert len(model.columns) == model.pipeline.n_features_in_ == 65
    vector = np.arange(91, dtype=np.float32)
    captured = []
    def probabilities(x):
        captured.append(x.copy())
        return np.asarray([[0.1, 0.9]])
    monkeypatch.setattr(model.pipeline, 'predict_proba', probabilities)
    assert model.predict(vector)['decision_score'] == pytest.approx(
        model.decision_estimator.decision_function(vector[metadata['columns']].reshape(1, 65))[0])
    np.testing.assert_array_equal(captured[0], vector[metadata['columns']].reshape(1, 65))


def test_selected_model_requires_metadata(monkeypatch, tmp_path):
    path = tmp_path / 'missing_metadata.joblib'
    shutil.copy2(models.ROOT / 'models/svm_65_c_261024.joblib', path)
    monkeypatch.setenv('MODEL_PATH', str(path))
    with pytest.raises(ValueError, match='need columns and feature_names metadata'):
        models.AccentModel()
