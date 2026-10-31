"""Frozen Whisper controls for the fixed 24 October 2026 fusion experiment.

This module does not use the separate 2,000-clip experiment or change its files.
"""
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
import hashlib
import io
import json
import platform

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import soundfile as sf
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SEED = 42
MODEL = 'openai/whisper-base.en'
REVISION = '911407f4214e0e1d82085af863093ec0b66f9cd6'
REFERENCE_SHA = '87389b0aeb1ac273eff9e669f298b99bc3a73642ead1357d6690aecf5988aad4'
DATASET_ID = '899dd94a72cff7dc6536993ea0c91ce0aed927a80983050b781c0b71d2f0f4ba'
META = ['label', 'speaker_group', 'split', 'source', 'source_clip_id',
        'speaker_id', 'accent_group', 'sampling_rate']
MODELS = ['Whisper full512 SVM', 'Whisper PCA32 SVM']
REFERENCES = ['SVM65', 'Fusion SVM']
COLORS = ['#2378a5', '#58a58a', '#8c91a1', '#e18b37']


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def config(project_root, data_root):
    upstream = Path('/home/mh/nusproj/E1_Feature_EngineeringV4/outputs')
    cfg = dict(root=Path(project_root), data=Path(data_root),
               output=Path(project_root) / 'outputs/frozen_whisper_40000_20261028',
               e1=upstream / 'e1_feature_study_accent_40000',
               fusion=upstream / 'svm65_fusion_comparison_40000',
               reference=Path('/home/mh/nusproj/Singlish/accent_classifier_data/manifests/clips_40000.csv.gz'),
               c_values=[0.01, 0.1, 1., 10.], folds=3, bootstrap=2000,
               workers=8, batch_size=64, rebuild_whisper=False, run_folds=True)
    cfg['output'].mkdir(parents=True, exist_ok=True)
    (cfg['output'] / 'charts').mkdir(exist_ok=True)
    return cfg


def load_dataset(cfg):
    """Match actual Parquet metadata, fixed reference and original E1 split."""
    started = perf_counter()
    paths = sorted(cfg['data'].glob('*_subset/part-*.parquet'))
    if not paths:
        raise ValueError('No Parquet shards found.')
    cfg['parquet_paths'] = paths
    reference_path = cfg['data'] / 'manifests/clips_40000.csv.gz'
    assert file_hash(reference_path) == file_hash(cfg['reference']) == REFERENCE_SHA
    reference = pd.read_csv(reference_path, low_memory=False)
    columns = ['clip_id', 'source', 'speaker_id', 'accent_raw', 'accent_group',
               'is_singaporean', 'duration_sec', 'sentence', 'sampling_rate', 'split']
    frame = pd.concat([pq.read_table(path, columns=columns).to_pandas()
                       for path in paths], ignore_index=True)

    def canonical(data):
        data = data.copy().rename(columns={'clip_id': 'source_clip_id'})
        data['clip_id'] = data.source + '::' + data.source_clip_id
        data['speaker_group'] = data.source + '::' + data.speaker_id
        data['label'] = data.is_singaporean.astype(np.int64)
        data['sampling_rate'] = data.sampling_rate.astype(np.int64)
        data['split'] = data['split'].replace({'val': 'validation'})
        return data.set_index('clip_id').sort_index()

    frame, reference = canonical(frame), canonical(reference)
    assert len(frame) == 40000 and frame.index.is_unique
    assert frame.label.value_counts().to_dict() == {0: 20000, 1: 20000}
    compare = META + ['accent_raw']
    pd.testing.assert_frame_equal(frame[compare], reference[compare], check_dtype=False)
    assert frame.sentence.fillna('').equals(reference.sentence.fillna(''))
    assert np.allclose(frame.duration_sec, reference.duration_sec, atol=1/16000, rtol=0)
    assert frame.sampling_rate.eq(16000).all() and frame.duration_sec.between(3, 6).all()
    snapshot = frame[META].reset_index()
    original = pd.read_csv(cfg['e1'] / 'speaker_splits.csv')
    pd.testing.assert_frame_equal(snapshot, original, check_dtype=False)
    dataset_id = digest(snapshot.to_dict(orient='records'))
    assert dataset_id == DATASET_ID
    assert frame.groupby('speaker_group').split.nunique().eq(1).all()
    assert frame.groupby('speaker_group').label.nunique().eq(1).all()
    assert set(frame.split) == {'train', 'validation', 'test'}
    frame['decoded_sha256'] = reference.decoded_sha256
    frame['samples'] = reference.samples.astype(int)
    frame = frame.reset_index()
    frame.to_csv(cfg['output'] / 'dataset_manifest.csv', index=False)
    split = {name: np.flatnonzero(frame.split.eq(name))
             for name in ['train', 'validation', 'test']}
    split['development'] = np.sort(np.r_[split['train'], split['validation']])
    assert [len(split[k]) for k in ['train', 'validation', 'test']] == [27797, 6084, 6119]
    summary = frame.groupby(['split', 'label']).agg(
        clips=('clip_id', 'size'), speakers=('speaker_group', 'nunique')).reset_index()
    summary.to_csv(cfg['output'] / 'split_summary.csv', index=False)
    cfg['dataset_id'] = dataset_id
    write_json(cfg['output'] / 'dataset_identity.json', {
        'dataset_id': dataset_id, 'reference_manifest_sha256': REFERENCE_SHA,
        'speaker_overlap': 0, 'metadata_seconds': perf_counter() - started,
        'original_split_sha256': file_hash(cfg['e1'] / 'speaker_splits.csv'),
        'parquet_files': [{'path': str(p), 'bytes': p.stat().st_size} for p in paths]})
    print('Metadata matches the final fusion data. Speaker overlap: 0.', flush=True)
    return frame, split, summary


def inspect_clip(record):
    blob = record['audio_flac']
    with sf.SoundFile(io.BytesIO(blob)) as stream:
        rate, channels, subtype = stream.samplerate, stream.channels, stream.subtype
        pcm = stream.read(dtype='int16', always_2d=True)
    if rate != 16000 or channels != 1 or subtype != 'PCM_16':
        raise ValueError(f'Unexpected audio format for {record["clip_id"]}.')
    if not 48000 <= len(pcm) <= 96000:
        raise ValueError('Expected 3–6 seconds of real audio.')
    return dict(clip_id=record['source'] + '::' + record['clip_id'],
                encoded_sha256=hashlib.sha256(blob).hexdigest(),
                decoded_sha256=hashlib.sha256(str(rate).encode() + pcm.astype('<i2').tobytes()).hexdigest(),
                samples=len(pcm), sampling_rate=rate)


def audit_audio(cfg, frame):
    """Decode every clip on every run; do not trust a stale verification file."""
    started, rows = perf_counter(), []
    with ThreadPoolExecutor(max_workers=cfg['workers']) as pool:
        for path in cfg['parquet_paths']:
            for batch in pq.ParquetFile(path).iter_batches(
                    batch_size=256, columns=['clip_id', 'source', 'audio_flac']):
                rows.extend(pool.map(inspect_clip, batch.to_pylist()))
            print(f'Audio audit: {len(rows):,}/40,000 decoded', flush=True)
    audit = pd.DataFrame(rows).set_index('clip_id').sort_index()
    assert audit.index.equals(pd.Index(frame.clip_id, name='clip_id'))
    assert np.array_equal(audit.decoded_sha256, frame.decoded_sha256)
    assert np.array_equal(audit.samples, frame.samples)
    assert audit.decoded_sha256.nunique() == 40000, 'Duplicate decoded audio.'
    base = json.loads((cfg['e1'] / 'current_cache_metadata.json').read_text())
    original = pd.DataFrame(base['audio']).set_index('clip_id').sort_index()
    assert original.index.equals(audit.index)
    same_encoded = int((original.sha256 == audit.encoded_sha256).sum())
    audit.to_csv(cfg['output'] / 'audio_audit.csv')
    result = dict(status='passed', clips_decoded=len(audit), decoded_hash_matches=40000,
                  encoded_hash_matches_original_cache=same_encoded,
                  duplicate_decoded_audio=0, seconds=perf_counter()-started,
                  reference_manifest_sha256=REFERENCE_SHA,
                  note='Decoded PCM matches the fixed original reference. No new audio preparation applied.')
    write_json(cfg['output'] / 'audio_audit.json', result)
    cfg['audio_audit_passed'] = True
    print(f'All 40,000 decoded audio hashes match. Encoded FLAC matches: {same_encoded:,}.', flush=True)
    return result


def load_whisper(cfg, frame):
    """Use exactly the fusion embeddings after full waveform and identity checks."""
    assert cfg.get('audio_audit_passed'), 'Decode and audit the new dataset first.'
    original_metadata = json.loads((cfg['fusion'] / 'model_metadata.json').read_text())
    original_path = cfg['fusion'] / 'feature_cache' / (
        'whisper_' + original_metadata['whisper_provenance']['cache_id'] + '.npz')
    fresh_path = cfg['output'] / 'whisper_embeddings_fresh.npz'
    if cfg['rebuild_whisper']:
        X, provenance = extract_whisper(cfg, frame)
        np.savez_compressed(fresh_path, features=X, clip_ids=frame.clip_id.to_numpy(dtype=str),
                            labels=frame.label.to_numpy(), groups=frame.speaker_group.to_numpy(dtype=str),
                            provenance_json=np.asarray(json.dumps(provenance)))
        path = fresh_path
        mode = 'fresh encoder extraction'
    else:
        path = original_path
        with np.load(path, allow_pickle=False) as cache:
            for key, expected in [('clip_ids', frame.clip_id), ('labels', frame.label),
                                  ('groups', frame.speaker_group)]:
                assert np.array_equal(cache[key], expected), key
            X = cache['features'].copy()
            provenance = json.loads(str(cache['provenance_json'].item()))
        assert provenance == original_metadata['whisper_provenance']
        assert provenance['dataset_id'] == cfg['dataset_id']
        assert provenance['model'] == MODEL and provenance['resolved_revision'] == REVISION
        assert provenance['pooling'] == 'mean_valid_encoder_frames_v1'
        mode = 'verified original fusion cache'
    assert X.shape == (40000, 512) and np.isfinite(X).all()
    provenance = dict(provenance, reuse_mode=mode, cache_path=str(path),
                      cache_file_sha256=file_hash(path), frozen=True,
                      decoded_audio_matches_reference=True)
    write_json(cfg['output'] / 'whisper_provenance.json', provenance)
    print('Whisper features:', X.shape, '|', mode, flush=True)
    return X, provenance


def extract_whisper(cfg, frame):
    """Optional exact-method rerun. No trimming, RMS change or fine-tuning."""
    import torch
    import transformers
    from transformers import WhisperFeatureExtractor, WhisperModel
    extractor = WhisperFeatureExtractor.from_pretrained(MODEL, revision=REVISION,
                                                        token=False, local_files_only=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    whisper = WhisperModel.from_pretrained(MODEL, revision=REVISION,
                                          token=False, local_files_only=True).to(device).eval()
    whisper.requires_grad_(False)
    encoder = whisper.get_encoder()
    # Float32 matches the original comparison notebook, with no mixed precision.
    X = np.empty((len(frame), 512), dtype=np.float32)
    offsets = {key: i for i, key in enumerate(frame.clip_id)}
    seen = set()
    started = perf_counter()
    batch_size = cfg['batch_size'] if device.type == 'cuda' else 4
    with torch.inference_mode():
        for path in cfg['parquet_paths']:
            for batch in pq.ParquetFile(path).iter_batches(
                    batch_size=batch_size, columns=['clip_id', 'source', 'audio_flac']):
                records = batch.to_pylist()
                audio = [sf.read(io.BytesIO(r['audio_flac']), dtype='float32')[0] for r in records]
                mel = extractor(audio, sampling_rate=16000, return_tensors='pt', do_normalize=False)
                hidden = encoder(mel.input_features.to(device)).last_hidden_state
                lengths = torch.tensor([(len(a)+319)//320 for a in audio], device=device)
                mask = torch.arange(hidden.shape[1], device=device)[None, :] < lengths[:, None]
                pooled = (hidden * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
                ids = [r['source']+'::'+r['clip_id'] for r in records]
                X[[offsets[k] for k in ids]] = pooled.float().cpu().numpy()
                seen.update(ids)
            print(f'Whisper extraction: {len(seen):,}/40,000', flush=True)
    assert len(seen) == len(frame)
    provenance = dict(model=MODEL, resolved_revision=REVISION, width=512,
                      dataset_id=cfg['dataset_id'], pooling='mean_valid_encoder_frames_v1',
                      sample_rate=16000, seconds=perf_counter()-started,
                      torch_version=torch.__version__, transformers_version=transformers.__version__)
    del whisper, encoder
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return X, provenance


def make_model(name, c):
    steps = [('scale', StandardScaler())]
    if name == MODELS[1]:
        steps += [('pca', PCA(n_components=32, random_state=SEED)),
                  ('component_scale', StandardScaler())]
    elif name != MODELS[0]:
        raise ValueError(name)
    return Pipeline(steps + [('classifier', SVC(
        C=c, kernel='rbf', gamma='scale', class_weight='balanced',
        cache_size=2048, random_state=SEED))])


def prediction_frame(model, X, frame, rows, name, stage, **extra):
    values = X[rows]
    result = frame.iloc[rows][['clip_id', 'speaker_group', 'source', 'accent_group', 'label']].copy()
    result['model'], result['stage'] = name, stage
    result['prediction'] = model.predict(values)
    result['decision_score'] = model.decision_function(values)
    for key, value in extra.items():
        result[key] = value
    return result.reset_index(drop=True)


def metrics(frame):
    y, p = frame.label.to_numpy(), frame.prediction.to_numpy()
    cm = confusion_matrix(y, p, labels=[0, 1])
    return dict(accuracy=float(accuracy_score(y, p)),
                balanced_accuracy=float(balanced_accuracy_score(y, p)),
                macro_f1=float(f1_score(y, p, average='macro', zero_division=0)),
                roc_auc=float(roc_auc_score(y, frame.decision_score)),
                sg_recall=float(cm[1, 1]/cm[1].sum()), non_sg_fpr=float(cm[0, 1]/cm[0].sum()),
                errors=int(np.sum(y != p)), false_positive=int(cm[0, 1]), false_negative=int(cm[1, 0]),
                confusion_matrix=cm.tolist())


def flat_metrics(frame):
    return {k: v for k, v in metrics(frame).items() if k != 'confusion_matrix'}


def select_models(cfg, X, frame, split):
    train, val = split['train'], split['validation']
    y = frame.label.to_numpy()
    trials, selected, selected_predictions = [], {}, []
    for name in MODELS:
        candidates = []
        for c in cfg['c_values']:
            started = perf_counter()
            model = make_model(name, c).fit(X[train], y[train])
            fit_seconds = perf_counter() - started
            pred = prediction_frame(model, X, frame, val, name, 'validation')
            row = dict(model=name, C=c, fit_seconds=fit_seconds, **flat_metrics(pred))
            trials.append(row)
            candidates.append((row, pred))
            print(f'{name} C={c:g}: validation BA {row["balanced_accuracy"]:.4%}; fit {fit_seconds:.1f}s', flush=True)
        row, pred = max(candidates, key=lambda item: (
            item[0]['balanced_accuracy'], item[0]['macro_f1'], -item[0]['C']))
        selected[name] = row['C']
        selected_predictions.append(pred)
    table = pd.DataFrame(trials)
    table.to_csv(cfg['output'] / 'validation_C_trials.csv', index=False)
    pd.concat(selected_predictions).to_csv(cfg['output'] / 'validation_predictions.csv', index=False)
    write_json(cfg['output'] / 'selected_C.json', selected)
    return selected, table


def speaker_folds(cfg, X, frame, split, selected):
    if not cfg['run_folds']:
        return pd.DataFrame()
    dev = split['development']
    y, groups = frame.label.to_numpy(), frame.speaker_group.to_numpy()
    splitter = StratifiedGroupKFold(n_splits=cfg['folds'], shuffle=True, random_state=SEED)
    rows, predictions = [], []
    original = pd.read_csv(cfg['fusion'] / 'speaker_fold_predictions.csv')
    for fold, (a, b) in enumerate(splitter.split(X[dev], y[dev], groups[dev]), start=1):
        fit, held = dev[a], dev[b]
        assert set(groups[fit]).isdisjoint(groups[held])
        expected = original.loc[(original.model == 'Fusion SVM') & (original.fold == fold)]
        assert set(expected.clip_id) == set(frame.clip_id.iloc[held]), 'Original fold differs.'
        for name in MODELS:
            started = perf_counter()
            model = make_model(name, selected[name]).fit(X[fit], y[fit])
            pred = prediction_frame(model, X, frame, held, name, 'speaker fold', fold=fold)
            rows.append(dict(model=name, fold=fold, seconds=perf_counter()-started, **flat_metrics(pred)))
            predictions.append(pred)
            print(f'Fold {fold}: {name} BA={rows[-1]["balanced_accuracy"]:.4%}', flush=True)
    predictions = pd.concat(predictions, ignore_index=True)
    for name in MODELS:
        part = predictions[predictions.model.eq(name)]
        assert len(part) == len(dev) and part.clip_id.nunique() == len(dev)
    predictions.to_csv(cfg['output'] / 'speaker_fold_predictions.csv', index=False)
    table = pd.DataFrame(rows)
    reference_rows = [dict(model=name, fold=int(fold), **flat_metrics(part.rename(columns={'true_label':'label'})))
                      for (name, fold), part in original.groupby(['model', 'fold'])]
    table = pd.concat([table, pd.DataFrame(reference_rows)], ignore_index=True)
    table.to_csv(cfg['output'] / 'speaker_fold_scores.csv', index=False)
    return table


def final_test(cfg, X, frame, split, selected):
    dev, test = split['development'], split['test']
    y = frame.label.to_numpy()
    predictions, timing = [], []
    for index, name in enumerate(MODELS):
        started = perf_counter()
        model = make_model(name, selected[name]).fit(X[dev], y[dev])
        fit_seconds = perf_counter() - started
        assert model.named_steps['scale'].n_samples_seen_ == len(dev)
        assert np.allclose(model.named_steps['scale'].mean_, X[dev].mean(0, dtype=np.float64))
        started = perf_counter()
        pred = prediction_frame(model, X, frame, test, name, 'test')
        timing.append(dict(model=name, final_fit_seconds=fit_seconds,
                           test_predict_and_score_seconds=perf_counter()-started))
        joblib.dump(dict(model=model, selected_C=selected[name], dataset_id=cfg['dataset_id'],
                         encoder=MODEL, revision=REVISION, frozen=True),
                    cfg['output'] / ('whisper_full512_svm.joblib' if index == 0 else 'whisper_pca32_svm.joblib'))
        predictions.append(pred)
        print(name, 'test:', flat_metrics(pred), flush=True)
    original = pd.read_csv(cfg['fusion'] / 'test_predictions.csv')
    expected = frame.iloc[test].set_index('clip_id')
    for name in REFERENCES:
        part = original[original.model.eq(name)].set_index('clip_id').reindex(expected.index)
        assert len(part) == 6119 and not part.isna().any().any()
        assert np.array_equal(part.true_label, expected.label)
        assert np.array_equal(part.speaker_group, expected.speaker_group)
        assert np.array_equal(part.source, expected.source)
        part['label'], part['accent_group'] = part.true_label, expected.accent_group
        part = part.drop(columns='true_label').reset_index()
        predictions.append(part)
    all_predictions = pd.concat(predictions, ignore_index=True)
    all_predictions.to_csv(cfg['output'] / 'test_predictions.csv', index=False)
    all_predictions[all_predictions.label.ne(all_predictions.prediction)].to_csv(
        cfg['output'] / 'test_errors.csv', index=False)
    pd.DataFrame(timing).to_csv(cfg['output'] / 'timing.csv', index=False)
    return all_predictions


def speaker_bootstrap(cfg, predictions):
    """Paired class-stratified whole-speaker bootstrap for BA and macro F1."""
    names = MODELS + REFERENCES
    parts = {name: predictions[predictions.model.eq(name)].set_index('clip_id').sort_index()
             for name in names}
    reference = parts[names[0]][['speaker_group', 'label']]
    for part in parts.values():
        pd.testing.assert_frame_equal(part[['speaker_group', 'label']], reference)
    speakers = reference.groupby('speaker_group').label.first().sort_index()
    assert reference.groupby('speaker_group').label.nunique().eq(1).all()
    groups = speakers.index.get_indexer(reference.speaker_group)
    rng = np.random.default_rng(SEED)
    weights = np.zeros((cfg['bootstrap'], len(speakers)), dtype=np.int32)
    for label in [0, 1]:
        ids = np.flatnonzero(speakers.eq(label))
        draws = ids[rng.integers(len(ids), size=(cfg['bootstrap'], len(ids)))]
        np.add.at(weights, (np.arange(cfg['bootstrap'])[:, None], draws), 1)

    def from_cm(cm):
        tn, fp, fn, tp = np.asarray(cm).T
        return dict(balanced_accuracy=.5*(tp/(tp+fn) + tn/(tn+fp)),
                    macro_f1=.5*(2*tp/(2*tp+fp+fn) + 2*tn/(2*tn+fp+fn)))

    samples, scores, rows = {}, {}, []
    for name, part in parts.items():
        cm = np.zeros((len(speakers), 4), dtype=np.int64)
        np.add.at(cm, (groups, 2*part.label.to_numpy()+part.prediction.to_numpy()), 1)
        samples[name] = from_cm(weights @ cm)
        result = metrics(part)
        result['speaker_bootstrap_ci95'] = {metric: np.quantile(draws, [.025, .975]).tolist()
                                            for metric, draws in samples[name].items()}
        scores[name] = result
        rows.append(dict(model=name, **{k:v for k,v in result.items()
                                       if k not in ['confusion_matrix', 'speaker_bootstrap_ci95']},
                         ba_ci95_low=result['speaker_bootstrap_ci95']['balanced_accuracy'][0],
                         ba_ci95_high=result['speaker_bootstrap_ci95']['balanced_accuracy'][1],
                         f1_ci95_low=result['speaker_bootstrap_ci95']['macro_f1'][0],
                         f1_ci95_high=result['speaker_bootstrap_ci95']['macro_f1'][1]))
    paired = {}
    for control in MODELS:
        key = 'Fusion SVM minus ' + control
        paired[key] = {}
        for metric in ['balanced_accuracy', 'macro_f1']:
            delta = samples['Fusion SVM'][metric]-samples[control][metric]
            paired[key][metric] = dict(difference=scores['Fusion SVM'][metric]-scores[control][metric],
                                        ci95=np.quantile(delta, [.025, .975]).tolist())
        paired[key]['errors'] = dict(
            fusion_only_correct=int(((parts['Fusion SVM'].prediction == reference.label) &
                                     (parts[control].prediction != reference.label)).sum()),
            whisper_only_correct=int(((parts['Fusion SVM'].prediction != reference.label) &
                                      (parts[control].prediction == reference.label)).sum()))
    table = pd.DataFrame(rows)
    table.to_csv(cfg['output'] / 'test_comparison.csv', index=False)
    write_json(cfg['output'] / 'test_metrics.json', scores)
    write_json(cfg['output'] / 'paired_differences.json', {
        'method': '2,000 paired whole-speaker draws, stratified by class; seed 42',
        'scope': 'Fixed fitted models and split; not training or model-selection uncertainty.',
        'comparisons': paired})
    return table, scores, paired


def subgroup_metrics(cfg, predictions):
    rows = []
    for category in ['source', 'accent_group']:
        for (name, group), part in predictions.groupby(['model', category]):
            rows.append(dict(model=name, category=category, group=group, clips=len(part),
                             speakers=part.speaker_group.nunique(),
                             errors=int(part.label.ne(part.prediction).sum()),
                             accuracy=float(part.label.eq(part.prediction).mean()),
                             positive_class=bool(part.label.eq(1).all())))
    result = pd.DataFrame(rows)
    result.to_csv(cfg['output'] / 'subgroup_scores.csv', index=False)
    return result


def save_chart(cfg, fig, filename):
    fig.tight_layout()
    fig.savefig(cfg['output'] / 'charts' / filename, dpi=160, bbox_inches='tight')
    plt.show()
    plt.close(fig)


def charts(cfg, trials, folds, table, scores, paired, subgroups):
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, color in zip(MODELS, COLORS):
        part = trials[trials.model.eq(name)]
        ax.plot(part.C, 100*part.balanced_accuracy, 'o-', label=name, color=color)
    ax.set(xscale='log', xlabel='SVM C', ylabel='Balanced accuracy (%)',
           title='Choose C with validation speakers only')
    ax.legend()
    save_chart(cfg, fig, 'validation_C.png')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    labels = ['Whisper full512', 'Whisper PCA32', 'Acoustic65', 'Fusion65+32']
    x = np.arange(4)
    values = table.balanced_accuracy.to_numpy()*100
    error = np.vstack([values-table.ba_ci95_low.to_numpy()*100,
                       table.ba_ci95_high.to_numpy()*100-values])
    axes[0].errorbar(x, values, yerr=error, fmt='none', ecolor='#333333', capsize=5)
    axes[0].scatter(x, values, c=COLORS, s=70, zorder=3)
    axes[0].set(xticks=x, xticklabels=labels, ylabel='Balanced accuracy (%)',
                title='Test score with 95% speaker intervals')
    axes[0].tick_params(axis='x', rotation=20)
    for i, v in enumerate(values):
        axes[0].annotate(f'{v:.2f}%', (i, v), xytext=(0, 10), textcoords='offset points', ha='center', fontsize=9)
    bars = axes[1].bar(labels, table.errors, color=COLORS)
    axes[1].bar_label(bars, padding=3)
    axes[1].set(ylabel='Misclassified clips', title='Errors on the same 6,119 test clips')
    axes[1].set_ylim(0, max(table.errors)*1.15)
    axes[1].tick_params(axis='x', rotation=20)
    save_chart(cfg, fig, 'test_comparison.png')

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8))
    for ax, name, title in zip(axes, MODELS+REFERENCES, labels):
        cm = np.asarray(scores[name]['confusion_matrix'])
        ax.imshow(cm, cmap='Blues', vmin=0, vmax=cm.max())
        for (r, c), v in np.ndenumerate(cm):
            ax.text(c, r, str(v), ha='center', va='center', color='white' if v > cm.max()/2 else 'black')
        ax.set(xticks=[0,1], yticks=[0,1], xticklabels=['Non-SG','SG'], yticklabels=['Non-SG','SG'],
               xlabel='Predicted', ylabel='Actual', title=title)
    save_chart(cfg, fig, 'confusion_matrices.png')

    fig, ax = plt.subplots(figsize=(8, 4))
    for i, (key, result) in enumerate(paired.items()):
        item = result['balanced_accuracy']
        value, lo, hi = 100*item['difference'], *[100*v for v in item['ci95']]
        ax.errorbar(value, i, xerr=[[value-lo],[hi-value]], fmt='o', capsize=5, color=COLORS[i])
        ax.annotate(f'{value:+.3f} [{lo:+.3f}, {hi:+.3f}]', (value, i),
                    xytext=(0, 13), textcoords='offset points', ha='center', fontsize=9)
    ax.axvline(0, color='#666666', linestyle='--')
    ax.set(yticks=[0,1], yticklabels=['Fusion minus full512', 'Fusion minus PCA32'],
           xlabel='Balanced accuracy difference (percentage points)', ylim=(-.5,1.5),
           title='Paired comparison on the same test speakers')
    save_chart(cfg, fig, 'paired_gain.png')

    if not folds.empty:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        for name, color, label in zip(MODELS+REFERENCES, COLORS, labels):
            part = folds[folds.model.eq(name)].sort_values('fold')
            ax.plot(part.fold, 100*part.balanced_accuracy, 'o-', label=label, color=color)
        ax.set(xticks=[1,2,3], xlabel='Held-out development speaker fold', ylabel='Balanced accuracy (%)',
               title='Fixed-C checks across three speaker folds')
        ax.legend(ncol=2)
        save_chart(cfg, fig, 'speaker_folds.png')

    source = subgroups[subgroups.category.eq('source')].copy()
    fig, ax = plt.subplots(figsize=(10, 4.5))
    groups = sorted(source.group.unique())
    for i, (name, color, label) in enumerate(zip(MODELS+REFERENCES, COLORS, labels)):
        part = source[source.model.eq(name)].set_index('group').reindex(groups)
        ax.bar(np.arange(len(groups))+(i-1.5)*.2, 100*(1-part.accuracy), .2, color=color, label=label)
    ax.set(xticks=np.arange(len(groups)), xticklabels=groups, ylabel='Error rate (%)',
           title='Test errors by recording source (classes differ by source)')
    ax.legend(ncol=2)
    save_chart(cfg, fig, 'source_errors.png')


def write_report(cfg, table, scores, paired, provenance, folds):
    full = scores[MODELS[0]]
    delta = paired['Fusion SVM minus '+MODELS[0]]['balanced_accuracy']
    lo, hi = delta['ci95']
    interpretation = ('The paired interval includes zero. This test does not establish a fusion gain over full Whisper.'
                      if lo <= 0 <= hi else
                      'The paired interval is above zero. Fusion has higher measured balanced accuracy on this fixed test split.'
                      if lo > 0 else
                      'The paired interval is below zero. Full Whisper has higher measured balanced accuracy on this fixed test split.')
    selected = json.loads((cfg['output'] / 'selected_C.json').read_text())
    lines = ['# Frozen Whisper on the final fusion dataset', '',
             'Completed on 28 October 2026. The encoder weights stayed fixed. Only the SVM heads were trained.', '',
             'Data: 40,000 original prepared clips. Train: 27,797; validation: 6,084; test: 6,119.',
             'Test: 3,085 non-Singapore clips and 3,034 Singapore clips from 1,482 speakers. Speaker overlap is zero.',
             'All 40,000 decoded audio hashes match the fixed reference. No extra trimming or volume change was applied.',
             f'Embeddings: {provenance["reuse_mode"]}. Encoder: `{MODEL}` at `{REVISION}`.', '',
             '| Model | Validation C | Test BA | Macro F1 | ROC AUC | FP | FN | Errors |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in table.to_dict(orient='records'):
        name = row['model']
        c = selected.get(name, 1. if name == 'SVM65' else 10.)
        lines.append(f'| {name} | {c:g} | {row["balanced_accuracy"]:.2%} | {row["macro_f1"]:.4f} | '
                     f'{row["roc_auc"]:.6f} | {row["false_positive"]} | {row["false_negative"]} | {row["errors"]} |')
    lines += ['', '![Test results](../outputs/frozen_whisper_40000_20261028/charts/test_comparison.png)', '',
              f'Fusion minus full Whisper: **{100*delta["difference"]:+.3f} percentage points** of balanced accuracy.',
              f'Paired 95% speaker interval: **[{100*lo:+.3f}, {100*hi:+.3f}] percentage points**.',
              interpretation, '',
              '## Test rules', '',
              '- The SVM C grid is 0.01, 0.1, 1, 10. Validation balanced accuracy selects C. Ties use macro F1, then smaller C.',
              '- Standard scaling, PCA and the SVM are fitted inside each fitting partition.',
              '- Full Whisper uses 512 inputs. The PCA control uses 32 scaled components, as in the fusion branch.',
              '- Chosen settings are fixed before testing. The final heads refit on training plus validation clips.',
              '- `predict()` supplies class labels. Decision scores supply ROC AUC. Scores are not calibrated probabilities.',
              '- Acoustic65 and Fusion SVM predictions come from the saved 24 October uncalibrated experiment.',
              '- All four models use identical test clip IDs, labels and speakers.',
              '- The 2,000 bootstrap draws resample complete speakers within each class. All models share each draw.', '',
              '## Limits', '',
              'The test set was already used in the earlier experiments. This is a retrospective control, not a new untouched test set.',
              'All positive clips come from MNSC. Negative clips come from other sources. These results do not isolate accent from recording source.',
              'The speaker folds use C selected on the fixed validation set. They are stability checks, not nested model selection.',
              'The saved fusion model uses the original selected65 acoustic inputs. This run does not repeat acoustic feature selection inside folds.',
              'The bootstrap intervals describe this split and these fitted models. They do not include training or selection uncertainty.',
              'A gain from fusion does not prove that acoustic features add accent information independent of recording conditions.', '',
              '## Files and rerun', '',
              '- Notebook: `notebooks/E2_01_whisper_classifier_beginner.ipynb`.',
              '- Results: `outputs/frozen_whisper_40000_20261028/`.',
              '- Earlier 2,000-clip notebook: `notebooks/E2_01_whisper_classifier_matched_2000.ipynb`.', '',
              '```bash',
              '/home/mh/miniconda3/envs/singlish-py314/bin/python scripts/run_frozen_whisper_40k.py',
              '```', '']
    (cfg['root'] / 'results/FROZEN_WHISPER_40000_20261028.md').write_text('\n'.join(lines))
    versions = {name: version(name) for name in ['numpy','pandas','scikit-learn','torch','transformers','pyarrow','soundfile','nbclient']}
    write_json(cfg['output'] / 'run_metadata.json', {
        'python': platform.python_version(), 'packages': versions, 'seed': SEED,
        'dataset_id': cfg['dataset_id'], 'model': MODEL, 'revision': REVISION,
        'selected_C': selected, 'folds': cfg['folds'] if cfg['run_folds'] else 0,
        'bootstrap_repeats': cfg['bootstrap'], 'whisper_provenance': provenance,
        'fusion_reference_predictions_sha256': file_hash(cfg['fusion'] / 'test_predictions.csv'),
        'helper_sha256': file_hash(Path(__file__)),
        'raw_fusion_comparison': True, 'calibrated_probability_comparison': False})
    return interpretation
