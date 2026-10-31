"""Check acoustic feature parity on held-out Parquet audio, then exercise the API.

Run from the repository root; requires pyarrow for this offline check only.
Study waveforms are already processed and must bypass serving preprocessing
when comparing against the cached training features.
"""
import argparse
import csv
from io import BytesIO
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from fastapi.testclient import TestClient
from mvp.app import app
from mvp.audio import decode_audio
from mvp.features import FEATURE_NAMES, extract_features

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('study_root', type=Path, help='E1 project containing data/ and outputs/')
parser.add_argument('--data-root', type=Path, help='Shared Parquet data directory, if outside the study project')
parser.add_argument('--output', type=Path)
args = parser.parse_args()
study = args.study_root / 'outputs/e1_feature_study_accent_40000'
with np.load(study / 'acoustic_features_91.npz', allow_pickle=False) as cache:
    assert cache['feature_names'].tolist() == FEATURE_NAMES
    features = cache['features'].copy()
    ids = cache['clip_ids'].tolist()
with (study / 'speaker_splits.csv').open() as stream:
    rows = list(csv.DictReader(stream))
selected = []
for label in ['0', '1']:
    candidates = [r for r in rows if r['split'] == 'test' and r['label'] == label][:3]
    assert len(candidates) == 3
    selected.extend(candidates)
wanted = {row['clip_id'] for row in selected}
audio = {}
data_root = args.data_root or args.study_root / 'data'
assert data_root.is_dir(), f'Missing {data_root}; use --data-root for shared study audio'
for path in sorted(data_root.glob('*subset/*.parquet')):
    for batch in pq.ParquetFile(path).iter_batches(columns=['source', 'clip_id', 'audio_flac']):
        for row in batch.to_pylist():
            identity = row['source'] + '::' + row['clip_id']
            if identity in wanted:
                audio[identity] = row['audio_flac']
    if len(audio) == len(wanted):
        break
assert set(audio) == wanted
records = []
with TestClient(app) as client:
    ml = client.app.state.models.ml
    for row in selected:
        y, rate = sf.read(BytesIO(audio[row['clip_id']]), dtype='float32')
        assert rate == 16000 and y.ndim == 1
        vector = extract_features(y)
        cached = features[ids.index(row['clip_id'])]
        np.testing.assert_allclose(vector, cached, rtol=1e-5, atol=1e-6)
        expected = int(ml.pipeline.predict_proba(cached[ml.columns].reshape(1, -1))[0, 1] >= 0.5)
        actual = ml.predict(vector)['class_id']
        assert actual == expected
        stream = BytesIO()
        sf.write(stream, y, rate, format='WAV', subtype='PCM_16')
        data = stream.getvalue()
        # The public API processes raw uploads; these inputs are deliberately
        # reprocessed here, so their API features need not equal the study cache.
        processed, _, _ = decode_audio(data)
        api_prediction = ml.predict(extract_features(processed))
        api_expected = api_prediction['class_id']
        started = perf_counter()
        response = client.post('/api/classify', files={'audio': ('study.wav', data)})
        elapsed = perf_counter() - started
        assert response.status_code == 200, response.text
        assert response.json()['models']['ml']['prediction'] == api_prediction
        records.append({'clip_id': row['clip_id'], 'true_class': int(row['label']),
            'cached_prediction': expected, 'api_prediction': api_expected,
            'api_probability': api_prediction['probability'],
            'feature_max_abs_difference': float(np.max(np.abs(vector - cached))),
            'warm_request_seconds': round(elapsed, 4)})
report = {'model_version': ml.manifest['model_version'],
    'checked_recordings': len(records), 'feature_rtol': 1e-5, 'feature_atol': 1e-6,
    'api_note': 'API reprocesses uploads; cache parity uses decoded study audio directly.',
    'records': records}
if args.output:
    args.output.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
