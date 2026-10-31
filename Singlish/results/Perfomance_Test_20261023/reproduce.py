"""Run the frozen models on the fixed evaluation clips."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import sys
from urllib.request import urlretrieve

import joblib
import numpy as np
import pandas as pd
import soundfile as sf
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from mvp.features import extract_features
from mvp.embeddings import WhisperEmbedding
from mvp.models import fitted_svm


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@torch.inference_mode()
def features(waves, size, whisper):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.bfloat16 if device == 'cuda' else torch.float32
    whisper.encoder.to(device=device, dtype=dtype)
    values = []
    for start in range(0, len(waves), size):
        batch = waves[start:start + size]
        inputs = whisper.extractor(batch, sampling_rate=16000, do_normalize=False,
                                   device=device, return_tensors='pt').input_features.to(device=device, dtype=dtype)
        hidden = whisper.encoder(input_features=inputs).last_hidden_state.float()
        lengths = torch.tensor([(len(w) + 319) // 320 for w in batch], device=device).clamp(1, hidden.shape[1])
        mask = torch.arange(hidden.shape[1], device=device)[None, :] < lengths[:, None]
        values.append(((hidden * mask[:, :, None]).sum(1) / lengths[:, None]).cpu().numpy())
    return np.vstack(values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=Path('data/release_comparison'))
    parser.add_argument('--models', type=Path, default=Path('mvp/models'))
    parser.add_argument('--output', type=Path, default=Path('outputs/release_comparison_reproduced'))
    parser.add_argument('--cache', type=Path)
    args = parser.parse_args()
    metadata = json.loads((HERE / 'metadata.json').read_text())
    assert sha(HERE.parents[1] / 'mvp/features.py') == metadata['code_sha256']['mvp/features.py']
    cache = args.cache or args.output / 'cache'
    cache.mkdir(parents=True, exist_ok=True)
    args.models.mkdir(parents=True, exist_ok=True)
    expected = pd.read_csv(HERE / 'predictions.csv').set_index(['dataset', 'clip_id'])
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
    models, whisper, results = {}, None, []
    for spec in metadata['models']:
        path = args.models / spec['artifact']
        sidecar = path.with_name(path.stem + '_metadata.json')
        for local, url, checksum in [(path, spec['url'], spec['sha256']),
                (sidecar, spec['url'].removesuffix('.joblib') + '_metadata.json', spec['metadata_sha256'])]:
            if not local.exists():
                urlretrieve(url, local)
            assert sha(local) == checksum, local
        models[spec['id']] = joblib.load(path), json.loads(sidecar.read_text())
    for dataset, info in metadata['datasets'].items():
        manifest = HERE / 'manifests' / (dataset + '.csv')
        assert sha(manifest) == info['manifest_sha256']
        rows = pd.read_csv(manifest, keep_default_na=False).sort_values('clip_id')
        ids = rows.clip_id.to_numpy(dtype=str)
        cached = cache / (dataset + '.npz')
        if cached.exists():
            with np.load(cached, allow_pickle=False) as z:
                assert z['clip_ids'].tolist() == ids.tolist()
                assert str(z['manifest_sha256'].item()) == info['manifest_sha256']
                X, W = z['acoustic'].copy(), z['whisper'].copy()
        else:
            folder = args.data_root / dataset
            if any(folder.glob('*.parquet')):
                stored = pd.concat([pd.read_parquet(p) for p in sorted(folder.glob('*.parquet'))]).set_index('clip_id')
                blobs = stored.loc[ids, 'audio_flac'].tolist()
            else:
                blobs = [(folder / (r.source_clip_id + '.flac')).read_bytes() for r in rows.itertuples()]
            waves = []
            for blob, row in zip(blobs, rows.itertuples()):
                pcm, rate = sf.read(io.BytesIO(blob), dtype='int16', always_2d=True)
                assert hashlib.sha256(str(rate).encode() + pcm.astype('<i2').tobytes()).hexdigest() == row.pcm_sha256
                wave, rate = sf.read(io.BytesIO(blob), dtype='float32')
                assert rate == 16000 and wave.ndim == 1
                waves.append(wave)
            with ThreadPoolExecutor(max_workers=4) as pool:
                X = np.stack(list(pool.map(extract_features, waves)))
            if whisper is None:
                whisper = WhisperEmbedding(metadata['whisper']['model'], metadata['whisper']['revision'])
            batch = metadata['whisper']['batch_size'][dataset] if torch.cuda.is_available() else 4
            W = features(waves, batch, whisper)
            np.savez_compressed(cached, clip_ids=ids, acoustic=X, whisper=W,
                                manifest_sha256=np.asarray(info['manifest_sha256']))
        assert X.shape == (3000, 91) and W.shape == (3000, 512)
        for spec in metadata['models']:
            model, sidecar = models[spec['id']]
            columns = sidecar.get('columns', sidecar.get('acoustic_columns_in_91', sidecar.get('feature_columns_in_91')))
            values = X[:, columns]
            if spec['id'].startswith('fusion'):
                values = np.column_stack((values.astype(np.float32), W))
            labels = (model.predict_proba(values)[:, 1] >= .5).astype(int) if spec['decision_rule'] == 'calibrated_0.50' else fitted_svm(model).predict(values)
            mismatches = int(np.sum(labels != expected.loc[(dataset, ids), spec['id']].to_numpy()))
            correct = int(np.sum(labels == info['label']))
            results.append({'dataset': dataset, 'model': spec['id'], 'correct': correct,
                            'errors': len(labels) - correct, 'correct_pct': correct / len(labels) * 100,
                            'differences_from_saved_predictions': mismatches})
        print(dataset, 'complete', flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(results)
    result.to_csv(args.output / 'summary.csv', index=False)
    print(result.to_string(index=False))


if __name__ == '__main__':
    main()
