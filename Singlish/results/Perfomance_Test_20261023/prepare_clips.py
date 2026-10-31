"""Recover the fixed clip lists from the original sources."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile

import pandas as pd
import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import hf_hub_download

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'accent_classifier_data/scripts'))
from cv_utils import preprocess_waveform, encode_flac


def prepare(raw, row, output, original=False):
    assert hashlib.sha256(raw).hexdigest() == row.raw_sha256, row.clip_id
    wave, rate = sf.read(io.BytesIO(raw), dtype='int16' if original else 'float32')
    if original:
        assert rate == 16000 and wave.ndim == 1
    else:
        wave, reason = preprocess_waveform(wave, rate, 3, 6)
        assert reason == 'ok', (row.clip_id, reason)
    blob = encode_flac(wave)
    pcm, rate = sf.read(io.BytesIO(blob), dtype='int16', always_2d=True)
    digest = hashlib.sha256(str(rate).encode() + pcm.astype('<i2').tobytes()).hexdigest()
    assert digest == row.pcm_sha256, row.clip_id
    (output / (row.source_clip_id + '.flac')).write_bytes(blob)


def main():
    metadata = json.loads((HERE / 'metadata.json').read_text())
    path = HERE.parents[1] / 'accent_classifier_data/scripts/cv_utils.py'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == metadata['code_sha256']['accent_classifier_data/scripts/cv_utils.py']
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=metadata['datasets'], required=True)
    parser.add_argument('--output', type=Path, default=Path('data/release_comparison'))
    parser.add_argument('--cv-archive', type=Path)
    parser.add_argument('--mnsc-source', type=Path)
    args = parser.parse_args()
    info = metadata['datasets'][args.dataset]
    rows = pd.read_csv(HERE / 'manifests' / (args.dataset + '.csv'), keep_default_na=False)
    output = args.output / args.dataset
    output.mkdir(parents=True, exist_ok=True)
    done = 0
    if args.dataset == 'cv_unseen_3000':
        if not args.cv_archive:
            parser.error('--cv-archive is required for Common Voice')
        wanted = {info['source']['clip_path'].format(source_clip_id=r.source_clip_id): r for r in rows.itertuples()}
        with tarfile.open(args.cv_archive, 'r|gz') as archive:
            for member in archive:
                if member.name in wanted:
                    prepare(archive.extractfile(member).read(), wanted.pop(member.name), output)
                    done += 1
                    if not wanted:
                        break
        assert not wanted, f'{len(wanted)} clips missing from archive'
    else:
        source = info['source']
        for shard, group in rows.groupby('source_shard', sort=True):
            path = args.mnsc_source / shard if args.mnsc_source else hf_hub_download(
                source['repo'], shard, repo_type='dataset', revision=source['revision'])
            parquet = pq.ParquetFile(path)
            offset = 0
            for number in range(parquet.num_row_groups):
                end = offset + parquet.metadata.row_group(number).num_rows
                selected = group.loc[group.source_row.between(offset, end - 1)]
                if len(selected):
                    audio = parquet.read_row_group(number, columns=['context']).to_pylist()
                    for row in selected.itertuples():
                        prepare(audio[row.source_row - offset]['context']['bytes'], row, output,
                                original=args.dataset == 'mnsc_part2_unseen_3000')
                        done += 1
                offset = end
            print(shard, done, flush=True)
    assert done == info['clips']
    print(f'Saved {done} verified clips to {output}')


if __name__ == '__main__':
    main()
