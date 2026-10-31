"""Decode every clip and audit duplicates, duration and preprocessing outputs."""
import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import soundfile as sf

from cv_utils import (DATA_DIR, SOURCE_SUBSETS, TARGET_SR, TARGET_RMS_DBFS,
                      PEAK_LIMIT, SILENT_RMS, TRIM_FRAME_MS, TRIM_HOP_MS,
                      TRIM_PAD_SEC, TRIM_TOP_DB)
from reference_dataset import CLIP_TARGETS, SPEC, load_reference, read_receipts
from prepare_mnsc_singaporean import transcript_key


def check_metadata(data_dir):
    frames = []
    for source, folder in SOURCE_SUBSETS.items():
        files = sorted((Path(data_dir) / folder).glob('part-*.parquet'))
        if not files:
            raise ValueError(f'Missing {source}; run its preparation script first.')
        frames.extend(pq.read_table(f, columns=['clip_id', 'source', 'speaker_id', 'accent_raw',
            'accent_group', 'is_singaporean', 'split', 'sampling_rate', 'duration_sec']).to_pandas() for f in files)
    frame = pd.concat(frames, ignore_index=True)
    expected = {**CLIP_TARGETS, 'Korean': 329, 'Mainland Chinese': 800, 'Singaporean': 20000}
    if frame.groupby('accent_group').size().to_dict() != expected:
        raise ValueError(f'Expected clip counts: {expected}; got {frame.groupby("accent_group").size().to_dict()}')
    if frame.groupby('is_singaporean').size().to_dict() != {False: 20000, True: 20000}:
        raise ValueError('Expected 20,000 clips per class.')
    positives = frame[frame.is_singaporean]
    if (not positives.source.eq('mnsc').all()
            or positives.groupby('accent_raw').size().to_dict() != {'Singaporean (MNSC PART1)': 7500, 'Singaporean (MNSC PART2)': 12500}):
        raise ValueError('Expected 7,500 PART1 and 12,500 PART2 clips.')
    if not frame.sampling_rate.eq(16000).all() or not frame.duration_sec.between(3, 6).all():
        raise ValueError('Expected 16 kHz audio lasting 3–6 seconds.')
    if frame.groupby(['source', 'speaker_id']).split.nunique().max() != 1:
        raise ValueError('A speaker appears in multiple splits.')
    if frame.duplicated(['source', 'clip_id']).any() or not frame.split.isin(['train', 'val', 'test']).all():
        raise ValueError('Duplicate clips or invalid splits.')
    reference = load_reference()
    columns = ['source', 'clip_id', 'speaker_id', 'accent_raw', 'accent_group',
               'is_singaporean', 'split', 'sampling_rate']
    actual = frame.sort_values(['source', 'clip_id']).reset_index(drop=True)
    expected_rows = reference.sort_values(['source', 'clip_id']).reset_index(drop=True)
    if (len(actual) != SPEC['clips']
            or not np.array_equal(actual[columns].to_numpy(), expected_rows[columns].to_numpy())
            or not np.allclose(actual.duration_sec, expected_rows.duration_sec, rtol=0, atol=1 / TARGET_SR)):
        raise ValueError('Clip identities, labels, durations or speaker splits differ from the fixed 40,000-clip manifest.')
    for source, cap in [('mnsc', 30), ('common_voice', 200)]:
        if frame[frame.source.eq(source)].groupby('speaker_id').size().max() > cap:
            raise ValueError(f'{source}: speaker clip cap exceeded.')
    print(frame.groupby(['source', 'accent_group']).agg(clips=('clip_id', 'size'), speakers=('speaker_id', 'nunique')).to_string())
    print(f'Total: {len(frame):,} clips; 20,000 per class; no speaker overlap.')
    return frame


def inspect_audio(blob):
    with sf.SoundFile(io.BytesIO(blob)) as stream:
        rate, channels, subtype, format_name = stream.samplerate, stream.channels, stream.subtype, stream.format
        pcm = stream.read(dtype='int16', always_2d=True)
    y = pcm.astype(np.float64) / 32768
    rms = float(np.sqrt(np.mean(y * y))) if y.size else 0.0
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    target = 10 ** (TARGET_RMS_DBFS / 20)
    tolerance = 2 / 32768
    issues, warnings = [], []
    if (rate, channels, subtype, format_name) != (TARGET_SR, 1, 'PCM_16', 'FLAC'):
        issues.append('audio_format')
    if not 3 * rate <= len(pcm) <= 6 * rate:
        issues.append('duration')
    if rms < SILENT_RMS:
        issues.append('silence')
    if peak > PEAK_LIMIT + tolerance:
        issues.append('peak_limit')
    if rms > target + tolerance or min(abs(rms - target), abs(peak - PEAK_LIMIT)) > tolerance:
        issues.append('loudness')
    left = right = 0
    if channels == 1 and len(y):
        frame = round(rate * TRIM_FRAME_MS / 1000)
        hop = round(rate * TRIM_HOP_MS / 1000)
        if len(y) >= frame:
            starts = np.arange(0, len(y) - frame + 1, hop)
            energy = np.concatenate(([0.0], np.cumsum(y[:, 0] ** 2)))
            power = (energy[starts + frame] - energy[starts]) / frame
            active = np.flatnonzero(power > power.max() * 10 ** (-TRIM_TOP_DB / 10))
            if len(active):
                left = max(0, int(starts[active[0]]) - round(rate * TRIM_PAD_SEC))
                right = max(0, len(y) - int(starts[active[-1]]) - frame - round(rate * TRIM_PAD_SEC))
                if max(left, right) > frame + hop:
                    warnings.append('boundary_trim_recheck')
    return {'decoded_sha256': hashlib.sha256(str(rate).encode() + pcm.astype('<i2').tobytes()).hexdigest(),
            'encoded_sha256': hashlib.sha256(blob).hexdigest(), 'samples': len(pcm),
            'actual_duration_sec': len(pcm) / rate, 'sample_rate': rate, 'channels': channels,
            'subtype': subtype, 'rms_dbfs': 20 * np.log10(rms) if rms else None,
            'peak': peak, 'extra_left_trim_sec': left / rate, 'extra_right_trim_sec': right / rate,
            'issues': issues, 'warnings': warnings}


def verify(data_dir=DATA_DIR, report_dir=None):
    root = Path(data_dir)
    check_metadata(root)
    reference = load_reference().set_index(['source', 'clip_id'])
    output = Path(report_dir) if report_dir else root / 'verification'
    output.mkdir(parents=True, exist_ok=True)
    rows, encoded, decoded = [], {}, {}
    for folder in SOURCE_SUBSETS.values():
        for path in sorted((root / folder).glob('part-*.parquet')):
            receipts = read_receipts(path)
            for batch in pq.ParquetFile(path).iter_batches(batch_size=64):
                for record in batch.to_pylist():
                    identity = record['source'] + '::' + record['clip_id']
                    info = {k: record[k] for k in ['source', 'clip_id', 'speaker_id', 'accent_group', 'sentence', 'split']}
                    try:
                        info.update(inspect_audio(record['audio_flac']))
                        expected_pcm = receipts.get(record['clip_id'],
                            reference.loc[(record['source'], record['clip_id']), 'decoded_sha256'])
                        if info['decoded_sha256'] != expected_pcm:
                            info['issues'].append('reference_pcm_mismatch')
                        if abs(info['actual_duration_sec'] - record['duration_sec']) > 1 / info['sample_rate']:
                            info['issues'].append('duration_metadata')
                        if info['sample_rate'] != record['sampling_rate']:
                            info['issues'].append('sample_rate_metadata')
                        for name, seen in [('encoded_sha256', encoded), ('decoded_sha256', decoded)]:
                            info[name + '_duplicate_of'] = seen.get(info[name], '')
                            seen.setdefault(info[name], identity)
                    except (RuntimeError, ValueError) as exc:
                        info['issues'] = ['decode_error']
                        info['error'] = str(exc)
                    info['issues'] = ';'.join(info['issues'])
                    info['warnings'] = ';'.join(info.get('warnings', []))
                    rows.append(info)
            print(f'Checked {len(rows):,} clips', flush=True)
    frame = pd.DataFrame(rows)
    utterances = frame[frame.sentence.fillna('').str.strip().ne('')].copy()
    utterances['transcript_key'] = utterances.sentence.map(transcript_key)
    repeated = utterances[utterances.duplicated(['source', 'speaker_id', 'transcript_key'], keep=False)]
    repeated.to_csv(output / 'repeated_utterances.csv', index=False)
    frame.to_csv(output / 'clips.csv', index=False)
    summary = {'clips_checked': len(frame), 'encoded_duplicate_clips': int(frame.encoded_sha256_duplicate_of.fillna('').ne('').sum()),
               'decoded_duplicate_clips': int(frame.decoded_sha256_duplicate_of.fillna('').ne('').sum()),
               'clips_with_issues': int(frame.issues.ne('').sum()),
               'issue_counts': frame.loc[frame.issues.ne(''), 'issues'].str.split(';').explode().value_counts().to_dict(),
               'clips_with_warnings': int(frame.warnings.ne('').sum()),
               'warning_counts': frame.loc[frame.warnings.ne(''), 'warnings'].str.split(';').explode().value_counts().to_dict(),
               'duration_min_sec': float(frame.actual_duration_sec.min()),
               'duration_max_sec': float(frame.actual_duration_sec.max()),
               'peak_max': float(frame.peak.max()), 'rms_dbfs_min': float(frame.rms_dbfs.min()),
               'rms_dbfs_max': float(frame.rms_dbfs.max()),
               'boundary_trim_tolerance_sec': (TRIM_FRAME_MS + TRIM_HOP_MS) / 1000,
               'repeated_speaker_transcript_rows': len(repeated),
               'checks': 'All clips decoded; PCM hash duplicates, format, duration and peak/RMS limits checked; boundary-silence rechecks recorded separately.',
               'warning_note': 'Boundary trimming before PCM16 quantization need not be idempotent afterward. Replay the raw input to resolve a boundary recheck.',
               'limitation': 'Final audio cannot prove the original resampler, processing order, or absence of padding/cropping; raw-input replay is needed for that.'}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_dir', type=Path, default=DATA_DIR)
    parser.add_argument('--report_dir', type=Path)
    args = parser.parse_args()
    result = verify(args.data_dir, args.report_dir)
    raise SystemExit(bool(result['encoded_duplicate_clips'] or result['decoded_duplicate_clips']
                          or result['clips_with_issues'] or result['repeated_speaker_transcript_rows']))
