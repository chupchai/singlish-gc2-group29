"""Rebuild the fixed 40,000-clip selection, retaining reference audio and splits.

The manifest contains metadata and PCM checksums only. Audio is retrieved from
its original sources; no local experiment directory is needed. Preparation is
resumable and replaces a source's output only after every selected clip passes.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import shutil
import struct
import threading
import time
import zipfile
import zlib

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
import soundfile as sf

from cv_utils import (AUDIO_PROCESSING, RAW_DIR, SOURCE_SUBSETS, SUBSET_SCHEMA,
                      TARGET_SR, ArchiveStreamer, decode_audio_bytes, encode_flac,
                      find_latest_archive, preprocess_waveform)

MANIFEST_DIR = Path(__file__).resolve().parents[1] / 'manifests'
SPEC = json.loads((MANIFEST_DIR / 'dataset_40000.json').read_text())
CLIP_TARGETS = {k: v for k, v in SPEC['accent_counts'].items()
                if k not in ('Singaporean', 'Korean', 'Mainland Chinese')}
MNSC_REPO = 'AudioLLMs/Multitask-National-Speech-Corpus-v1-extend'
CV22_REPO = 'malaysia-ai/common_voice_22_0'
HTTP_LOCAL = threading.local()


def load_reference():
    path = MANIFEST_DIR / SPEC['selection_manifest']
    if hashlib.sha256(path.read_bytes()).hexdigest() != SPEC['manifest_sha256']:
        raise ValueError('Reference manifest checksum mismatch.')
    return pd.read_csv(path, keep_default_na=False, low_memory=False, dtype={'speaker_id': str})


def read_receipts(path):
    receipt = Path(path).with_suffix('.json')
    if not receipt.exists():
        return {}
    info = json.loads(receipt.read_text())
    if info.get('reference_manifest_sha256') != SPEC['manifest_sha256']:
        raise ValueError(f'Preparation receipt refers to a different clip selection: {receipt}')
    return info['decoded_sha256']


def pcm_hash(blob):
    pcm, sr = sf.read(io.BytesIO(blob), dtype='int16', always_2d=True)
    return hashlib.sha256(str(sr).encode() + pcm.astype('<i2').tobytes()).hexdigest()


def check_record(record, expected, expected_pcm=None):
    for name in SUBSET_SCHEMA.names:
        if name not in ('audio_flac', 'duration_sec') and record[name] != expected[name]:
            raise ValueError(f'{expected["clip_id"]}: reference {name} differs.')
    with sf.SoundFile(io.BytesIO(record['audio_flac'])) as stream:
        if (stream.samplerate, stream.channels, stream.subtype, stream.format, len(stream)) != (
                TARGET_SR, 1, 'PCM_16', 'FLAC', int(expected['samples'])):
            raise ValueError(f'{expected["clip_id"]}: reference audio format/duration differs.')
    if abs(record['duration_sec'] - expected['duration_sec']) > 1 / TARGET_SR:
        raise ValueError(f'{expected["clip_id"]}: reference duration metadata differs.')
    if pcm_hash(record['audio_flac']) != (expected_pcm or expected['decoded_sha256']):
        raise ValueError(f'{expected["clip_id"]}: audio differs from the reference PCM checksum. '
                         'Check the source version and audio dependency versions; no substitute clip was selected.')


def process_blob(blob, expected):
    if expected['raw_sha256'] and hashlib.sha256(blob).hexdigest() != expected['raw_sha256']:
        raise ValueError(f'{expected["clip_id"]}: raw source checksum differs.')
    x, sr = decode_audio_bytes(blob)
    y, reason = preprocess_waveform(x, sr, 3, 6)
    if y is None:
        raise ValueError(f'{expected["clip_id"]}: preprocessing failed ({reason}).')
    record = {k: expected[k] for k in SUBSET_SCHEMA.names if k != 'audio_flac'}
    record['duration_sec'] = len(y) / TARGET_SR
    record['audio_flac'] = encode_flac(y)
    return record


class ReferenceWriter:
    """Checkpoint validated clips, then publish the complete source folder."""
    def __init__(self, root, source, reference):
        self.root, self.source = Path(root), source
        self.reference = reference
        self.expected = {r['clip_id']: r for r in reference.to_dict('records')}
        self.target = self.root / SOURCE_SUBSETS[source]
        self.stage = self.root / (SOURCE_SUBSETS[source] + '_reference_staging')
        self.stage.mkdir(parents=True, exist_ok=True)
        self.seen, self.buffer, self.hashes = set(), [], {}
        self.index = 0
        for path in sorted(self.stage.glob('part-*.parquet')):
            receipts = read_receipts(path)
            for batch in pq.ParquetFile(path).iter_batches(batch_size=64):
                for row in batch.to_pylist():
                    cid = row['clip_id']
                    if cid in self.seen or cid not in self.expected:
                        raise ValueError(f'Unexpected/duplicate checkpoint clip: {cid}')
                    check_record(row, self.expected[cid], receipts.get(cid))
                    self.hashes[cid] = pcm_hash(row['audio_flac'])
                    self.seen.add(cid)
            self.index = max(self.index, int(path.stem.split('-')[1]) + 1)
        # Reuse matching audio from a shared archive or an earlier completed run.
        for path in sorted(self.target.glob('part-*.parquet')):
            receipts = read_receipts(path)
            for batch in pq.ParquetFile(path).iter_batches(batch_size=64):
                for row in batch.to_pylist():
                    cid = row['clip_id']
                    if cid in self.expected and cid not in self.seen:
                        try:
                            self.add(row, expected_pcm=receipts.get(cid))
                        except ValueError:
                            # A prior selection may use the same ID with different
                            # audio or splits. Fetch the reference version instead.
                            continue
        self.flush()

    def add(self, row, *, from_source=False, expected_pcm=None):
        cid = row['clip_id']
        if cid in self.seen:
            return
        # Fresh audio is authenticated by the pinned source location/raw digest.
        # Record its PCM digest because libsoxr rounding varies across platforms.
        # Existing/checkpoint audio must match an original digest or a receipt.
        actual_pcm = pcm_hash(row['audio_flac'])
        check_record(row, self.expected[cid], actual_pcm if from_source else expected_pcm)
        self.hashes[cid] = actual_pcm
        self.buffer.append(row)
        self.seen.add(cid)
        if len(self.buffer) >= 250:
            self.flush()

    def flush(self):
        if self.buffer:
            path = self.stage / f'part-{self.index:04d}.parquet'
            temporary = path.with_suffix('.partial')
            pq.write_table(pa.Table.from_pylist(self.buffer, schema=SUBSET_SCHEMA), temporary)
            receipt = path.with_suffix('.json')
            receipt_tmp = receipt.with_suffix('.partial.json')
            receipt_tmp.write_text(json.dumps({
                'reference_manifest_sha256': SPEC['manifest_sha256'],
                'decoded_sha256': {r['clip_id']: self.hashes[r['clip_id']] for r in self.buffer}
            }, indent=2) + '\n')
            receipt_tmp.replace(receipt)
            temporary.replace(path)
            self.index += 1
            self.buffer = []
            print(f'{self.source}: {len(self.seen):,}/{len(self.expected):,} clips checked', flush=True)

    @property
    def missing(self):
        return self.reference[~self.reference.clip_id.isin(self.seen)]

    def finish(self):
        self.flush()
        if len(self.missing):
            raise ValueError(f'{self.source}: missing {len(self.missing):,} reference clips.')
        # Keep the previous output until the new source has passed every check.
        backup = self.target.with_name(self.target.name + '_previous')
        if backup.exists():
            if not self.target.exists():
                backup.rename(self.target)
            else:
                shutil.rmtree(backup)
        if self.target.exists():
            self.target.rename(backup)
        self.stage.rename(self.target)
        if backup.exists():
            shutil.rmtree(backup)
        from extract_common_voice_subset import summarise
        summarise(self.reference).to_csv(self.root / f'{self.target.name}_summary.csv', index=False)
        info = {'source': self.source, 'clips': len(self.reference),
                'reference_manifest_sha256': SPEC['manifest_sha256'],
                'audio_processing': AUDIO_PROCESSING,
                'groups': {g: {s: {'n_clips': len(x), 'speakers': sorted(x.speaker_id.unique())}
                              for s, x in sub.groupby('split')}
                           for g, sub in self.reference.groupby('accent_group')}}
        (self.root / f'{self.target.name}_split_info.json').write_text(json.dumps(info, indent=2) + '\n')
        # Write metadata for sources that have actually completed (any run order).
        sources = []
        for source, folder in SOURCE_SUBSETS.items():
            path = self.root / f'{folder}_split_info.json'
            if path.exists() and json.loads(path.read_text()).get('reference_manifest_sha256') == SPEC['manifest_sha256']:
                sources.append(source)
        load_reference().query('source in @sources').to_csv(self.root / 'manifest.csv', index=False)


def fetch_zip_clip(row, session=None):
    """Fetch one pinned CV22 ZIP member by range and verify its name, CRC and size."""
    url = (f'https://huggingface.co/datasets/{CV22_REPO}/resolve/'
           f'{SPEC["revisions"]["cv22_audio"]}/{row["source_shard"]}')
    if session is None:
        if not hasattr(HTTP_LOCAL, 'session'):
            HTTP_LOCAL.session = requests.Session()
        session = HTTP_LOCAL.session
    start = int(row['source_byte_offset'])
    end = start + 30 + len(row['source_member'].encode()) + 4096 + int(row['zip_compressed_size']) - 1
    for attempt in range(4):
        try:
            response = session.get(url, headers={'Range': f'bytes={start}-{end}'}, timeout=(20, 90))
            response.raise_for_status()
            blob = response.content
            if response.status_code != 206 or blob[:4] != b'PK\x03\x04':
                raise ValueError('Expected ZIP byte-range response.')
            name_len, extra_len = struct.unpack_from('<HH', blob, 26)
            if blob[30:30 + name_len].decode() != row['source_member']:
                raise ValueError('ZIP member identity differs.')
            offset = 30 + name_len + extra_len
            data = blob[offset:offset + int(row['zip_compressed_size'])]
            compression = int(row['zip_compression'])
            if compression == zipfile.ZIP_DEFLATED:
                data = zlib.decompress(data, -15)
            elif compression != zipfile.ZIP_STORED:
                raise ValueError('Unsupported ZIP compression.')
            if len(data) != int(row['zip_file_size']) or zlib.crc32(data) != int(row['zip_crc']):
                raise ValueError('ZIP member size/CRC differs.')
            return data
        except (requests.RequestException, ValueError, zlib.error):
            if attempt == 3:
                raise
            time.sleep(attempt + 1)


def prepare_cv(writer, args):
    missing = writer.missing
    originals = missing[~missing.source_version.eq('Common Voice English 22.0')]
    if len(originals):
        archive = args.archive or find_latest_archive(RAW_DIR)
        wanted = originals.set_index('clip_id').to_dict('index')
        print(f'Reading original Common Voice clips from {archive}', flush=True)
        with ArchiveStreamer(archive) as stream:
            timer = time.monotonic()
            for entry in stream:
                cid = Path(entry.basename).stem
                if entry.basename.endswith('.mp3') and cid in wanted:
                    expected = dict(wanted.pop(cid), clip_id=cid)
                    writer.add(process_blob(entry.read(), expected), from_source=True)
                    if not wanted:
                        break
                if time.monotonic() - timer >= 30:
                    print(f'Common Voice archive: {stream.fraction:.1%}; {len(wanted):,} clips remaining', flush=True)
                    timer = time.monotonic()
        writer.flush()
        if wanted:
            raise ValueError(f'{len(wanted):,} original clips missing from {archive}. '
                             'Use the full English 27.0 archive; no replacement clips will be sampled.')
    rows = writer.missing.to_dict('records')
    with ThreadPoolExecutor(max_workers=8) as pool:
        for start in range(0, len(rows), 64):
            batch = rows[start:start + 64]
            for row, blob in zip(batch, pool.map(fetch_zip_clip, batch)):
                writer.add(process_blob(blob, row), from_source=True)


def prepare_mnsc(writer, args):
    from huggingface_hub import HfFileSystem, try_to_load_from_cache
    from prepare_mnsc_singaporean import extract_speaker_id, transcript_key
    fs = HfFileSystem(token=False, block_size=1024 * 1024)
    revision = SPEC['revisions']['mnsc']
    for name, group in writer.missing.groupby('source_shard', sort=True):
        # Range-read only the parquet row groups that hold selected clips.
        local = args.source_dir / name if args.source_dir else None
        if not local:
            cached = try_to_load_from_cache(MNSC_REPO, name, repo_type='dataset',
                        revision=revision, cache_dir=args.cache_dir)
            local = Path(cached) if isinstance(cached, str) else None
        if local and local.is_file():
            stream = local.open('rb')
        else:
            if args.no_download:
                raise ValueError(f'MNSC shard not available locally: {name}; rerun without --no_download.')
            stream = fs.open(f'datasets/{MNSC_REPO}@{revision}/{name}', 'rb')
        with stream:
            pf = pq.ParquetFile(stream)
            offsets = np.cumsum([0] + [pf.metadata.row_group(i).num_rows for i in range(pf.num_row_groups)])
            wanted = {int(r['source_row']): r for r in group.to_dict('records')}
            for index in range(pf.num_row_groups):
                selected = {k - offsets[index]: v for k, v in wanted.items()
                            if offsets[index] <= k < offsets[index + 1]}
                if not selected:
                    continue
                rows = pf.read_row_group(index, columns=['context', 'answer', 'other_attributes']).to_pylist()
                for position, expected in selected.items():
                    raw = rows[position]
                    if 'mnsc_' + extract_speaker_id(raw['other_attributes']) != expected['speaker_id']:
                        raise ValueError(f'{expected["clip_id"]}: MNSC source speaker differs.')
                    # Some original transcripts use the corpus Transcription field
                    # (punctuation differs); clip identity is fixed by shard and row.
                    if not transcript_key(raw['answer']):
                        raise ValueError(f'{expected["clip_id"]}: empty MNSC source transcript.')
                    writer.add(process_blob(raw['context']['bytes'], expected), from_source=True)
        writer.flush()
        print(f'MNSC processed {name}', flush=True)


def prepare_speechocean(writer, args):
    from huggingface_hub import hf_hub_download
    from prepare_speechocean762 import HF_FILES, HF_REPO_DEFAULT, process_files
    files = []
    for name in HF_FILES:
        local = args.source_dir / name
        if not local.exists():
            # Support older downloads with the files directly in source_dir.
            local = args.source_dir / Path(name).name
        if not local.exists():
            if args.no_download:
                raise ValueError(f'Missing {name}; rerun without --no_download.')
            local = Path(hf_hub_download(HF_REPO_DEFAULT, name, repo_type='dataset',
                         revision=SPEC['revisions']['speechocean762'], token=False,
                         local_dir=str(args.source_dir)))
        files.append(local)
    for row in process_files(files, 18, 3, 6)['rows']:
        if row['clip_id'] in writer.expected and row['clip_id'] not in writer.seen:
            expected = writer.expected[row['clip_id']]
            row['split'] = expected['split']
            writer.add(row, from_source=True)


def prepare_saa(writer, args):
    from prepare_speech_accent_archive import iter_hf_mirror, iter_local, process_recordings
    stats = {'recordings': 0, 'recordings_without_chunks': 0, 'chunks': 0,
             'duplicate_recordings': 0, 'speakers_by_language': {}, 'recording_sec': [],
             'skips': {'too_short': 0, 'too_long': 0, 'silent': 0, 'decode_error': 0}}
    if args.source_dir:
        _, recordings = iter_local(args.source_dir, ['korean'], stats)
    else:
        _, recordings = iter_hf_mirror('CodecSR/speech_accent_archive_other', 'original',
                         SPEC['revisions']['speech_accent_archive'], args.cache_dir,
                         ['korean'], args.no_download, stats)
    # Chunking parameters are fixed to the reference generation settings.
    from argparse import Namespace
    settings = Namespace(min_duration=3, max_duration=6, chunk_target_min=4,
                         chunk_target_max=5.5, silence_db=-35, min_pause=.15)
    wanted_speakers = set(writer.missing.speaker_id)
    def checked_recordings():
        for key, language, blob in recordings:
            if 'saa_' + key not in wanted_speakers:
                continue
            expected = writer.reference[writer.reference.speaker_id.eq('saa_' + key)].iloc[0]
            if hashlib.sha256(blob).hexdigest() != expected.raw_sha256:
                raise ValueError(f'{key}: raw Speech Accent Archive checksum differs.')
            yield key, language, blob

    for row in process_recordings(checked_recordings(), settings, stats):
        if row['clip_id'] in writer.expected and row['clip_id'] not in writer.seen:
            row['split'] = writer.expected[row['clip_id']]['split']
            writer.add(row, from_source=True)


def prepare_reference(source, args):
    reference = load_reference()
    writer = ReferenceWriter(args.data_dir, source, reference[reference.source.eq(source)])
    if len(writer.missing):
        {'common_voice': prepare_cv, 'mnsc': prepare_mnsc,
         'speechocean762': prepare_speechocean,
         'speech_accent_archive': prepare_saa}[source](writer, args)
    writer.finish()
    return 0
