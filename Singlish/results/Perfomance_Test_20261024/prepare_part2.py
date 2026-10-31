"""Re-extract Part 2 inputs with the fixed 24 October preparation; keep models frozen."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
from io import BytesIO
import json
import multiprocessing
from pathlib import Path
import platform
import sys
from time import perf_counter

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from mvp.features import extract_features, normalize_loudness, trim_silence


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def quantized(wave):
    stream = BytesIO()
    sf.write(stream, wave, 16000, format="FLAC", subtype="PCM_16")
    return sf.read(BytesIO(stream.getvalue()), dtype="float32")[0]


def process(item):
    identity, blob, expected_pcm = item
    wave, rate = sf.read(BytesIO(blob), dtype="float32")
    pcm, _ = sf.read(BytesIO(blob), dtype="int16", always_2d=True)
    assert rate == 16000 and wave.ndim == 1
    assert hashlib.sha256(str(rate).encode() + pcm.astype("<i2").tobytes()).hexdigest() == expected_pcm
    trimmed = trim_silence(wave)
    assert 3 <= len(trimmed) / rate <= 6
    # Keep the historical normalized protocol independent of serving changes.
    prepared = quantized(normalize_loudness(trimmed))
    variants = {"trim_only": trimmed, "normalize_only": quantized(normalize_loudness(wave)),
                "trim_and_normalize": prepared}
    vectors = {name: extract_features(value) for name, value in variants.items()}
    return identity, vectors, prepared, len(wave) / rate, len(prepared) / rate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT.parent / "E1_Feature_Engineering/data/mnsc_part2_unseen_3000")
    parser.add_argument("--raw-cache", type=Path, default=ROOT / "tmp/release-comparison-portable-cache/mnsc_part2_unseen_3000.npz")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/performance_test_20261024_prepared")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--review-cache", type=Path, help="Optional independent SVM ablation cache to cross-check")
    args = parser.parse_args()
    started = perf_counter()
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((HERE / "metadata.json").read_text())
    prior = ROOT / metadata["baseline_report"]
    baseline = json.loads((prior / "metadata.json").read_text())
    assert sha(ROOT / "mvp/features.py") == baseline["code_sha256"]["mvp/features.py"]
    dataset = "mnsc_part2_unseen_3000"
    manifest = prior / "manifests" / (dataset + ".csv")
    assert sha(manifest) == baseline["datasets"][dataset]["manifest_sha256"]
    with manifest.open() as stream:
        rows = sorted(csv.DictReader(stream), key=lambda r: r["clip_id"])
    ids = [r["clip_id"] for r in rows]
    assert len(set(ids)) == len(ids) == 3000
    blobs, parquet_hashes = {}, {}
    for path in sorted(args.data_root.glob("part-*.parquet")):
        digest = sha(path)
        assert digest == baseline["datasets"][dataset]["original_prepared_parquet_sha256"][path.name]
        parquet_hashes[path.name] = digest
        for batch in pq.ParquetFile(path).iter_batches(columns=["clip_id", "audio_flac"]):
            for r in batch.to_pylist():
                assert r["clip_id"] not in blobs
                blobs[r["clip_id"]] = r["audio_flac"]
    assert set(blobs) == set(ids)
    assert sha(args.raw_cache) == metadata["feature_cache_sha256"][dataset]
    with np.load(args.raw_cache, allow_pickle=False) as z:
        assert z["clip_ids"].tolist() == ids
        raw = z["acoustic"].copy()
        raw_whisper = z["whisper"].copy()
    for i in (0, 1000, 2000, 2999):
        wave, _ = sf.read(BytesIO(blobs[ids[i]]), dtype="float32")
        np.testing.assert_allclose(extract_features(wave), raw[i], rtol=1e-5, atol=1e-4)
    print("Raw feature, Parquet and manifest checks passed.", flush=True)
    items = [(r["clip_id"], blobs[r["clip_id"]], r["pcm_sha256"]) for r in rows]
    processed = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("fork")) as pool:
        for i, result in enumerate(pool.map(process, items, chunksize=10), 1):
            processed.append(result)
            if i % 300 == 0:
                print(f"Acoustic ablations {i}/3000; {perf_counter() - started:.1f}s", flush=True)
    assert [r[0] for r in processed] == ids
    arrays = {"original_raw": raw}
    for variant in processed[0][1]:
        arrays[variant] = np.stack([r[1][variant] for r in processed])
    review_hash = None
    if args.review_cache:
        with np.load(args.review_cache, allow_pickle=False) as review:
            assert review["clip_ids"].tolist() == ids
            for variant, values in arrays.items():
                np.testing.assert_allclose(values, review[variant], rtol=1e-5, atol=1e-4)
        review_hash = sha(args.review_cache)
        print("Fresh acoustic arrays match the independent review cache.", flush=True)
    # Import CUDA only after the forked acoustic workers have exited.
    import torch
    import transformers
    from mvp.embeddings import WhisperEmbedding
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    whisper = WhisperEmbedding(baseline["whisper"]["model"], baseline["whisper"]["revision"])
    whisper.encoder.to(device=device, dtype=dtype)
    waves = [r[2] for r in processed]
    batch_size = 64 if device == "cuda" else 4
    values = []
    with torch.inference_mode():
        for start in range(0, len(waves), batch_size):
            batch = waves[start:start + batch_size]
            inputs = whisper.extractor(batch, sampling_rate=16000, do_normalize=False,
                                       device=device, return_tensors="pt").input_features.to(device=device, dtype=dtype)
            hidden = whisper.encoder(input_features=inputs).last_hidden_state.float()
            lengths = torch.tensor([(len(w) + 319) // 320 for w in batch], device=device).clamp(1, hidden.shape[1])
            mask = torch.arange(hidden.shape[1], device=device)[None, :] < lengths[:, None]
            values.append(((hidden * mask[:, :, None]).sum(1) / lengths[:, None]).cpu().numpy())
            if start % (batch_size * 10) == 0:
                print(f"Prepared Whisper embeddings {min(start + len(batch), 3000)}/3000", flush=True)
    prepared_whisper = np.vstack(values)
    assert prepared_whisper.shape == (3000, 512) and np.isfinite(prepared_whisper).all()
    output = args.output / "part2_preprocessing_features.npz"
    np.savez_compressed(output, clip_ids=np.asarray(ids), manifest_sha256=np.asarray(sha(manifest)),
                        raw_durations=np.asarray([r[3] for r in processed]),
                        prepared_durations=np.asarray([r[4] for r in processed]),
                        whisper_raw=raw_whisper, whisper_prepared=prepared_whisper, **arrays)
    verification = {"clips": 3000, "pcm_hashes_verified": 3000, "frozen_preparation_checks": 3000,
                    "preprocessing_protocol": "24_oct_trim_normalize_pcm16",
                    "raw_feature_spot_checks": 4, "raw_over_six_seconds": sum(r[3] > 6 for r in processed),
                    "prepared_duration_min": min(r[4] for r in processed),
                    "prepared_duration_max": max(r[4] for r in processed),
                    "cropped_clips": 0, "discarded_clips": 0, "threshold": 0.5, "training": False,
                    "prepared_cache_sha256": sha(output), "original_cache_sha256": sha(args.raw_cache),
                    "manifest_sha256": sha(manifest), "parquet_sha256": parquet_hashes,
                    "review_cache_sha256": review_hash, "python": platform.python_version(),
                    "numpy": np.__version__, "torch": torch.__version__, "transformers": transformers.__version__,
                    "whisper_device": device, "whisper_dtype": str(dtype), "whisper_batch_size": batch_size,
                    "whisper_revision": baseline["whisper"]["revision"],
                    "prepare_script_sha256": sha(__file__), "features_code_sha256": sha(ROOT / "mvp/features.py"),
                    "seconds": perf_counter() - started}
    (args.output / "preparation_verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    print(json.dumps(verification, indent=2), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
