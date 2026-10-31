"""Compare the final 24/25 October Ensembles on user-labelled non-Singapore outliers."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
from pathlib import Path
import platform
import sys
from time import perf_counter

import joblib
import numpy as np
import sklearn
import torch
from threadpoolctl import threadpool_limits
import soundfile as sf
import transformers

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from mvp.audio import AudioError, read_audio, SHORT_CLIP_MESSAGE
from mvp.embeddings import WhisperEmbedding
from mvp.features import (FEATURE_NAMES, SAMPLE_RATE, MIN_SECONDS, MAX_SECONDS,
                          extract_features, normalize_loudness, trim_silence)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_rows(path, rows):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prepare(data, day):
    # Freeze each release's preparation independently of the live MVP model.
    waveform, info = read_audio(data)
    waveform = trim_silence(waveform)
    duration = len(waveform) / SAMPLE_RATE
    if duration < MIN_SECONDS:
        raise AudioError(SHORT_CLIP_MESSAGE, "invalid_duration")
    truncated = duration > MAX_SECONDS
    waveform = waveform[:SAMPLE_RATE * MAX_SECONDS].astype(np.float32)
    warnings = ["Only the first six seconds after silence trimming were classified."] if truncated else []
    if np.sqrt(np.mean(waveform.astype(np.float64) ** 2)) < 1e-5:
        raise AudioError("The recording is silent or too quiet. Record clear English speech.")
    audio = {"duration_seconds": round(info.duration, 4),
             "classified_duration_seconds": round(min(duration, MAX_SECONDS), 4),
             "truncated": truncated}
    if day == "24":
        waveform = normalize_loudness(waveform)
        stream = BytesIO()
        sf.write(stream, waveform, SAMPLE_RATE, format="FLAC", subtype="PCM_16")
        stream.seek(0)
        waveform, _ = sf.read(stream, dtype="float32")
    return waveform, audio, warnings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips", type=Path, default=ROOT.parent / "E1_Feature_Engineering/outlier")
    parser.add_argument("--models", type=Path, default=ROOT / "mvp/models")
    parser.add_argument("--output", type=Path, default=HERE)
    args = parser.parse_args()
    started = perf_counter()
    source = read_rows(args.clips / "manifest.csv")
    listed = {r["file"]: r for r in source}
    assert len(listed) == len(source)
    files = sorted(p.relative_to(args.clips).as_posix() for p in args.clips.rglob("*.wav"))
    assert files and set(files) <= set(listed)
    missing = sorted(set(listed) - set(files))
    frozen = json.loads((ROOT / "results/Perfomance_Test_20261025/metadata.json").read_text())
    models, specs = {}, {}
    for day in ("24", "25"):
        spec = next(s for s in frozen["models"] if s["id"] == f"fusion65_2610{day}")
        path = args.models / f"fusion_svm_c_2610{day}.joblib"
        sidecar = path.with_name(path.stem + "_metadata.json")
        assert sha(path) == spec["sha256"], path
        assert sha(sidecar) == spec["metadata_sha256"], sidecar
        model, info = joblib.load(path), json.loads(sidecar.read_text())
        assert model.e4_probability_metadata_ == info
        assert info["threshold"] == 0.5 and np.array_equal(model.classes_, [0, 1])
        columns = info["acoustic_columns_in_91"]
        assert info["acoustic_feature_names"] == [FEATURE_NAMES[i] for i in columns]
        assert model.n_features_in_ == len(columns) + info["whisper_width"] == 577
        models[day] = model, info, columns
        specs[day] = {"artifact": str(path.resolve()), "sha256": sha(path),
                      "metadata_sha256": sha(sidecar), "threshold": 0.5,
                      "preprocessing": "trim_normalize_pcm16" if day == "24" else "trim_preserve_gain_float32"}
    revision = models["25"][1]["whisper_provenance"]["resolved_revision"]
    assert models["24"][1]["whisper_provenance"]["resolved_revision"] == revision
    encoder = WhisperEmbedding("openai/whisper-base.en", revision)
    predictions, rejected, inventory = [], [], []
    for number, name in enumerate(files, 1):
        path = args.clips / name
        blob = path.read_bytes()
        assert sha(path) == listed[name]["sha256"], path
        record = {"file": name, "language": listed[name]["language"], "true_label": 0,
                  "sha256": sha(path)}
        inventory.append(record.copy())
        for day in ("24", "25"):
            try:
                waveform, audio, warnings = prepare(blob, day)
            except AudioError as error:
                rejected.append({"file": name, "model": "ensemble_2610" + day,
                                 "code": error.code, "message": error.message})
                record["status_2610" + day] = error.code
                record["p_singapore_2610" + day] = ""
                record["prediction_2610" + day] = ""
                continue
            vector = extract_features(waveform)
            embedding = encoder(waveform)
            assert vector.shape == (91,) and embedding.shape == (512,)
            assert np.isfinite(vector).all() and np.isfinite(embedding).all()
            model, info, columns = models[day]
            values = np.concatenate((vector[columns].astype(np.float32), embedding)).reshape(1, -1)
            probabilities = model.predict_proba(values)[0]
            assert np.isfinite(probabilities).all() and np.isclose(probabilities.sum(), 1)
            probability = float(probabilities[1])
            label = int(probability >= info["threshold"])
            record["status_2610" + day] = "scored"
            record["p_singapore_2610" + day] = probability
            record["prediction_2610" + day] = label
            record["classified_seconds_2610" + day] = audio["classified_duration_seconds"]
            record["prepared_rms_2610" + day] = float(np.sqrt(np.mean(waveform.astype(np.float64) ** 2)))
            record["truncated_2610" + day] = audio["truncated"]
        predictions.append(record)
        print(f"{number}/{len(files)} {name}: " + ", ".join(
            f"{day} Oct={record['prediction_2610' + day]}" for day in ("24", "25")), flush=True)
    summaries = []
    for language in [*sorted({r["language"] for r in predictions}), "all"]:
        rows = [r for r in predictions if language == "all" or r["language"] == language]
        for day in ("24", "25"):
            scored = [r for r in rows if r["status_2610" + day] == "scored"]
            correct = sum(r["prediction_2610" + day] == 0 for r in scored)
            count = len(scored)
            summaries.append({"language": language, "model": "ensemble_2610" + day,
                              "available_clips": len(rows), "scored_clips": count,
                              "rejected_clips": len(rows) - count,
                              "correct_non_singapore": correct, "false_singapore": count - correct,
                              "specificity_pct": correct / count * 100 if count else "",
                              "false_positive_rate_pct": (count - correct) / count * 100 if count else ""})
    args.output.mkdir(parents=True, exist_ok=True)
    write_rows(args.output / "predictions.csv", predictions)
    write_rows(args.output / "summary.csv", summaries)
    if rejected:
        write_rows(args.output / "rejected.csv", rejected)
    metadata = {"date": datetime.now(timezone.utc).isoformat(), "clip_root": str(args.clips.resolve()),
                "label_source": "All available clips treated as non-Singapore (0), as specified by the user.",
                "dataset_context": "Non-English Google FLEURS outliers: Malay, Mandarin and Tamil. Source language alone does not verify accent labels; this is an outlier stress test.",
                "manifest_sha256": sha(args.clips / "manifest.csv"), "listed_clips": len(source),
                "available_clips": len(files), "missing_files": missing, "rejected": rejected,
                "clips": inventory, "models": specs,
                "whisper": {"model": "openai/whisper-base.en", "revision": revision,
                            "device": "cpu", "dtype": "float32", "pooling": "mean real ceil(samples/320) encoder steps"},
                "code_sha256": {name: sha(ROOT / name) for name in
                                ("mvp/audio.py", "mvp/features.py", "mvp/embeddings.py")},
                "script_sha256": sha(__file__), "python": platform.python_version(),
                "numpy": np.__version__, "sklearn": sklearn.__version__, "torch": torch.__version__,
                "transformers": transformers.__version__, "seconds": perf_counter() - started}
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    report = ["# Outlier test — 24 vs 25 October Ensembles", "",
              "Final calibrated E4 models; Singapore when p_singapore >= 0.50. "
              "Each model uses its matching preprocessing. Both acoustic and Whisper features "
              "are freshly extracted from that same prepared waveform with CPU float32 inference.", "",
              f"All {len(files)} available WAVs are labelled non-Singapore as requested. "
              f"The source manifest lists {len(source)} clips. Missing: {', '.join(missing) or 'none'}.", "",
              "| Language | Clips | 24 Oct correct non-SG | 24 Oct false SG | 25 Oct correct non-SG | 25 Oct false SG |",
              "|---|---:|---:|---:|---:|---:|"]
    for language in [*sorted({r["language"] for r in predictions}), "all"]:
        a, b = [r for r in summaries if r["language"] == language]
        report.append(f'| {language} | {a["available_clips"]} | {a["correct_non_singapore"]}/{a["scored_clips"]} '
                      f'| {a["false_singapore"]} | {b["correct_non_singapore"]}/{b["scored_clips"]} | {b["false_singapore"]} |')
    report += ["", f"Rejected model/clip evaluations: {len(rejected)}. Missing files and rejections are never counted as correct classifications.", "",
               metadata["dataset_context"], "",
               "## Reproduce", "", "```bash", "HF_HUB_OFFLINE=1 mvp/.venv/bin/python results/Outlier_Test_20261025/reproduce.py", "```", "",
               "Per-clip probabilities are in `predictions.csv`; aggregate results are in `summary.csv`. "
               "`metadata.json` records input/model hashes, missing files, preprocessing and runtime.", ""]
    (args.output / "Outlier_Test_Report.md").write_text("\n".join(report))
    print("\n".join(report), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
