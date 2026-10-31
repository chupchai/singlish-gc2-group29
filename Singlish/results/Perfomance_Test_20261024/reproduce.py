"""Evaluate the 24 October models on the fixed 23 October performance-test features."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import sys
from time import perf_counter

import joblib
import numpy as np
import sklearn
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from mvp.features import FEATURE_NAMES
from mvp.models import fitted_svm


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


def prepared_part2(args, metadata, loaded, summaries, predictions):
    """Replace only the primary Part 2 inputs; the historical outputs are kept separately."""
    correction = metadata["preprocessing_correction"]
    cache = args.part2_cache or ROOT / correction["cache_path"]
    assert sha(cache) == correction["cache_sha256"]
    with np.load(cache, allow_pickle=False) as z:
        ids = z["clip_ids"].tolist()
        arrays = {name: z[name].copy() for name in
                  ("original_raw", "trim_only", "normalize_only", "trim_and_normalize")}
        whisper = z["whisper_prepared"].copy()
        raw_durations, durations = z["raw_durations"].copy(), z["prepared_durations"].copy()
    assert len(ids) == 3000 and arrays["trim_and_normalize"].shape == (3000, 91)
    assert whisper.shape == (3000, 512) and np.isfinite(whisper).all()
    assert np.all((durations >= 3) & (durations <= 6))
    dataset = "mnsc_part2_unseen_3000"
    current_rows = {r["clip_id"]: r for r in predictions if r["dataset"] == dataset}
    assert set(current_rows) == set(ids)
    ablation = []
    ablation_predictions = [{"clip_id": identity, "raw_duration_seconds": float(raw_duration),
                             "prepared_duration_seconds": float(duration)}
                            for identity, raw_duration, duration in zip(ids, raw_durations, durations)]
    ml_id = next(spec["id"] for spec in metadata["models"] if spec["id"].startswith("svm"))
    model, info, columns = loaded[ml_id]
    for variant, features in arrays.items():
        values = features[:, columns]
        probability = model.predict_proba(values)[:, 1]
        labels = (probability >= 0.5).astype(int)
        uncalibrated = fitted_svm(model).predict(values)
        if variant == "original_raw":
            expected = [current_rows[identity][ml_id + "_p_singapore"] for identity in ids]
            np.testing.assert_allclose(probability, expected, rtol=1e-10, atol=1e-10)
        ablation.append({"variant": variant, "clips": len(ids), "recall_pct": labels.mean() * 100,
                         "misses": int((labels == 0).sum()),
                         "uncalibrated_recall_pct": uncalibrated.mean() * 100})
        for row, p, label in zip(ablation_predictions, probability, labels):
            row[variant + "_p_singapore"] = float(p)
            row[variant + "_prediction"] = int(label)
    for model_id in (ml_id, next(spec["id"] for spec in metadata["models"] if spec["id"].startswith("fusion")),
                     "svm65", "fusion65"):
        model, info, columns = loaded[model_id]
        values = arrays["trim_and_normalize"][:, columns]
        if model_id.startswith("fusion"):
            values = np.column_stack((values.astype(np.float32), whisper))
        probabilities = model.predict_proba(values)[:, 1]
        labels = (probabilities >= 0.5).astype(int)
        for identity, probability, label in zip(ids, probabilities, labels):
            current_rows[identity][model_id] = int(label)
            current_rows[identity][model_id + "_p_singapore"] = float(probability)
        for row in summaries:
            if row["dataset"] == dataset and row["model"] == model_id:
                assert row["clips"] == len(ids)
                row.update(correct=int(labels.sum()), errors=int((labels == 0).sum()),
                           correct_pct=labels.mean() * 100)
    for row in [*summaries, *predictions]:
        row["audio_preparation"] = ("trim_normalize_pcm16" if row["dataset"] == dataset
                                     else "upstream_trim_normalize_pcm16")
    write_rows(args.supporting / "part2_preprocessing_summary.csv", ablation)
    write_rows(args.supporting / "part2_preprocessing_predictions.csv", ablation_predictions)
    return {"prepared_cache_sha256": sha(cache), "clips": len(ids),
            "raw_over_six_seconds": int((raw_durations > 6).sum()),
            "prepared_min_seconds": float(durations.min()), "prepared_max_seconds": float(durations.max()),
            "cropped_clips": 0, "discarded_clips": 0}


def part2_diagnostics(args, metadata, loaded, study):
    folder = args.study_root / "outputs/e1_feature_study_accent_40000"
    with np.load(folder / "acoustic_features_91.npz", allow_pickle=False) as z:
        assert z["feature_names"].tolist() == FEATURE_NAMES
        features, ids = z["features"].copy(), z["clip_ids"].tolist()
    with np.load(args.cache / "mnsc_part2_unseen_3000.npz", allow_pickle=False) as z:
        raw = z["acoustic"].copy()
    records = {r["clip_id"]: r for r in study}
    internal, external = [], []
    for count in (65, 91):
        if count == 65:
            model = loaded[next(s["id"] for s in metadata["models"] if s["id"].startswith("svm"))][0]
        else:
            path = folder / "svm_calibrated_probability_testing/svm_91_calibrated.joblib"
            assert sha(path) == metadata["preprocessing_correction"]["diagnostic_svm91_sha256"]
            model = joblib.load(path)
        info = model.e1_probability_metadata_
        assert info["dataset_id"] == metadata["dataset_id"] and info["clips"] == 40000
        columns = info["columns"]
        for part in ("part1", "part2"):
            selected = [i for i, identity in enumerate(ids) if records[identity]["split"] == "test"
                        and int(records[identity]["label"]) == 1
                        and part in records[identity]["source_clip_id"].lower()]
            values = features[selected][:, columns]
            probability = model.predict_proba(values)[:, 1]
            internal.append({"features": count, "part": part.upper(), "clips": len(selected),
                             "prepared_calibrated_recall_pct": (probability >= 0.5).mean() * 100,
                             "prepared_uncalibrated_recall_pct": fitted_svm(model).predict(values).mean() * 100})
        values = raw[:, columns]
        external.append({"features": count, "input": "original_raw_part2", "clips": len(raw),
                         "calibrated_recall_pct": (model.predict_proba(values)[:, 1] >= 0.5).mean() * 100,
                         "uncalibrated_recall_pct": fitted_svm(model).predict(values).mean() * 100})
    write_rows(args.supporting / "internal_test_by_part.csv", internal)
    write_rows(args.supporting / "part2_calibration_feature_diagnostics.csv", external)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=ROOT / "tmp/release-comparison-portable-cache")
    parser.add_argument("--models", type=Path, default=ROOT / "mvp/models")
    parser.add_argument("--study-root", type=Path, default=ROOT.parent / "E1_Feature_EngineeringV4")
    parser.add_argument("--part2-cache", type=Path, help="Prepared Part 2 features from prepare_part2.py")
    parser.add_argument("--output", type=Path, default=HERE)
    args = parser.parse_args()
    args.supporting = args.output / "supporting"
    metadata = json.loads((HERE / "metadata.json").read_text())
    prior = ROOT / metadata["baseline_report"]
    assert sha(prior / "metadata.json") == metadata["baseline_metadata_sha256"]
    assert sha(prior / "predictions.csv") == metadata["baseline_predictions_sha256"]
    baseline = json.loads((prior / "metadata.json").read_text())
    assert sha(ROOT / "mvp/features.py") == baseline["code_sha256"]["mvp/features.py"]
    split_path = args.study_root / metadata["study_splits"]
    assert sha(split_path) == metadata["study_splits_sha256"]
    study = read_rows(split_path)
    assert len(study) == 40000
    fit_speakers = {r["speaker_group"] for r in study if r["split"] in ("train", "validation")}
    fit_ids = {r["clip_id"] for r in study if r["split"] in ("train", "validation")}
    all_ids = {r["clip_id"] for r in study}
    saved = {(r["dataset"], r["clip_id"]): r for r in read_rows(prior / "predictions.csv")}
    specs = [*metadata["models"], *baseline["models"]]
    loaded = {}
    for spec in specs:
        path = args.models / spec["artifact"]
        sidecar = path.with_name(path.stem + "_metadata.json")
        assert sha(path) == spec["sha256"], path
        assert sha(sidecar) == spec["metadata_sha256"], sidecar
        model, info = joblib.load(path), json.loads(sidecar.read_text())
        assert np.array_equal(model.classes_, [0, 1])
        columns = info.get("columns", info.get("acoustic_columns_in_91", info.get("feature_columns_in_91")))
        names = info.get("feature_names", info.get("acoustic_feature_names"))
        assert names == [FEATURE_NAMES[i] for i in columns]
        if spec in metadata["models"]:
            embedded = getattr(model, "e1_probability_metadata_" if spec["id"].startswith("svm") else "e4_probability_metadata_")
            assert embedded == info
            assert info["dataset_id"] == metadata["dataset_id"]
            assert info["clips"] == 40000 and info["threshold"] == 0.5
        loaded[spec["id"]] = model, info, columns
    summaries, predictions, audits = [], [], {}
    baseline_summaries = []
    started = perf_counter()
    for dataset, settings in baseline["datasets"].items():
        manifest = prior / "manifests" / (dataset + ".csv")
        assert sha(manifest) == settings["manifest_sha256"]
        rows = sorted(read_rows(manifest), key=lambda r: r["clip_id"])
        ids = [r["clip_id"] for r in rows]
        cache_path = args.cache / (dataset + ".npz")
        assert sha(cache_path) == metadata["feature_cache_sha256"][dataset]
        with np.load(cache_path, allow_pickle=False) as z:
            assert z["clip_ids"].tolist() == ids
            assert z["manifest_sha256"].item() == settings["manifest_sha256"]
            acoustic, whisper = z["acoustic"].copy(), z["whisper"].copy()
        assert acoustic.shape == (3000, 91) and whisper.shape == (3000, 512)
        assert np.isfinite(acoustic).all() and np.isfinite(whisper).all()
        independent = np.asarray([r["speaker_group"] not in fit_speakers for r in rows])
        speakers = {r["speaker_group"] for r in rows}
        assert len(speakers) == settings["speakers"]
        audits[dataset] = {
            "clips": len(rows), "speakers": len(speakers),
            "fit_speaker_overlap": len(speakers & fit_speakers),
            "clips_with_fit_speakers": int((~independent).sum()),
            "speaker_independent_clips": int(independent.sum()),
            "fit_clip_id_overlap": len(set(ids) & fit_ids),
            "all_study_clip_id_overlap": len(set(ids) & all_ids),
            "speaker_overlap_by_split": {split: len(speakers & {r["speaker_group"] for r in study if r["split"] == split})
                                         for split in ("train", "validation", "test")},
        }
        dataset_predictions = {r["clip_id"]: {
            "dataset": dataset, "clip_id": r["clip_id"], "speaker_group": r["speaker_group"],
            "true_label": settings["label"], "speaker_independent": int(clean),
        } for r, clean in zip(rows, independent)}
        for spec in specs:
            model, info, columns = loaded[spec["id"]]
            values = acoustic[:, columns]
            if spec["id"].startswith("fusion"):
                values = np.column_stack((values.astype(np.float32), whisper))
            probabilities = model.predict_proba(values)[:, 1] if spec["decision_rule"] == "calibrated_0.50" else None
            labels = (probabilities >= 0.5).astype(int) if probabilities is not None else fitted_svm(model).predict(values)
            current = spec in metadata["models"]
            if not current:
                expected = np.asarray([int(saved[(dataset, identity)][spec["id"]]) for identity in ids])
                mismatches = int((labels != expected).sum())
                assert mismatches == 0, (dataset, spec["id"], mismatches)
                baseline_summaries.append({"dataset": dataset, "model": spec["id"], "clips": len(ids),
                                           "correct": int((labels == settings["label"]).sum()),
                                           "differences_from_saved_predictions": mismatches})
            if current or spec["id"] in ("svm65", "fusion65"):
                for subset, mask in [("fixed_3000", np.ones(len(ids), dtype=bool)),
                                     ("speaker_independent", independent)]:
                    correct = int((labels[mask] == settings["label"]).sum())
                    count = int(mask.sum())
                    summaries.append({"dataset": dataset, "subset": subset, "model": spec["id"],
                                      "artifact": spec["artifact"], "clips": count, "correct": correct,
                                      "errors": count - correct, "correct_pct": correct / count * 100,
                                      "metric": "recall" if settings["label"] else "specificity"})
                for i, identity in enumerate(ids):
                    dataset_predictions[identity][spec["id"]] = int(labels[i])
                    dataset_predictions[identity][spec["id"] + "_p_singapore"] = float(probabilities[i])
        predictions.extend(dataset_predictions.values())
        print(dataset, "complete;", audits[dataset], flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    args.supporting.mkdir(parents=True, exist_ok=True)
    write_rows(args.supporting / "raw_part2_summary.csv", [r for r in summaries
               if r["dataset"] == "mnsc_part2_unseen_3000"])
    write_rows(args.supporting / "raw_part2_predictions.csv", [r for r in predictions
               if r["dataset"] == "mnsc_part2_unseen_3000"])
    preparation = prepared_part2(args, metadata, loaded, summaries, predictions)
    part2_diagnostics(args, metadata, loaded, study)
    write_rows(args.output / "summary.csv", summaries)
    write_rows(args.output / "predictions.csv", predictions)
    write_rows(args.supporting / "baseline_reproduction.csv", baseline_summaries)
    (args.supporting / "overlap_audit.json").write_text(json.dumps(audits, indent=2) + "\n")
    runtime = {"python": platform.python_version(), "numpy": np.__version__,
               "scikit-learn": sklearn.__version__, "joblib": joblib.__version__,
               "seconds": perf_counter() - started, "reproduce_sha256": sha(__file__),
               "metadata_sha256": sha(HERE / "metadata.json"),
               "baseline_predictions_checked": sum(r["clips"] for r in baseline_summaries),
               "baseline_prediction_differences": sum(r["differences_from_saved_predictions"] for r in baseline_summaries),
               "part2_preparation": preparation,
               "method": "Frozen joblibs; training-consistent trimmed, normalized PCM16 Part 2 audio with fresh Whisper embeddings. Original raw Part 2 protocol retained in supporting/; no fitting or threshold tuning."}
    (args.supporting / "run_metadata.json").write_text(json.dumps(runtime, indent=2) + "\n")
    for row in summaries:
        if row["subset"] == "fixed_3000" or row["dataset"] == "cv_unseen_3000":
            print(row)
    print("Baseline verification:", runtime["baseline_predictions_checked"], "predictions; zero differences.")


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        main()
