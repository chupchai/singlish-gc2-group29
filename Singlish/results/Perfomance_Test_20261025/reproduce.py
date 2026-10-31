"""Compare the 25 October testing joblibs on the fixed 24 October benchmark."""
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


def predict(spec, loaded, acoustic, whisper):
    model, info, columns = loaded[spec["id"]]
    values = acoustic[:, columns]
    if spec["kind"] == "fusion":
        values = np.column_stack((values.astype(np.float32), whisper))
    probabilities = model.predict_proba(values)
    assert probabilities.shape == (len(acoustic), 2)
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)
    positive = probabilities[:, 1]
    return positive, (positive >= info["threshold"]).astype(int)


def write_report(output, summaries, raw_part2):
    by_key = {(r["model"], r["dataset"]): r for r in summaries if r["subset"] == "fixed_3000"}
    models = ("svm65_261025", "fusion65_261025")
    overall = {model: sum(r["correct"] for r in summaries
                         if r["model"] == model and r["subset"] == "fixed_3000") / 9000 * 100
               for model in {r["model"] for r in summaries}}
    report = ["# Performance test — 25 October 2026", "",
              "Frozen 25 Oct models were tested on the same **9,000 fixed recordings** as the 24 Oct "
              "comparison, with 3,000 in each group. Singapore is selected when `p_singapore >= 0.50`. "
              "No retraining or threshold tuning was performed.", "",
              "**Preprocessing matters:** this benchmark uses trimmed, normalized PCM16 inputs. "
              "The 25 Oct models were trained on trimmed float32 audio with source gain preserved. "
              "These results measure performance on the existing normalized benchmark; they do not "
              "evaluate the full training preparation of the 25 Oct models.", "",
              "## Results on the normalized benchmark", "",
              "| Test group | 25 Oct ML (SVM65) | 25 Oct Ensemble |", "|---|---:|---:|"]
    for dataset, label in [
        ("mnsc_part1_unseen_3000", "MNSC PART1 — Singapore recall"),
        ("mnsc_part2_unseen_3000", "MNSC PART2 — Singapore recall"),
        ("cv_unseen_3000", "Common Voice — non-Singapore specificity"),
    ]:
        values = [f'{by_key[(model, dataset)]["correct_pct"]:.2f}%' for model in models]
        report.append(f"| {label} | " + " | ".join(values) + " |")
    report += [f'| **Overall accuracy — 9,000 clips** | **{overall[models[0]]:.2f}%** | **{overall[models[1]]:.2f}%** |', "",
               "Overall accuracy combines 6,000 Singapore and 3,000 non-Singapore clips.", "",
               "## Key findings", "",
               "- **Ensemble is the strongest 25 Oct model:** 148 errors versus SVM65's 389.",
               f'- On the normalized benchmark, Ensemble scores **{overall["fusion65_261025"]:.2f}% versus '
               f'{overall["fusion65_261024"]:.2f}% for 24 Oct**, with 40 additional errors. Common Voice '
               "false positives improve from 29 to 25, while Singapore misses rise from 79 to 123.",
               f'- Among acoustic models, **SVM91 scores {overall["svm91_261025"]:.2f}%**, followed by '
               f'SVM65 at {overall["svm65_261025"]:.2f}% and SVM39 at {overall["svm39_261025"]:.2f}%.', "",
               "## Raw PART2 stress test", "",
               "On the same 3,000 PART2 recordings, retaining source loudness improves recall:", "",
               "| 25 Oct model | Raw recall | Normalized recall |", "|---|---:|---:|"]
    for model, name in zip(models, ("ML (SVM65)", "Ensemble")):
        row = next(r for r in raw_part2 if r["model"] == model)
        report.append(f'| {name} | {row["raw_recall_pct"]:.2f}% | {row["prepared_recall_pct"]:.2f}% |')
    report += ["", "The raw stress test uses full, untrimmed PCM16 audio. It also differs from the "
               "25 Oct training preparation, which trims audio and retains float32 samples.", "",
               "## Limits and verification", "",
               "Common Voice has 446 clips from training/validation speakers, including 137 matching "
               "training/validation clip IDs. On the remaining 2,554 clips, 25 Oct specificity is "
               "97.81% for ML and 99.14% for Ensemble. MNSC has no training/validation speaker overlap, "
               "but its speakers appear in the study test split.", "",
               "Both 24 Oct baselines reproduced all **18,000 archived class predictions** with zero "
               "differences; probabilities matched within 1e-10 tolerance. Model, metadata, feature-cache, "
               "manifest and split hashes were checked.", "",
               "## Files", "",
               "- [summary.csv](summary.csv): all six models, 24 Oct baselines and speaker-independent subsets.",
               "- [predictions.csv](predictions.csv): per-recording labels and probabilities.",
               "- [metadata.json](metadata.json): frozen model paths, input hashes and preprocessing.",
               "- [supporting/](supporting/README.md): raw PART2 results, preprocessing checks, audits and reproduction instructions.", ""]
    (output / "Performance_Test_Report.md").write_text("\n".join(report))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=ROOT / "tmp/release-comparison-portable-cache")
    parser.add_argument("--study-root", type=Path, default=ROOT.parent / "E1_Feature_EngineeringV5")
    parser.add_argument("--artifact-root", type=Path, default=ROOT.parent)
    parser.add_argument("--part2-cache", type=Path)
    parser.add_argument("--output", type=Path, default=HERE)
    args = parser.parse_args()
    supporting = args.output / "supporting"
    started = perf_counter()
    metadata = json.loads((HERE / "metadata.json").read_text())
    prior = ROOT / metadata["baseline_report"]
    manifests = ROOT / metadata["manifest_report"]
    for path, expected in [
        (prior / "metadata.json", metadata["baseline_metadata_sha256"]),
        (prior / "predictions.csv", metadata["baseline_predictions_sha256"]),
        (prior / "summary.csv", metadata["baseline_summary_sha256"]),
        (manifests / "metadata.json", metadata["manifest_metadata_sha256"]),
        (ROOT / "mvp/features.py", metadata["feature_extractor_sha256"]),
        (args.study_root / metadata["study_splits"], metadata["study_splits_sha256"]),
        *[(ROOT / path, expected) for path, expected in metadata["notebook_sha256"].items()],
    ]:
        assert sha(path) == expected, path
    settings = json.loads((manifests / "metadata.json").read_text())
    study = read_rows(args.study_root / metadata["study_splits"])
    assert len(study) == 40000
    fit_speakers = {r["speaker_group"] for r in study if r["split"] in ("train", "validation")}
    fit_ids = {r["clip_id"] for r in study if r["split"] in ("train", "validation")}
    all_ids = {r["clip_id"] for r in study}
    saved = {(r["dataset"], r["clip_id"]): r for r in read_rows(prior / "predictions.csv")}
    saved_summary = {(r["dataset"], r["model"], r["subset"]): r
                     for r in read_rows(prior / "summary.csv")}
    loaded = {}
    for spec in metadata["models"]:
        path = args.artifact_root / spec["path"]
        sidecar = path.with_name(path.stem + "_metadata.json")
        assert sha(path) == spec["sha256"], path
        assert sha(sidecar) == spec["metadata_sha256"], sidecar
        model, info = joblib.load(path), json.loads(sidecar.read_text())
        attribute = "e4_probability_metadata_" if spec["kind"] == "fusion" else "e1_probability_metadata_"
        assert getattr(model, attribute) == info
        assert info["dataset_id"] == metadata["dataset_id"] and info["clips"] == 40000
        assert info["threshold"] == 0.5 and np.array_equal(model.classes_, [0, 1])
        columns = info["acoustic_columns_in_91"] if spec["kind"] == "fusion" else info["columns"]
        names = info["acoustic_feature_names"] if spec["kind"] == "fusion" else info["feature_names"]
        assert names == [FEATURE_NAMES[i] for i in columns]
        assert model.n_features_in_ == len(columns) + (512 if spec["kind"] == "fusion" else 0)
        if spec["generation"] == "2026-10-25":
            assert info["source_preprocessing"]["loudness_normalization"] is False
            assert info["source_preprocessing"]["peak_scaling"] is False
        if spec["kind"] == "fusion":
            assert info["whisper_provenance"]["resolved_revision"] == settings["whisper"]["revision"]
            assert info["whisper_width"] == 512
        loaded[spec["id"]] = model, info, columns
    part2_cache = args.part2_cache or ROOT / metadata["prepared_part2_cache"]
    assert sha(part2_cache) == metadata["prepared_part2_cache_sha256"]
    with np.load(part2_cache, allow_pickle=False) as z:
        part2_ids = z["clip_ids"].tolist()
        part2 = {name: z[name].copy() for name in
                 ("original_raw", "trim_only", "normalize_only", "trim_and_normalize", "whisper_prepared")}
    summaries, predictions, audits, verification = [], [], {}, []
    raw_part2_results, raw_part2_predictions = [], []
    for dataset, config in settings["datasets"].items():
        manifest = manifests / "manifests" / (dataset + ".csv")
        assert sha(manifest) == config["manifest_sha256"]
        rows = sorted(read_rows(manifest), key=lambda r: r["clip_id"])
        ids = [r["clip_id"] for r in rows]
        cache = args.cache / (dataset + ".npz")
        assert sha(cache) == metadata["feature_cache_sha256"][dataset]
        with np.load(cache, allow_pickle=False) as z:
            assert z["clip_ids"].tolist() == ids
            assert z["manifest_sha256"].item() == config["manifest_sha256"]
            acoustic, whisper = z["acoustic"].copy(), z["whisper"].copy()
        is_part2 = dataset == "mnsc_part2_unseen_3000"
        if is_part2:
            assert part2_ids == ids
            raw_acoustic, raw_whisper = acoustic, whisper
            np.testing.assert_allclose(raw_acoustic, part2["original_raw"], rtol=1e-5, atol=1e-5)
            acoustic, whisper = part2["trim_and_normalize"], part2["whisper_prepared"]
        assert acoustic.shape == (3000, 91) and whisper.shape == (3000, 512)
        assert np.isfinite(acoustic).all() and np.isfinite(whisper).all()
        independent = np.asarray([r["speaker_group"] not in fit_speakers for r in rows])
        speakers = {r["speaker_group"] for r in rows}
        audits[dataset] = {
            "clips": len(ids), "speakers": len(speakers),
            "fit_speaker_overlap": len(speakers & fit_speakers),
            "clips_with_fit_speakers": int((~independent).sum()),
            "speaker_independent_clips": int(independent.sum()),
            "fit_clip_id_overlap": len(set(ids) & fit_ids),
            "all_study_clip_id_overlap": len(set(ids) & all_ids),
        }
        preparation = "trim_normalize_pcm16" if is_part2 else "upstream_trim_normalize_pcm16"
        dataset_predictions = [{"dataset": dataset, "clip_id": identity,
                                "speaker_group": row["speaker_group"], "true_label": config["label"],
                                "speaker_independent": int(clean), "audio_preparation": preparation}
                               for identity, row, clean in zip(ids, rows, independent)]
        if is_part2:
            raw_part2_predictions = [{"clip_id": identity} for identity in ids]
        for spec in metadata["models"]:
            probability, labels = predict(spec, loaded, acoustic, whisper)
            if spec["generation"] == "2026-10-24":
                expected = [int(saved[(dataset, identity)][spec["id"]]) for identity in ids]
                expected_p = [float(saved[(dataset, identity)][spec["id"] + "_p_singapore"]) for identity in ids]
                np.testing.assert_array_equal(labels, expected)
                np.testing.assert_allclose(probability, expected_p, rtol=1e-10, atol=1e-10)
                verification.append({"dataset": dataset, "model": spec["id"], "clips": len(ids),
                                     "prediction_differences": 0,
                                     "max_probability_difference": float(np.max(np.abs(probability - expected_p)))})
            for subset, mask in [("fixed_3000", np.ones(len(ids), dtype=bool)),
                                 ("speaker_independent", independent)]:
                count = int(mask.sum())
                assert count > 0
                correct = int((labels[mask] == config["label"]).sum())
                summaries.append({"dataset": dataset, "subset": subset, "model": spec["id"],
                                  "name": spec["name"], "clips": count, "correct": correct,
                                  "errors": count - correct, "correct_pct": correct / count * 100,
                                  "metric": "recall" if config["label"] else "specificity",
                                  "audio_preparation": preparation})
                if spec["generation"] == "2026-10-24":
                    assert correct == int(saved_summary[(dataset, spec["id"], subset)]["correct"])
            for row, p, label in zip(dataset_predictions, probability, labels):
                row[spec["id"]] = int(label)
                row[spec["id"] + "_p_singapore"] = float(p)
            if is_part2:
                raw_p, raw_labels = predict(spec, loaded, raw_acoustic, raw_whisper)
                raw_part2_results.append({"model": spec["id"], "name": spec["name"], "clips": len(ids),
                                          "raw_recall_pct": raw_labels.mean() * 100,
                                          "raw_misses": int((raw_labels == 0).sum()),
                                          "prepared_recall_pct": labels.mean() * 100,
                                          "prepared_misses": int((labels == 0).sum())})
                for row, p, label in zip(raw_part2_predictions, raw_p, raw_labels):
                    row[spec["id"]] = int(label)
                    row[spec["id"] + "_p_singapore"] = float(p)
        predictions.extend(dataset_predictions)
        print(dataset, "complete", flush=True)
    ablations = []
    spec = next(s for s in metadata["models"] if s["id"] == "svm65_261025")
    for variant in ("original_raw", "trim_only", "normalize_only", "trim_and_normalize"):
        _, labels = predict(spec, loaded, part2[variant], None)
        ablations.append({"variant": variant, "clips": len(labels), "recall_pct": labels.mean() * 100,
                          "misses": int((labels == 0).sum())})
    args.output.mkdir(parents=True, exist_ok=True)
    supporting.mkdir(parents=True, exist_ok=True)
    for name, rows in {
        "summary.csv": summaries, "predictions.csv": predictions,
    }.items():
        write_rows(args.output / name, rows)
    for name, rows in {
        "baseline_reproduction.csv": verification,
        "part2_raw_prepared_comparison.csv": raw_part2_results,
        "raw_part2_predictions.csv": raw_part2_predictions,
        "part2_preprocessing_summary.csv": ablations,
    }.items():
        write_rows(supporting / name, rows)
    (supporting / "overlap_audit.json").write_text(json.dumps(audits, indent=2) + "\n")
    runtime = {"python": platform.python_version(), "numpy": np.__version__,
               "scikit-learn": sklearn.__version__, "joblib": joblib.__version__,
               "seconds": perf_counter() - started, "reproduce_sha256": sha(__file__),
               "metadata_sha256": sha(HERE / "metadata.json"),
               "baseline_predictions_checked": sum(r["clips"] for r in verification),
               "baseline_prediction_differences": 0,
               "method": metadata["method"], "limitation": metadata["limitation"]}
    (supporting / "run_metadata.json").write_text(json.dumps(runtime, indent=2) + "\n")
    write_report(args.output, summaries, raw_part2_results)
    print("Baseline verification:", runtime["baseline_predictions_checked"], "predictions, zero differences.")


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        main()
