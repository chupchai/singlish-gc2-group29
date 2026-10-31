"""Load the ML and Ensemble models and validate their feature contracts."""
import json
import os
from threading import Lock
from pathlib import Path
import joblib
import numpy as np
from .features import FEATURE_NAMES

ROOT = Path(__file__).resolve().parent


def prediction(label, score, probability=None):
    return {"class_id": label,
            "label": "singapore_accent" if label == 1 else "non_singapore_accent",
            "display_label": "Singapore Accent" if label == 1 else "Non-Singapore Accent",
            "probability": probability, "decision_score": score}


def fitted_svm(pipeline):
    if callable(getattr(pipeline, "decision_function", None)):
        return pipeline
    calibrated = getattr(pipeline, "calibrated_classifiers_", [])
    if len(calibrated) != 1:
        raise ValueError("Expected one fitted SVM inside the calibration wrapper")
    return calibrated[0].estimator


def predict_values(pipeline, svm, values):
    score = float(svm.decision_function(values)[0])
    probability = None
    if callable(getattr(pipeline, "predict_proba", None)):
        probabilities = np.asarray(pipeline.predict_proba(values))[0]
        if (probabilities.shape != (2,) or not np.isfinite(probabilities).all()
                or np.any(probabilities < 0) or np.any(probabilities > 1)
                or not np.isclose(probabilities.sum(), 1)):
            raise ValueError("Expected two valid class probabilities")
        # Match both notebooks, including Singapore at an exact 0.50 tie.
        label = int(probabilities[1] >= 0.5)
        probability = float(probabilities[label])
    else:
        label = int(svm.predict(values)[0])
    if label not in (0, 1) or not np.isfinite(score):
        raise ValueError("Invalid SVM decision score or class")
    return prediction(label, score, probability)


class AccentModel:
    def __init__(self):
        self.manifest = json.loads((ROOT / "models/manifest.json").read_text())
        # A relative override is resolved from mvp/, also inside the container.
        path = Path(os.environ.get("MODEL_PATH") or self.manifest["model_file"])
        if not path.is_absolute():
            path = ROOT / path
        artifact = joblib.load(path)
        if isinstance(artifact, dict):
            if not {"model", "columns", "feature_names"} <= artifact.keys():
                raise ValueError("Expected model, columns and feature_names in the export")
            self.pipeline = artifact["model"]
            metadata = artifact
        else:
            # Calibrated exports store the estimator directly, with optional
            # feature metadata alongside it. Width/classes are always checked.
            self.pipeline = artifact
            metadata_path = path.with_name(path.stem + "_metadata.json")
            metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else None
        if metadata is None:
            if self.pipeline.n_features_in_ != len(FEATURE_NAMES):
                raise ValueError("Selected-feature models need columns and feature_names metadata")
            self.columns = list(range(len(FEATURE_NAMES)))
        else:
            self.columns = metadata["columns"]
            if (not self.columns or any(type(i) is not int or not 0 <= i < len(FEATURE_NAMES) for i in self.columns)
                    or len(set(self.columns)) != len(self.columns)
                    or metadata["feature_names"] != [FEATURE_NAMES[i] for i in self.columns]):
                raise ValueError("Model feature names/order do not match the 91-feature extractor")
        if self.pipeline.n_features_in_ != len(self.columns) or not np.array_equal(self.pipeline.classes_, [0, 1]):
            raise ValueError("Invalid fitted input width or class mapping")
        self.decision_estimator = fitted_svm(self.pipeline)
        if (not callable(getattr(self.decision_estimator, "decision_function", None))
                or not np.array_equal(self.decision_estimator.classes_, [0, 1])):
            raise ValueError("ML needs a binary decision_function with classes [0, 1]")
        # No checksum or fixed SVM hyperparameters: compatible exports can be swapped.
        self.manifest["model_version"] = os.environ.get("MODEL_VERSION") or path.stem

    def predict(self, vector):
        if vector.shape != (len(FEATURE_NAMES),) or not np.isfinite(vector).all():
            raise ValueError("Expected 91 finite acoustic features")
        values = vector[self.columns].reshape(1, -1)
        return predict_values(self.pipeline, self.decision_estimator, values)


class EnsembleModel:
    """E4 feature concatenation, including its fitted preprocessing pipeline."""
    def __init__(self):
        manifest = json.loads((ROOT / "models/manifest.json").read_text())
        path = Path(os.environ.get("ENSEMBLE_MODEL_PATH") or manifest["ensemble_file"])
        if not path.is_absolute():
            path = ROOT / path
        self.pipeline = joblib.load(path)
        metadata = json.loads(path.with_name(path.stem + "_metadata.json").read_text())
        self.columns = metadata["acoustic_columns_in_91"]
        self.width = metadata["whisper_width"]
        if (len(self.columns) != 65 or len(set(self.columns)) != 65
                or any(type(i) is not int or not 0 <= i < len(FEATURE_NAMES) for i in self.columns)
                or metadata["acoustic_feature_names"] != [FEATURE_NAMES[i] for i in self.columns]):
            raise ValueError("Ensemble feature names/order do not match the 91-feature extractor")
        self.decision_estimator = fitted_svm(self.pipeline)
        if (self.width != 512 or metadata["positive_label"] != 1
                or self.pipeline.n_features_in_ != len(self.columns) + self.width
                or not np.array_equal(self.pipeline.classes_, [0, 1])
                or not callable(getattr(self.decision_estimator, "decision_function", None))):
            raise ValueError("Invalid Ensemble input width or class mapping")
        self.version = os.environ.get("ENSEMBLE_MODEL_VERSION") or path.stem
        self.whisper_model = metadata["whisper_provenance"]["model"]
        self.whisper_revision = metadata["whisper_provenance"]["resolved_revision"]
        if self.whisper_revision != manifest["whisper_revision"]:
            raise ValueError("Ensemble Whisper revision differs from the manifest")

    def predict(self, vector, embedding):
        if (vector.shape != (len(FEATURE_NAMES),) or not np.isfinite(vector).all()
                or embedding.shape != (self.width,) or not np.isfinite(embedding).all()):
            raise ValueError("Expected 91 finite acoustic features and a 512-value Whisper embedding")
        values = np.concatenate([vector[self.columns].astype(np.float32), embedding]).reshape(1, -1)
        return predict_values(self.pipeline, self.decision_estimator, values)


class AccentModels:
    def __init__(self):
        from .embeddings import WhisperEmbedding
        self.ml = AccentModel()
        self.ensemble = EnsembleModel()
        self.embedding = WhisperEmbedding(self.ensemble.whisper_model, self.ensemble.whisper_revision)
        self.lock = Lock()

    def versions(self):
        return {"ml": self.ml.manifest["model_version"], "ensemble": self.ensemble.version}

    def predict(self, vector, waveform):
        # Avoid concurrent encoder allocations on the small demo host.
        with self.lock:
            ml = self.ml.predict(vector)
            ensemble = self.ensemble.predict(vector, self.embedding(waveform))
        versions = self.versions()
        return {"prediction": ensemble, "final_model": "ensemble",
                "models": {"ml": {"name": "ML", "version": versions["ml"], "prediction": ml},
                           "ensemble": {"name": "Ensemble", "version": versions["ensemble"],
                                        "prediction": ensemble}},
                "agreement": "agree" if ml["class_id"] == ensemble["class_id"] else "disagree",
                "model_version": versions["ensemble"]}
