"""Load only explicitly configured, trusted, SHA-pinned local model bundles."""

import hashlib
import io
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from churn.data import CATEGORIES, FEATURES, NUMERIC
from churn.schemas import CustomerBase, Prediction, example_customer

PINNED_REPORT = "reports/milestone3/a644309446d34b7ca25ea160124aeba6/results.json"


class ModelUnavailable(RuntimeError):
    """Safe, public-facing startup error; underlying exception text must not be logged."""


class PredictionFailed(RuntimeError):
    """Safe inference error without a payload, path or stack trace."""


@dataclass(frozen=True)
class ModelSettings:
    root: Path
    metadata_path: Path
    artifact_path: Path | None = None
    threshold: float | None = None

    def __post_init__(self) -> None:
        if self.threshold is not None and (
            isinstance(self.threshold, bool)
            or not np.isfinite(self.threshold)
            or not 0 <= self.threshold <= 1
        ):
            raise ValueError("CHURN_THRESHOLD must be finite and in [0, 1].")

    @classmethod
    def from_environment(cls) -> "ModelSettings":
        root = Path(os.getenv("CHURN_PROJECT_ROOT", str(Path.cwd()))).resolve()
        raw_threshold = os.getenv("CHURN_THRESHOLD")
        return cls(
            root=root,
            metadata_path=root / os.getenv("CHURN_MODEL_METADATA", PINNED_REPORT),
            artifact_path=(
                root / os.environ["CHURN_MODEL_PATH"] if "CHURN_MODEL_PATH" in os.environ else None
            ),
            threshold=float(raw_threshold) if raw_threshold is not None else None,
        )


@dataclass(frozen=True)
class LoadedModel:
    estimator: Any
    model_name: str
    model_version: str
    artifact_sha256: str
    threshold: float
    positive_index: int

    def probability(self, frame: pd.DataFrame) -> float:
        try:
            output = np.asarray(self.estimator.predict_proba(frame), dtype=float)
            if output.shape != (1, 2) or not np.isfinite(output).all():
                raise ValueError("Invalid probability shape or values.")
            if ((output < 0) | (output > 1)).any() or not np.allclose(output.sum(axis=1), 1):
                raise ValueError("Invalid probability distribution.")
            return float(output[0, self.positive_index])
        except Exception:
            raise PredictionFailed("Prediction could not be completed.") from None

    def predict(self, customer: CustomerBase) -> Prediction:
        probability = self.probability(customer.to_frame())
        return Prediction(
            prediction="Yes" if probability >= self.threshold else "No",
            churn_probability=probability,
            threshold=self.threshold,
            model_version=self.model_version,
        )


def load_model(settings: ModelSettings) -> LoadedModel:
    """Read/hash bytes before deserialization; no remote lookup, fitting or registry resolution."""
    try:
        metadata = json.loads(settings.metadata_path.read_text(encoding="utf-8"))
        if metadata["input_features"] != FEATURES:
            raise ValueError("Manifest feature contract mismatch.")
        decision = metadata["frozen_decision"]
        threshold = settings.threshold if settings.threshold is not None else decision["threshold"]
        if isinstance(threshold, bool) or not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("Invalid threshold.")
        version = metadata["final_run_id"]
        if (
            not isinstance(version, str)
            or len(version) != 32
            or any(c not in "0123456789abcdef" for c in version)
        ):
            raise ValueError("Invalid immutable run identity.")
        artifact = settings.artifact_path or settings.root / metadata["artifact"]
        data = artifact.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest != metadata["artifact_sha256"]:
            raise ValueError("Model fingerprint mismatch.")
        estimator = joblib.load(io.BytesIO(data))
        if list(estimator.feature_names_in_) != FEATURES:
            raise ValueError("Fitted model feature contract mismatch.")
        # This milestone supports the inspected complete, uncalibrated pipeline.
        preprocess = estimator.named_steps["preprocess"]
        if list(preprocess.transformers_[0][2]) != NUMERIC or list(
            preprocess.transformers_[1][2]
        ) != list(CATEGORIES):
            raise ValueError("Fitted preprocessing contract mismatch.")
        class_names = {
            "RandomForest": "RandomForestClassifier",
            "LogisticRegression": "LogisticRegression",
            "HistGradientBoosting": "HistGradientBoostingClassifier",
            "Dummy": "DummyClassifier",
        }
        if type(estimator.named_steps["model"]).__name__ != class_names.get(decision["model_type"]):
            raise ValueError("Model identity mismatch.")
        classes = np.asarray(estimator.classes_)
        if classes.shape != (2,) or set(classes.tolist()) != {0, 1}:
            raise ValueError("Expected verified binary No=0/Yes=1 class mapping.")
        positive = int(np.flatnonzero(classes == 1)[0])
        loaded = LoadedModel(
            estimator, decision["model_type"], version, digest, float(threshold), positive
        )
        from churn.schemas import Customer

        loaded.predict(Customer.model_validate(example_customer()))  # startup usability probe
        return loaded
    except Exception:
        raise ModelUnavailable("Configured model could not be loaded or validated.") from None
