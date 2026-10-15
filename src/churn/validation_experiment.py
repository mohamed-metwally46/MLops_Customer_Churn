"""Tracked validation-only calibration/threshold decisions, then one final evaluation."""

import hashlib
import json
import os
import platform
import subprocess
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_validate
from threadpoolctl import threadpool_limits

from churn.data import FEATURES, SEED, dataset_path, load_dataset, partitions
from churn.evaluation import (
    calibration_metrics,
    select_threshold,
    threshold_table,
    validate_costs,
)
from churn.tracking import Tracking
from churn.train import SCORING, candidates, scores


@dataclass(frozen=True)
class ValidationConfig:
    folds: int = 5
    fn_cost: float = 1.0
    fp_cost: float = 1.0
    threshold_policy: str = "fixed"

    def __post_init__(self):
        validate_costs(self.fn_cost, self.fp_cost)
        if self.threshold_policy not in {"fixed", "scenario"}:
            raise ValueError("Threshold policy must be fixed or scenario.")
        if not isinstance(self.folds, int) or isinstance(self.folds, bool) or self.folds < 2:
            raise ValueError("CV folds must be an integer >= 2.")

    @classmethod
    def from_environment(cls):
        return cls(
            fn_cost=float(os.getenv("CHURN_FN_COST", "1")),
            fp_cost=float(os.getenv("CHURN_FP_COST", "1")),
            threshold_policy=os.getenv("CHURN_THRESHOLD_POLICY", "fixed"),
        )


def sigmoid_model(pipeline, folds: int):
    """Full preprocessing pipeline is cloned INSIDE calibration CV, never prefit on folds."""
    return CalibratedClassifierCV(
        estimator=clone(pipeline),
        method="sigmoid",
        ensemble=False,
        n_jobs=1,
        cv=StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED),
    )


def run_validation(root: Path, config: ValidationConfig | None = None) -> dict:
    config = config or ValidationConfig.from_environment()
    frame, quality = load_dataset(dataset_path(root))
    baseline = json.loads((root / "reports/model_results.json").read_text(encoding="utf-8"))
    if baseline["selection"]["used"]:
        raise ValueError("This workflow requires the documented full-feature baseline.")
    if quality["sha256"] != baseline["dataset_sha256"]:
        raise ValueError("Dataset differs from the preserved baseline; review before rerunning.")
    baseline_artifact = root / baseline["artifact"]
    if hashlib.sha256(baseline_artifact.read_bytes()).hexdigest() != baseline["artifact_sha256"]:
        raise ValueError("Historical artifact fingerprint mismatch.")
    splits = partitions(frame)
    hashes = split_hashes(splits)
    if hashes != baseline["split_index_hashes"]:
        raise ValueError("Partition fingerprints differ from the historical baseline.")
    with threadpool_limits(limits=1):
        return tracked_validation(splits, baseline, root, config, Tracking(root))


def split_hashes(splits: dict) -> dict:
    return {
        name: hashlib.sha256(np.asarray(x.index, dtype="int64").tobytes()).hexdigest()
        for name, (x, _) in splits.items()
    }


def tracked_validation(
    splits: dict, baseline: dict, root: Path, config: ValidationConfig, tracking: Tracking
) -> dict:
    """Only training-CV-derived baseline fields are used for candidate configuration."""
    train_x, train_y = splits["train"]
    validation_x, validation_y = splits["validation"]
    cv = StratifiedKFold(n_splits=config.folds, shuffle=True, random_state=SEED)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=False
    )
    provenance = {
        "dataset_sha256": baseline["dataset_sha256"],
        "seed": SEED,
        "split_sizes": {k: len(x) for k, (x, _) in splits.items()},
        "split_index_hashes": split_hashes(splits),
        "code_commit": commit.stdout.strip() if commit.returncode == 0 else "uncommitted",
        "code_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path(__file__).parent.glob("*.py")
        },
        "python": platform.python_version(),
        "dependencies": {p: version(p) for p in ["scikit-learn", "numpy", "pandas", "mlflow"]},
        "feature_selection": "all 19 predictors; historical training-CV L1 selector rejected",
        "feature_selection_evidence": baseline["selection"],
        "preprocessing": "fold-fitted median/mode/OHE; numeric scaling only for logistic",
    }
    tags = {
        "dataset_sha256": baseline["dataset_sha256"],
        "code_commit": provenance["code_commit"],
        "purpose": "validation-calibration-threshold",
        "test_previously_evaluated": "true",
    }
    with tracking.run("validation-threshold-experiment", tags=tags) as parent:
        tracking.params(
            parent,
            {
                **asdict(config),
                "seed": SEED,
                "split": "70/15/15 stratified",
                "feature_selection": provenance["feature_selection"],
            },
        )
        tracking.json(parent, "provenance.json", provenance)
        comparison, fitted = {}, {}
        for name, (pipeline, _) in candidates().items():
            pipeline.set_params(**baseline["comparison"][name]["parameters"])
            with tracking.run(name, parent, {**tags, "role": "candidate"}) as run_id:
                tracking.params(
                    run_id,
                    {
                        "model_type": name,
                        "seed": SEED,
                        "folds": config.folds,
                        "feature_selection": "passthrough",
                        "hyperparameter_source": "prior training CV",
                        **pipeline.named_steps["model"].get_params(),
                    },
                )
                cv_results = cross_validate(
                    pipeline,
                    train_x,
                    train_y,
                    scoring=SCORING,
                    cv=cv,
                    n_jobs=1,
                    error_score="raise",
                )
                summary = {
                    m: {
                        "mean": float(cv_results[f"test_{m}"].mean()),
                        "std": float(cv_results[f"test_{m}"].std()),
                    }
                    for m in SCORING
                }
                pipeline.fit(train_x, train_y)
                val = scores(validation_y, pipeline.predict_proba(validation_x)[:, 1])
                for metric, values in summary.items():
                    tracking.metrics(run_id, f"cv_{metric}_", values)
                tracking.metrics(run_id, "validation_", val)
                tracking.json(
                    run_id, "cv_folds.json", {m: cv_results[f"test_{m}"].tolist() for m in SCORING}
                )
                tracking.json(run_id, "validation.json", val)
                comparison[name] = {
                    "run_id": run_id,
                    "cv": summary,
                    "validation": val,
                    "parameters": baseline["comparison"][name]["parameters"],
                }
                fitted[name] = pipeline
        selected_name = max(
            comparison, key=lambda n: comparison[n]["cv"]["average_precision"]["mean"]
        )
        selected = fitted[selected_name]
        raw_probability = selected.predict_proba(validation_x)[:, 1]
        calibrated = sigmoid_model(selected, config.folds).fit(train_x, train_y)
        calibrated_probability = calibrated.predict_proba(validation_x)[:, 1]
        calibration = {
            "raw": calibration_metrics(validation_y, raw_probability),
            "sigmoid": calibration_metrics(validation_y, calibrated_probability),
        }
        use_sigmoid = calibration["sigmoid"]["brier"] < calibration["raw"]["brier"]
        chosen = calibrated if use_sigmoid else selected
        validation_probability = calibrated_probability if use_sigmoid else raw_probability
        table = threshold_table(
            validation_y, validation_probability, fn_cost=config.fn_cost, fp_cost=config.fp_cost
        )
        scenario = select_threshold(
            validation_y, validation_probability, config.fn_cost, config.fp_cost
        )
        threshold = scenario["threshold"] if config.threshold_policy == "scenario" else 0.5
        frozen = {
            "model_type": selected_name,
            "calibration": "sigmoid" if use_sigmoid else "raw",
            "threshold": threshold,
            "threshold_policy": config.threshold_policy,
            "fn_cost": config.fn_cost,
            "fp_cost": config.fp_cost,
            "calibration_rule": "minimum validation Brier; ties retain raw",
            "threshold_rule": "fixed 0.5 unless explicitly opting into validation-grid scenario",
            "model_rule": "maximum training CV AP; frozen prior CV hyperparameters",
        }
        with tracking.run("final-selected", parent, {**tags, "role": "final_selected"}) as final_id:
            tracking.params(
                final_id,
                {
                    **frozen,
                    "seed": SEED,
                    "folds": config.folds,
                    "feature_selection": provenance["feature_selection"],
                    **selected.named_steps["model"].get_params(),
                },
            )
            tracking.json(final_id, "frozen_decision.json", frozen)  # BEFORE any test access
            for metric, values in comparison[selected_name]["cv"].items():
                tracking.metrics(final_id, f"cv_{metric}_", values)
            tracking.json(final_id, "validation_thresholds.json", table)
            tracking.json(final_id, "validation_calibration.json", calibration)
            for name, values in calibration.items():
                tracking.metrics(final_id, f"validation_{name}_", values)
            validation_metrics = scores(validation_y, validation_probability, threshold)
            tracking.metrics(final_id, "validation_", validation_metrics)
            final = clone(chosen).fit(
                pd.concat([train_x, validation_x]), pd.concat([train_y, validation_y])
            )
            test_x, test_y = splits["test"]
            test_metrics = scores(test_y, final.predict_proba(test_x)[:, 1], threshold)
            tracking.metrics(final_id, "test_", test_metrics)  # only final selected run
            tracking.json(final_id, "test_evaluation.json", test_metrics)
            artifact_dir = root / "models" / f"validation-{final_id}"
            artifact_dir.mkdir(parents=True, exist_ok=False)
            artifact = artifact_dir / "pipeline.joblib"
            joblib.dump(final, artifact)
            tracking.client.log_artifact(final_id, str(artifact), artifact_path="model")
            retrieval = artifact_dir / "retrieved"
            retrieval.mkdir()
            downloaded = Path(
                tracking.client.download_artifacts(
                    final_id, "model/pipeline.joblib", str(retrieval)
                )
            )
            sha = hashlib.sha256(artifact.read_bytes()).hexdigest()
            if hashlib.sha256(downloaded.read_bytes()).hexdigest() != sha:
                raise RuntimeError("Retrieved MLflow model fingerprint mismatch.")
            np.testing.assert_allclose(
                joblib.load(downloaded).predict_proba(validation_x.iloc[:2]),
                final.predict_proba(validation_x.iloc[:2]),
            )
            result = {
                "provenance": provenance,
                "comparison": comparison,
                "config": asdict(config),
                "frozen_decision": frozen,
                "calibration": calibration,
                "thresholds": table,
                "scenario": scenario,
                "validation": validation_metrics,
                "test": test_metrics,
                "parent_run_id": parent,
                "final_run_id": final_id,
                "artifact_uri": f"runs:/{final_id}/model/pipeline.joblib",
                "artifact_sha256": sha,
                "input_features": FEATURES,
                "artifact": artifact.relative_to(root).as_posix(),
                "retrieval_verified": True,
            }
            tracking.json(final_id, "metadata.json", result)
            (artifact_dir / "metadata.json").write_text(
                json.dumps(result, indent=2), encoding="utf-8"
            )
            from churn.validation_report import write_report

            report_dir = root / "reports/milestone3" / parent
            write_report(result, report_dir)
            tracking.client.log_artifacts(final_id, str(report_dir), artifact_path="reports")
        return result


if __name__ == "__main__":
    outcome = run_validation(Path.cwd())
    print("Completed run:", outcome["parent_run_id"], "artifact:", outcome["artifact_uri"])
