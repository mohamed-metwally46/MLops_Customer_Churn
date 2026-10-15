"""Offline threshold/calibration/MLflow tests with synthetic data and local stores."""

import hashlib
import json

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from test_analysis import telco

from churn.data import partitions
from churn.evaluation import (
    calibration_metrics,
    select_threshold,
    threshold_metrics,
    threshold_table,
)
from churn.tracking import Tracking
from churn.train import candidates
from churn.validation_experiment import ValidationConfig, sigmoid_model, tracked_validation


def test_threshold_metrics_include_ties_and_volume():
    row = threshold_metrics([0, 0, 1, 1], [0.1, 0.5, 0.4, 0.9], 0.5)
    assert (row["tn"], row["fp"], row["fn"], row["tp"]) == (1, 1, 1, 1)
    assert row["precision"] == row["recall"] == row["f1"] == 0.5
    assert row["flagged_percent"] == 50
    none = threshold_metrics([0, 1], [0.1, 0.2], 1.0)
    assert none["precision"] == none["recall"] == none["f1"] == 0
    assert none["flagged_percent"] == 0


def test_costs_change_scenario_and_deterministic_tie_break():
    y, p, grid = [0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8], [0.3, 0.5, 0.9]
    assert select_threshold(y, p, 5, 1, grid)["threshold"] == 0.3
    assert select_threshold(y, p, 1, 5, grid)["threshold"] == 0.5
    assert select_threshold(y, p, 1, 1, grid)["threshold"] == 0.5
    rows = threshold_table(y, p)
    assert len(rows) == 21
    assert rows[0]["flagged_percent"] == 100


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_thresholds(threshold):
    with pytest.raises(ValueError, match="Threshold"):
        threshold_metrics([0, 1], [0.1, 0.9], threshold)


@pytest.mark.parametrize(
    "y,p",
    [
        ([], []),
        ([0, 0], [0.1, 0.2]),
        ([0, 2], [0.1, 0.2]),
        ([0, 1], [0.2]),
        ([0, 1], [float("nan"), 0.2]),
        ([0, 1], [0.2, 1.1]),
        ([[0, 1]], [0.2, 0.3]),
    ],
)
def test_invalid_probability_inputs(y, p):
    with pytest.raises(ValueError):
        calibration_metrics(y, p)


@pytest.mark.parametrize("fn,fp", [(-1, 1), (1, -1), (0, 0), (float("inf"), 1)])
def test_invalid_costs(fn, fp):
    with pytest.raises(ValueError, match="Costs"):
        ValidationConfig(fn_cost=fn, fp_cost=fp)


def test_calibration_scores_and_boundary_bins():
    perfect = calibration_metrics([0, 0, 1, 1], [0, 0, 1, 1])
    assert perfect["brier"] == perfect["ece_uniform"] == 0
    assert perfect["log_loss"] < 1e-10
    assert [b["count"] for b in perfect["reliability"]] == [2, 2]
    half = calibration_metrics([0, 1], [0.5, 0.5])
    assert half["brier"] == 0.25
    assert half["log_loss"] == pytest.approx(np.log(2))
    with pytest.raises(ValueError):
        calibration_metrics([0, 1], [0.1, 0.9], bins=1)
    with pytest.raises(ValueError):
        threshold_table([0, 1], [0.1, 0.9], thresholds=[0.5, 0.5])
    with pytest.raises(ValueError):
        ValidationConfig(threshold_policy="test-optimal")


def test_calibration_keeps_preprocessing_inside_fold_estimator():
    pipeline, _ = candidates()["LogisticRegression"]
    calibration = sigmoid_model(pipeline, 2)
    assert isinstance(calibration.estimator, Pipeline)
    assert not hasattr(calibration.estimator.named_steps["preprocess"], "transformers_")
    assert calibration.ensemble is False
    assert calibration.cv.random_state == 42
    clone(calibration)


def test_tracking_failure_and_immutable_retrieval(tmp_path):
    store = Tracking(tmp_path, experiment="offline-failure-test")
    with pytest.raises(RuntimeError):
        with store.run("failed") as failed:
            raise RuntimeError("synthetic failure")
    assert store.client.get_run(failed).info.status == "FAILED"
    with store.run("ok") as success:
        store.params(success, {"seed": 42})
        store.metrics(success, "validation_", {"brier": 0.2})
        store.json(success, "summary.json", {"aggregate": True})
    assert store.client.get_run(success).info.status == "FINISHED"
    destination = tmp_path / "download"
    destination.mkdir()
    downloaded = store.client.download_artifacts(success, "summary.json", str(destination))
    assert json.loads(open(downloaded).read()) == {"aggregate": True}


def test_tracked_workflow_never_fits_test_and_logs_test_only_for_final(tmp_path, monkeypatch):
    from sklearn.calibration import CalibratedClassifierCV
    from threadpoolctl import threadpool_limits

    from churn import validation_experiment as experiment

    frame = telco.__wrapped__()
    split = partitions(frame)
    baseline = {
        "dataset_sha256": "synthetic",
        "selection": {"used": False},
        "comparison": {"LogisticRegression": {"parameters": {}}},
    }
    pipeline, _ = candidates()["LogisticRegression"]
    monkeypatch.setattr(experiment, "candidates", lambda: {"LogisticRegression": (pipeline, {})})
    original_fit, original_predict = Pipeline.fit, Pipeline.predict_proba
    test_indices = set(split["test"][0].index)
    calls = []
    prediction_depth = [0]

    def fit_spy(self, x, y=None, **kwargs):
        if hasattr(x, "index"):
            assert not (set(x.index) & test_indices)
        return original_fit(self, x, y, **kwargs)

    def audit_predict(method):
        def predict_spy(self, x, **kwargs):
            is_test = hasattr(x, "index") and set(x.index) == test_indices
            if is_test and prediction_depth[0] == 0:
                calls.append("test")
            prediction_depth[0] += 1
            try:
                return method(self, x, **kwargs)
            finally:
                prediction_depth[0] -= 1

        return predict_spy

    monkeypatch.setattr(Pipeline, "fit", fit_spy)
    monkeypatch.setattr(Pipeline, "predict_proba", audit_predict(original_predict))
    monkeypatch.setattr(
        CalibratedClassifierCV, "predict_proba", audit_predict(CalibratedClassifierCV.predict_proba)
    )
    store = Tracking(tmp_path, experiment="offline-workflow-test")
    with threadpool_limits(limits=1):
        result = tracked_validation(split, baseline, tmp_path, ValidationConfig(folds=2), store)
    assert calls == ["test"]
    assert result["frozen_decision"]["threshold"] == 0.5
    assert result["retrieval_verified"] is True
    for run in store.client.search_runs([store.experiment_id]):
        test_metrics = [k for k in run.data.metrics if k.startswith("test_")]
        assert bool(test_metrics) == (run.info.run_id == result["final_run_id"])
    artifact = tmp_path / result["artifact"]
    assert hashlib.sha256(artifact.read_bytes()).hexdigest() == result["artifact_sha256"]
