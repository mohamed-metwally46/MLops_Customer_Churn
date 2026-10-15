"""Offline API tests on tiny deterministic saved pipelines, without the supplied dataset."""

import hashlib
import json
from dataclasses import replace

import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient
from test_analysis import telco

from churn.api import create_app
from churn.data import FEATURES
from churn.model_loader import ModelSettings, load_model
from churn.schemas import Customer, example_customer
from churn.train import candidates


@pytest.fixture
def saved_model(tmp_path):
    frame = telco.__wrapped__()
    pipeline, _ = candidates()["RandomForest"]
    pipeline.set_params(model__n_estimators=5, model__max_depth=3)
    pipeline.fit(frame[FEATURES], frame["Churn"].eq("Yes").astype(int))
    artifact = tmp_path / "pipeline.joblib"
    joblib.dump(pipeline, artifact)
    manifest = {
        "artifact": "pipeline.joblib",
        "input_features": FEATURES,
        "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "final_run_id": "a" * 32,
        "frozen_decision": {"model_type": "RandomForest", "threshold": 0.5},
    }
    metadata = tmp_path / "metadata.json"
    metadata.write_text(json.dumps(manifest))
    return ModelSettings(tmp_path, metadata), pipeline


def test_health_ready_info_and_direct_probability_parity(saved_model):
    settings, pipeline = saved_model
    with TestClient(create_app(settings)) as client:
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/ready").json() == {"status": "ready"}
        info = client.get("/model-info").json()
        assert info["model_name"] == "RandomForest"
        assert info["model_version"] == "a" * 32
        assert info["threshold"] == 0.5
        assert set(info["input_schema"]["required"]) == set(FEATURES)
        assert str(settings.root) not in json.dumps(info)
        response = client.post("/predict", json=example_customer())
        assert response.status_code == 200
        probability = pipeline.predict_proba(
            Customer.model_validate(example_customer()).to_frame()
        )[0, 1]
        assert response.json()["churn_probability"] == pytest.approx(probability, abs=1e-12)
        assert response.json()["prediction"] == ("Yes" if probability >= 0.5 else "No")
        assert client.get("/openapi.json").status_code == 200


@pytest.mark.parametrize(
    "field,value",
    [
        ("tenure", -1),
        ("tenure", 1.5),
        ("tenure", "12"),
        ("tenure", True),
        ("MonthlyCharges", -0.1),
        ("MonthlyCharges", 1e300),
        ("TotalCharges", 1e300),
        ("tenure", 10**200),
        ("MonthlyCharges", "65"),
        ("TotalCharges", "780"),
        ("SeniorCitizen", True),
        ("SeniorCitizen", 1.0),
        ("SeniorCitizen", "1"),
        ("SeniorCitizen", 2),
        ("Contract", "secret-invalid-category"),
        ("gender", "secret-invalid-category"),
        ("Churn", "secret-value"),
        ("customerID", "secret-value"),
        ("secret-key", "secret-value"),
    ],
)
def test_invalid_fields_are_rejected_without_echoing_values(saved_model, field, value, caplog):
    settings, _ = saved_model
    payload = {**example_customer(), field: value}
    with TestClient(create_app(settings)) as client:
        response = client.post("/predict", json=payload)
        assert response.status_code == 422
        assert "secret" not in response.text
        assert "secret" not in caplog.text
        assert all(
            "input" not in error and "ctx" not in error for error in response.json()["detail"]
        )


def test_missing_malformed_and_nonfinite_requests(saved_model):
    settings, _ = saved_model
    with TestClient(create_app(settings)) as client:
        missing = example_customer()
        missing.pop("tenure")
        assert client.post("/predict", json=missing).status_code == 422
        assert client.post("/predict", json=[]).status_code == 422
        response = client.post(
            "/predict", content='{"secret-value":', headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422 and "secret-value" not in response.text
        encoded = json.dumps({**example_customer(), "MonthlyCharges": float("nan")})
        assert (
            client.post(
                "/predict", content=encoded, headers={"Content-Type": "application/json"}
            ).status_code
            == 422
        )


def test_nulls_and_blank_totalcharges_use_fitted_imputation(saved_model):
    settings, pipeline = saved_model
    with TestClient(create_app(settings)) as client:
        for payload in [{c: None for c in FEATURES}, {**example_customer(), "TotalCharges": " "}]:
            response = client.post("/predict", json=payload)
            assert response.status_code == 200
            direct = pipeline.predict_proba(Customer.model_validate(payload).to_frame())[0, 1]
            assert response.json()["churn_probability"] == pytest.approx(direct)


@pytest.mark.parametrize("threshold,label", [(0.0, "Yes"), (1.0, "No")])
def test_configured_threshold(saved_model, threshold, label):
    settings, _ = saved_model
    with TestClient(create_app(replace(settings, threshold=threshold))) as client:
        result = client.post("/predict", json=example_customer()).json()
        assert result["threshold"] == threshold
        assert result["prediction"] == label


def test_threshold_tie_predicts_yes(saved_model, monkeypatch):
    settings, _ = saved_model
    with TestClient(create_app(settings)) as client:
        monkeypatch.setattr(
            client.app.state.model.estimator, "predict_proba", lambda frame: np.array([[0.5, 0.5]])
        )
        assert client.post("/predict", json=example_customer()).json()["prediction"] == "Yes"


def test_reversed_class_order_uses_positive_class_column(saved_model):
    settings, model = saved_model
    model.named_steps["model"].classes_ = np.array([1, 0])
    artifact = settings.root / "pipeline.joblib"
    joblib.dump(model, artifact)
    metadata = json.loads(settings.metadata_path.read_text())
    metadata["artifact_sha256"] = hashlib.sha256(artifact.read_bytes()).hexdigest()
    settings.metadata_path.write_text(json.dumps(metadata))
    with TestClient(create_app(settings)) as client:
        actual = client.post("/predict", json=example_customer()).json()["churn_probability"]
        expected = model.predict_proba(Customer.model_validate(example_customer()).to_frame())[0, 0]
        assert actual == pytest.approx(expected)


@pytest.mark.parametrize(
    "failure", ["missing", "corrupt", "checksum", "metadata", "features", "classes"]
)
def test_load_failure_is_live_but_not_ready(saved_model, failure, caplog):
    settings, model = saved_model
    metadata = json.loads(settings.metadata_path.read_text())
    artifact = settings.root / "pipeline.joblib"
    if failure == "missing":
        settings = replace(settings, artifact_path=settings.root / "missing.joblib")
    elif failure == "corrupt":
        artifact.write_bytes(b"secret-corrupt-bytes")
        metadata["artifact_sha256"] = hashlib.sha256(artifact.read_bytes()).hexdigest()
    elif failure == "checksum":
        metadata["artifact_sha256"] = "0" * 64
    elif failure == "metadata":
        metadata = {}
    elif failure == "features":
        metadata["input_features"] = ["Churn"]
    else:
        model.named_steps["model"].classes_ = np.array(["No", "Yes"])
        joblib.dump(model, artifact)
        metadata["artifact_sha256"] = hashlib.sha256(artifact.read_bytes()).hexdigest()
    settings.metadata_path.write_text(json.dumps(metadata))
    with TestClient(create_app(settings)) as client:
        assert client.get("/health").status_code == 200
        for path in ["/ready", "/model-info"]:
            response = client.get(path)
            assert response.status_code == 503
            assert str(settings.root) not in response.text
        assert client.post("/predict", json=example_customer()).status_code == 503
    assert "secret" not in caplog.text and str(settings.root) not in caplog.text


def test_checksum_mismatch_is_rejected_before_deserialization(saved_model, monkeypatch):
    settings, _ = saved_model
    metadata = json.loads(settings.metadata_path.read_text())
    metadata["artifact_sha256"] = "0" * 64
    settings.metadata_path.write_text(json.dumps(metadata))
    monkeypatch.setattr(
        joblib, "load", lambda *args: pytest.fail("Must not deserialize mismatched bytes")
    )
    with TestClient(create_app(settings)) as client:
        assert client.get("/ready").status_code == 503


def test_load_once_and_never_fit_during_startup_or_requests(saved_model, monkeypatch):
    settings, model = saved_model
    original_load = joblib.load
    calls = []

    def load_once(*args, **kwargs):
        calls.append("deserialize")
        return original_load(*args, **kwargs)

    monkeypatch.setattr(joblib, "load", load_once)
    monkeypatch.setattr(type(model), "fit", lambda *a, **k: pytest.fail("Inference must not fit"))
    with TestClient(create_app(settings)) as client:
        for _ in range(3):
            assert client.post("/predict", json=example_customer()).status_code == 200
    assert calls == ["deserialize"]


def test_prediction_failure_is_safe_and_clears_readiness(saved_model, monkeypatch, caplog):
    settings, _ = saved_model
    with TestClient(create_app(settings)) as client:
        monkeypatch.setattr(
            client.app.state.model.estimator,
            "predict_proba",
            lambda frame: np.array([[np.nan, np.nan]]),
        )
        response = client.post("/predict", json=example_customer())
        assert response.status_code == 500
        assert response.json() == {"detail": "Prediction could not be completed."}
        assert client.get("/ready").status_code == 503
        assert client.get("/health").status_code == 200
    assert "Traceback" not in caplog.text


@pytest.mark.parametrize("threshold", [-1, 2, float("nan"), float("inf"), True])
def test_invalid_configuration(saved_model, threshold):
    settings, _ = saved_model
    with pytest.raises(ValueError):
        replace(settings, threshold=threshold)


@pytest.mark.integration
def test_pinned_real_artifact_api_parity():
    """Optional real artifact smoke test; no dataset, network, training or MLflow server."""
    from pathlib import Path

    from churn.model_loader import PINNED_REPORT

    root = Path(__file__).resolve().parents[1]
    settings = ModelSettings(root, root / PINNED_REPORT)
    if not settings.metadata_path.is_file():
        pytest.skip("Pinned metadata not available.")
    metadata = json.loads(settings.metadata_path.read_text())
    if not (root / metadata["artifact"]).is_file():
        pytest.skip("Pinned artifact not available.")
    direct = load_model(settings).predict(Customer.model_validate(example_customer()))
    with TestClient(create_app(settings)) as client:
        assert client.get("/ready").status_code == 200
        assert client.post("/predict", json=example_customer()).json() == direct.model_dump()
