"""Offline bundle and build-context checks; these do not simulate a Docker build."""

import importlib.util
from pathlib import Path

import pytest
from docker.utils.build import exclude_paths

from churn.model_loader import ModelSettings, load_model
from churn.schemas import Customer, example_customer

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_synthetic_bundle_is_self_contained_and_never_overwritten(tmp_path):
    prepare = load_script("prepare_model_bundle")
    output = tmp_path / "ci-bundle"
    manifest = prepare.prepare_bundle(output, tmp_path, synthetic=True)
    assert manifest["synthetic_ci_only"] is True
    assert {p.name for p in output.iterdir()} == {"metadata.json", "pipeline.joblib"}
    loaded = load_model(ModelSettings(output, output / "metadata.json"))
    probability = loaded.predict(Customer.model_validate(example_customer())).churn_probability
    assert 0 <= probability <= 1
    with pytest.raises(FileExistsError):
        prepare.prepare_bundle(output, tmp_path, synthetic=True)


def test_production_export_preserves_original_bytes(tmp_path):
    import hashlib
    import json

    prepare = load_script("prepare_model_bundle")
    source = tmp_path / "source"
    manifest = prepare.prepare_bundle(source, tmp_path, synthetic=True)
    report = tmp_path / "report.json"
    metadata = {**manifest, "artifact": "source/pipeline.joblib"}
    report.write_text(json.dumps(metadata))
    prepare.PINNED_REPORT = "report.json"
    before = hashlib.sha256((source / "pipeline.joblib").read_bytes()).hexdigest()
    exported = prepare.prepare_bundle(tmp_path / "export", tmp_path)
    assert exported["artifact_sha256"] == before
    assert hashlib.sha256((source / "pipeline.joblib").read_bytes()).hexdigest() == before
    assert exported["synthetic_ci_only"] is False


def test_docker_build_context_allowlist_excludes_private_and_training_files():
    patterns = [
        line
        for line in (ROOT / ".dockerignore").read_text().splitlines()
        if line and not line.startswith("#")
    ]
    files = {Path(p).as_posix() for p in exclude_paths(str(ROOT), patterns) if (ROOT / p).is_file()}
    expected = {
        "Dockerfile",
        ".dockerignore",
        "README.md",
        "pyproject.toml",
        "requirements/build.txt",
        "requirements/runtime.lock",
    } | {
        f"src/churn/{name}.py"
        for name in ["__init__", "api", "data", "model_loader", "preprocessing", "schemas"]
    }
    assert files == expected


def test_ci_configuration_parses_and_has_no_publishing_permissions():
    load_script("validate_ci").validate_workflow(ROOT)


def test_ci_rejects_basetemp_that_depends_on_local_venv(tmp_path):
    workflow = tmp_path / ".github/workflows/ci.yml"
    workflow.parent.mkdir(parents=True)
    workflow.write_text(
        (ROOT / ".github/workflows/ci.yml")
        .read_text()
        .replace("${{ runner.temp }}/churn-pytest", ".venv/pytest-ci")
    )
    with pytest.raises(AssertionError):
        load_script("validate_ci").validate_workflow(tmp_path)


@pytest.mark.parametrize("failure", ["closed", "refused", "timeout", "url"])
def test_smoke_wait_retries_startup_connection_failures(monkeypatch, failure):
    from http.client import RemoteDisconnected
    from unittest.mock import Mock
    from urllib.error import URLError

    smoke = load_script("smoke_docker")
    errors = {
        "closed": RemoteDisconnected("Remote end closed connection without response"),
        "refused": ConnectionRefusedError(),
        "timeout": TimeoutError(),
        "url": URLError("starting"),
    }
    request = Mock(side_effect=[errors[failure], (200, {})])
    monkeypatch.setattr(smoke, "get", request)
    monkeypatch.setattr(smoke.time, "sleep", lambda _: None)
    smoke.wait_for_liveness("http://localhost")
    assert request.call_count == 2


def test_smoke_wait_is_bounded(monkeypatch):
    smoke = load_script("smoke_docker")
    monkeypatch.setattr(smoke, "get", lambda *_: (503, {}))
    with pytest.raises(RuntimeError, match="within 0 seconds"):
        smoke.wait_for_liveness("http://localhost", timeout=0)
