"""Replay must verify existing artifacts and never overwrite historical evidence."""

import hashlib
import json

import pytest

from churn.notebook_replay import verified_result


@pytest.mark.parametrize("issue", [None, "checksum", "identity"])
def test_verification_replay_preserves_results_and_checks_identity(tmp_path, issue):
    artifact = tmp_path / "fixture.joblib"
    artifact.write_bytes(b"synthetic-only; replay does not deserialize")
    metadata = {
        "artifact": artifact.name,
        "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "final_run_id": "a" * 32,
        "artifact_uri": "runs:/" + "a" * 32 + "/model/pipeline.joblib",
    }
    if issue == "checksum":
        metadata["artifact_sha256"] = "0" * 64
    elif issue == "identity":
        metadata["artifact_uri"] = "runs:/different/model/pipeline.joblib"
    report = tmp_path / "results.json"
    report.write_text(json.dumps(metadata))
    before = report.read_bytes()
    if issue:
        with pytest.raises(ValueError):
            verified_result(tmp_path, report)
    else:
        assert verified_result(tmp_path, report)["verification_replay"] is True
    assert report.read_bytes() == before
