"""Verify and replay recorded exploration results without fitting or test evaluation."""

import hashlib
import json
from pathlib import Path


def verified_result(root: Path, report: Path) -> dict:
    result = json.loads(report.read_text(encoding="utf-8"))
    artifact = root / result["artifact"]
    if hashlib.sha256(artifact.read_bytes()).hexdigest() != result["artifact_sha256"]:
        raise ValueError("Recorded experiment model fingerprint mismatch.")
    if result["artifact_uri"] != f"runs:/{result['final_run_id']}/model/pipeline.joblib":
        raise ValueError("Recorded experiment identity mismatch.")
    result["verification_replay"] = True
    return result
