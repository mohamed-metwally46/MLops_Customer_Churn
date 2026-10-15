"""Offline YAML and security-contract checks; not a substitute for a hosted CI run."""

import re
from pathlib import Path

import yaml


def validate_workflow(root: Path) -> None:
    # BaseLoader preserves the YAML 1.2 event key `on` instead of treating it as a Boolean.
    workflow = yaml.load((root / ".github/workflows/ci.yml").read_text(), Loader=yaml.BaseLoader)
    assert set(workflow["on"]) == {"pull_request", "push"}
    assert workflow["on"]["push"]["branches"] == ["main"]
    assert workflow["permissions"] == {"contents": "read"}
    steps = workflow["jobs"]["checks"]["steps"]
    for step in steps:
        if "uses" in step:
            assert re.fullmatch(r"actions/(checkout|setup-python)@[0-9a-f]{40}", step["uses"])
    checkout = next(s for s in steps if s.get("uses", "").startswith("actions/checkout@"))
    assert checkout["with"]["persist-credentials"] == "false"
    commands = "\n".join(step.get("run", "") for step in steps)
    for expected in [
        "pytest",
        "ruff check",
        "ruff format --check",
        "pip check",
        "docker build",
        "prepare_model_bundle.py --synthetic",
        "smoke_docker.py",
    ]:
        assert expected in commands
    assert not any(token in commands for token in ["docker push", "docker login", "secrets."])
    for path in [
        "Dockerfile",
        "requirements/runtime.lock",
        "scripts/prepare_model_bundle.py",
        "scripts/smoke_docker.py",
    ]:
        assert (root / path).is_file()


if __name__ == "__main__":
    validate_workflow(Path(__file__).resolve().parents[1])
    print("Workflow YAML and local security/configuration checks passed.")
