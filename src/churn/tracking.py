"""Explicit MLflow client, local SQLite default, and immutable artifact references."""

import os
from contextlib import contextmanager
from pathlib import Path

from mlflow.tracking import MlflowClient


class Tracking:
    def __init__(self, root: Path, uri: str | None = None, experiment: str | None = None):
        self.client = MlflowClient(
            tracking_uri=uri
            or os.getenv(
                "MLFLOW_TRACKING_URI", f"sqlite:///{(root / 'mlflow.db').resolve().as_posix()}"
            )
        )
        name = experiment or os.getenv("MLFLOW_EXPERIMENT_NAME", "churn-validation")
        existing = self.client.get_experiment_by_name(name)
        self.experiment_id = (
            existing.experiment_id
            if existing
            else self.client.create_experiment(
                name, artifact_location=(root / "mlartifacts").resolve().as_uri()
            )
        )

    @contextmanager
    def run(self, name: str, parent: str | None = None, tags: dict | None = None):
        run_tags = {"mlflow.runName": name, **(tags or {})}
        if parent:
            run_tags["mlflow.parentRunId"] = parent
        run_id = self.client.create_run(self.experiment_id, tags=run_tags).info.run_id
        try:
            yield run_id
        except BaseException:
            self.client.set_terminated(run_id, status="FAILED")
            raise
        else:
            self.client.set_terminated(run_id, status="FINISHED")

    def params(self, run_id: str, values: dict) -> None:
        for key, value in values.items():
            self.client.log_param(run_id, key, str(value))

    def metrics(self, run_id: str, prefix: str, values: dict) -> None:
        for key, value in values.items():
            if isinstance(value, (int, float)):
                self.client.log_metric(run_id, f"{prefix}{key}", float(value))

    def json(self, run_id: str, name: str, values: dict | list) -> None:
        self.client.log_dict(run_id, values, name)
