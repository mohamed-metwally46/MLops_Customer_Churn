"""Offline checks for the package and declared project configuration."""

import importlib.metadata
import tomllib
from pathlib import Path

import churn

ROOT = Path(__file__).resolve().parents[1]


def test_installed_package_metadata() -> None:
    assert importlib.metadata.version("churn-mlops") == "0.1.0"
    assert Path(churn.__file__).resolve() == ROOT / "src" / "churn" / "__init__.py"


def test_project_configuration() -> None:
    with (ROOT / "pyproject.toml").open("rb") as stream:
        config = tomllib.load(stream)
    assert config["project"]["requires-python"] == ">=3.11,<3.12"
    assert config["tool"]["setuptools"]["packages"]["find"]["where"] == ["src"]
    assert "integration" in config["tool"]["pytest"]["ini_options"]["markers"][0]
