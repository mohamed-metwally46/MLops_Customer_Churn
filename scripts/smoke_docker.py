"""Verify container HTTP parity, fail-closed loading and runtime image contents."""

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import uuid4

import numpy as np

from churn.model_loader import ModelSettings, load_model
from churn.schemas import Customer, example_customer


def docker(*args: str) -> str:
    return subprocess.run(
        ["docker", *args], check=True, capture_output=True, text=True, timeout=120
    ).stdout.strip()


def get(base: str, endpoint: str) -> tuple[int, dict]:
    try:
        with urlopen(base + endpoint, timeout=2) as response:
            return response.status, json.load(response)
    except HTTPError as error:
        return error.code, json.load(error)


def wait_for_liveness(base: str, timeout: float = 45) -> None:
    """Allow bounded connection failures while Docker forwards to a starting server."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            if get(base, "/health")[0] == 200:
                return
        except (URLError, ConnectionError, TimeoutError):
            # RemoteDisconnected is a ConnectionResetError/ConnectionError.
            pass
        if time.monotonic() >= deadline:
            raise RuntimeError(f"Container did not become live within {timeout:g} seconds.")
        time.sleep(0.2)


def check_container(image: str, bundle: Path, expected_ready: int) -> dict:
    container = docker(
        "run",
        "-d",
        "--read-only",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--tmpfs",
        "/tmp:rw,nosuid,noexec,size=64m",
        "--mount",
        f"type=bind,source={bundle},target=/model,readonly",
        "-p",
        "127.0.0.1::8000",
        image,
    )
    try:
        port = int(docker("port", container, "8000/tcp").splitlines()[0].rsplit(":", 1)[1])
        base = f"http://127.0.0.1:{port}"
        wait_for_liveness(base)
        ready, _ = get(base, "/ready")
        if ready != expected_ready:
            raise AssertionError(f"Expected readiness {expected_ready}, received {ready}.")
        result = {"health": 200, "ready": ready}
        request = Request(
            base + "/predict",
            data=json.dumps(example_customer()).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        if expected_ready == 200:
            status, info = get(base, "/model-info")
            assert status == 200
            with urlopen(request, timeout=5) as response:
                prediction = json.load(response)
                assert response.status == 200
            direct = (
                load_model(ModelSettings(bundle, bundle / "metadata.json"))
                .predict(Customer.model_validate(example_customer()))
                .model_dump()
            )
            for key in ["prediction", "threshold", "model_version"]:
                assert prediction[key] == direct[key]
            assert np.isclose(
                prediction["churn_probability"], direct["churn_probability"], rtol=0, atol=1e-12
            )
            assert info["model_version"] == direct["model_version"]
            result.update(
                model_info=status,
                predict=200,
                direct_parity=True,
                model_version=info["model_version"],
            )
        else:
            assert get(base, "/model-info")[0] == 503
            try:
                urlopen(request, timeout=5)
                raise AssertionError("Unready model accepted prediction.")
            except HTTPError as error:
                assert error.code == 503
            result.update(model_info=503, predict=503)
        return result
    except Exception:
        # Preserve diagnostics locally before removing only our own container.
        diagnostics = Path(__file__).resolve().parents[1] / ".venv/docker-diagnostics"
        diagnostics.mkdir(parents=True, exist_ok=True)
        for label, arguments in {
            "state": (
                "inspect",
                "--format",
                "{{.State.Status}} exit={{.State.ExitCode}}",
                container,
            ),
            "logs": ("logs", "--tail", "40", container),
        }.items():
            captured = subprocess.run(
                ["docker", *arguments], capture_output=True, text=True, timeout=10
            )
            (diagnostics / f"{container[:12]}-{label}.txt").write_text(
                captured.stdout + captured.stderr, encoding="utf-8"
            )
        print(f"Container failed; local diagnostics saved under {diagnostics}")
        raise
    finally:
        docker("rm", "-f", container)


def audit_image(image: str) -> dict:
    code = """
import importlib.util, json, os
from pathlib import Path
assert os.geteuid() == 10001
assert not list(Path('/model').glob('*'))
assert not any((Path('/app') / p).exists() for p in
               ['dataset', 'data', '.git', '.env', '.venv', 'reports', 'notebooks', 'tests'])
for name in ['mlflow', 'pytest', 'ruff', 'nbformat', 'nbclient', 'matplotlib',
             'churn.train', 'churn.validation_experiment', 'churn.tracking']:
    assert importlib.util.find_spec(name) is None, name
print(json.dumps({'non_root_uid': os.geteuid(), 'runtime_only': True, 'no_baked_model': True}))
"""
    return json.loads(
        docker("run", "--rm", "--read-only", "--entrypoint", "python", image, "-c", code)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="churn-api:local")
    parser.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args()
    docker("info", "--format", "{{.OSType}}")  # Fail promptly if the daemon is unavailable.
    root = Path(__file__).resolve().parents[1]
    bundle = args.bundle.resolve()
    load_model(ModelSettings(bundle, bundle / "metadata.json"))
    working = root / ".venv" / ("docker-smoke-" + uuid4().hex)
    missing, corrupt = working / "missing", working / "corrupt"
    for directory in [missing, corrupt]:
        directory.mkdir(parents=True)
        directory.chmod(0o755)
        shutil.copyfile(bundle / "metadata.json", directory / "metadata.json")
        (directory / "metadata.json").chmod(0o644)
    (corrupt / "pipeline.joblib").write_bytes(b"intentional checksum mismatch")
    (corrupt / "pipeline.joblib").chmod(0o644)
    result = {
        "valid": check_container(args.image, bundle, 200),
        "missing": check_container(args.image, missing, 503),
        "corrupt": check_container(args.image, corrupt, 503),
        "image_audit": audit_image(args.image),
    }
    output = root / "reports/milestone5/docker_smoke.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("Docker smoke passed:", json.dumps(result))


if __name__ == "__main__":
    main()
