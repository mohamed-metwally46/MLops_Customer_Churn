"""Bounded localhost Uvicorn smoke test using a synthetic request and pinned artifact."""

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

from churn.model_loader import ModelSettings, load_model
from churn.schemas import Customer, example_customer


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    log_path = root / ".venv" / "api-smoke.log"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "churn.api:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
            ],
            cwd=root,
            stdout=log,
            stderr=log,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        try:
            base = f"http://127.0.0.1:{port}"
            deadline = time.monotonic() + 30
            while True:
                try:
                    with urlopen(base + "/ready", timeout=1) as response:
                        assert response.status == 200
                    break
                except URLError:
                    if time.monotonic() >= deadline or process.poll() is not None:
                        raise RuntimeError("Local API readiness failed; inspect local smoke log.")
                    time.sleep(0.2)
            result = {}
            for endpoint in ["health", "ready", "model-info", "openapi.json"]:
                with urlopen(base + "/" + endpoint, timeout=5) as response:
                    assert response.status == 200
                    result[endpoint] = response.status
            request = Request(
                base + "/predict",
                data=json.dumps(example_customer()).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(request, timeout=5) as response:
                assert response.status == 200
                prediction = json.load(response)
            direct = (
                load_model(ModelSettings.from_environment())
                .predict(Customer.model_validate(example_customer()))
                .model_dump()
            )
            assert prediction == direct
            result["predict"] = prediction
            result["direct_bundle_parity"] = True
            output = root / "reports/milestone4/api_smoke.json"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(result, indent=2), encoding="utf-8")
            print("Local HTTP smoke passed:", json.dumps(result), flush=True)
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


if __name__ == "__main__":
    main()
