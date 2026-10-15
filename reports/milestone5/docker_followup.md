# Docker verification follow-up — 2026-10-15

Local Docker verification: VERIFIED. Hosted GitHub Actions execution remains pending.
The earlier verification report remains unchanged as historical evidence.

The supplied build log shows a successful `churn-api:local` build. Its subsequent
smoke failure was `http.client.RemoteDisconnected` during `/health` startup polling.
The helper caught URLError only; Docker can accept a forwarded connection before
the HTTP server is ready. It now retries ConnectionError (including that exception)
and TimeoutError within the existing 45-second bound. Assertions and malformed
responses still fail. Failed containers retain local state/log diagnostics under
ignored `.venv/docker-diagnostics/` before cleanup. Containers are no longer
automatically removed on exit, allowing diagnostics before explicit removal.

## Commands actually executed

All Python commands used the existing Python 3.11.9 virtual environment.

```powershell
docker info --format '{{.ServerVersion}}'
.\.venv\Scripts\python.exe -m pytest tests/test_container_setup.py -q --basetemp=.venv/pytest-docker-fix
.\.venv\Scripts\python.exe scripts/smoke_docker.py --image churn-api:local --bundle models/container-bundle
.\.venv\Scripts\python.exe -m pytest --basetemp=.venv/pytest-docker-followup --cov=churn --cov-report=term-missing
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe scripts/validate_ci.py
git check-ignore dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv
Get-FileHash dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv -Algorithm SHA256
git diff --check
git status --short
```

- Docker server: 29.6.1. Sandbox access initially failed; authorized elevated local
  access succeeded. No image rebuild was necessary for this host-side helper fix.
- Targeted tests: 9 passed. Full suite: 94 passed, 6 existing warnings, 81% coverage,
  53.80 seconds. New tests cover connection closure/refusal/timeout/URLError and
  bounded failure.
- Smoke: valid health/readiness/model-info/predict all 200; exact categorical
  output/threshold/version parity, probability parity within 1e-12 with local model.
  Model version: `512d717fe25e46aea32ed10744c49541`.
- Missing and checksum-corrupt model cases: health 200; readiness/model-info/predict
  503. Image audit: UID 10001, runtime-only imports, no baked model.
- Ruff lint passed; final formatting check passed (40 files); pip check reported no broken
  requirements; offline workflow checks passed. One formatting violation found
  during editing was corrected before final checks.
- Dataset remains Git-ignored. Docker context allowlist test passed and excludes
  dataset/models/secrets. Dataset SHA256 remains
  `88be4b93fbe0cc83421af1c503794c97c342eca914c1576db7c276e61d61358a`.
- All 47 files in the existing preservation manifest still match their hashes,
  including plan, dataset, previous reports and notebook evidence.
- Git diff check produced no errors. All project files remain untracked, so Git
  diff cannot review these files. Global Git ignore access warning persists.

Files modified: scripts/smoke_docker.py, tests/test_container_setup.py, README.md.
Files created: this report and reports/milestone5/docker_smoke.json (aggregate
machine-readable smoke results). No Dockerfile, dataset, model or plan changes.

The supplied build arguments PIP_DEFAULT_TIMEOUT and PIP_RETRIES are not declared
as Dockerfile ARGs, so they do not configure its pip invocation. The build succeeded
regardless. No change to image configuration was required for this smoke fix.

Notebooks were not rerun for this host-side verification-helper change; their
previous executed evidence is preserved. No real-data training or test-set model
evaluation was repeated. Hosted CI, full dependency locking/base digest pinning,
temporal validity, fairness, actual business costs, release gates and deployment
remain unresolved. No publication or deployment occurred.
