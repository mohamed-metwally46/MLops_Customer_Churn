# Milestone 5: Docker and GitHub Actions CI

**PARTIALLY VERIFIED**: implementation and feasible host checks passed. Docker
build/container/image checks are blocked by the unavailable daemon; a hosted
GitHub Actions run was not triggered. No image publishing, registry login, external
artifact upload, deployment or model-selection work was performed.

The authoritative plan remains unchanged. This user-requested milestone combines
containerization and basic CI; release publishing/deployment remain out of scope.

## Actual verification commands and outcomes

Existing environment: Python 3.11.9. Commands:

```powershell
$env:CHURN_VERIFICATION_LABEL = 'milestone5'
.\.venv\Scripts\python.exe scripts/run_notebooks.py --verification
.\.venv\Scripts\python.exe -m pytest --basetemp=.venv/pytest-m5-final --cov=churn --cov-report=term-missing
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe scripts/validate_ci.py
```

Both notebook replays executed all cells without errors, with separate milestone5
outputs. No fitted preprocessing, training, calibration or test evaluation was
performed by replay. pytest: **89 passed**, six existing warnings, in 48.42 seconds;
overall source coverage **81%**. Ruff lint/format and pip check passed. Four new
offline packaging checks passed: synthetic bundle, byte-preserving export,
default-deny build context and CI configuration. After declaring Docker SDK/PyYAML
as explicit dev extras and adding the CI configuration step, editable installation,
pip check and those four tests were repeated successfully.

Actionlint is not installed. PyYAML plus project configuration/security checks
passed, including events, main branch, read-only permissions, immutable action
references and absence of publish/login commands. This does not validate every
GitHub expression or prove a hosted workflow passed.

Warnings are retained for working/deprecated Starlette httpx TestClient, legacy
sklearn L1 penalty arguments and MLflow/SQLAlchemy noload. Jupyter emitted existing
Windows event-loop and local-TCP warnings while completing successfully.

## Docker attempts: blocked, not successful

```powershell
docker version
docker --config .venv/docker-config version
docker --config .venv/docker-config build -t churn-api:local .
.\.venv\Scripts\python.exe scripts/smoke_docker.py --image churn-api:local --bundle models/container-bundle
```

Docker CLI 29.6.1 is installed. Default configuration access initially also warned
that `.docker/config.json` was inaccessible. An empty workspace-local CLI config
removed that warning but did not fix the daemon connection. Actual build failed:

```text
ERROR: failed to connect to the docker API at npipe:////./pipe/docker_engine;
check if the path is correct and if the daemon is running:
open //./pipe/docker_engine: The system cannot find the file specified.
```

The smoke script failed immediately at `docker info`; it did not create containers.
Consequently no successful image build, Linux prediction parity, container failure
test, Docker health transition, image size or image-content inspection is claimed.
No docker_smoke.json success report exists. Minimum unblock: start a working Linux
Docker Engine / Docker Desktop in Linux-container mode, then rerun build and smoke.
The script checks valid/missing/corrupt mounts, all required endpoints, direct parity,
UID 10001 and image contents when the daemon is available.

Docker SDK's offline build-context matcher audited exactly these 12 files:
Dockerfile, .dockerignore, README.md, pyproject.toml, requirements/build.txt,
requirements/runtime.lock and six inference modules (__init__, api, data,
model_loader, preprocessing, schemas). This is context-pattern evidence, not
an actual image audit. The default-deny ignore policy also excludes future
unlisted secrets/files inside otherwise allowed directories.

## Artifact handling

```powershell
.\.venv\Scripts\python.exe scripts/prepare_model_bundle.py --output models/container-bundle
.\.venv\Scripts\python.exe scripts/prepare_model_bundle.py --synthetic --output models/ci-bundle
```

Both commands completed and bundles passed the existing loader. Production bundle
is a byte-identical copy of the pinned model for
512d717fe25e46aea32ed10744c49541 plus compact metadata. Synthetic CI fixture has
distinct identity 48461ab068b7d7ed9e8375eb60eacbb2 and an explicit synthetic marker.
Existing bundle directories cannot be overwritten. Binary bundles remain ignored.

Distribution selected: local read-only bind mount `/model`, with container
CHURN_MODEL_PATH=/model/pipeline.joblib and CHURN_MODEL_METADATA=/model/metadata.json.
This preserves checksum verification and avoids Windows-path assumptions. Nothing
is baked into/published with the image; an absent mount intentionally stays unready.
Future image-packaged models or secure pinned external storage require separate
configuration/authorization. Image and artifact identity both need release pinning.

## Generated runtime lock and fresh installation

uv 0.11.28 generated requirements/runtime.lock from actual package-index metadata:

```powershell
uv pip compile pyproject.toml --universal --python-version 3.11 --generate-hashes --no-annotate --constraints requirements/model-compatibility.in --output-file requirements/runtime.lock
.\.venv\Scripts\python.exe -m venv .venv/runtime-check
.\.venv\Scripts\python.exe -m pip wheel --no-build-isolation --no-deps --wheel-dir .venv/project-wheel .
```

Resolution completed with 26 packages. An initial uv pip sync attempt stalled
downloading wheels and was stopped rather than repeatedly retried. Pip's populated
254.9 MB cache supported the bounded fallback (PIP_RETRIES=0, PIP_TIMEOUT=15):

```powershell
.\.venv/runtime-check/Scripts/python.exe -m pip install --require-hashes --only-binary=:all: -r requirements/runtime.lock
.\.venv/runtime-check/Scripts/python.exe -m pip install --no-index --no-deps .venv/project-wheel/churn_mlops-0.1.0-py3-none-any.whl
.\.venv/runtime-check/Scripts/python.exe -m pip check
```

All 26 runtime packages and the project wheel installed successfully. Fresh pip
check passed. Imports were verified to resolve to this new environment; MLflow,
pytest, Ruff, nbformat and Matplotlib were absent. Actual pinned-model inference
reproduced probability 0.21887081002833744, label No, threshold 0.5 on the documented
synthetic example. See runtime_install.json. The lock is therefore not fabricated
or merely a freeze of untested versions.

The clean install validates Windows Python 3.11.9 dependency closure and model
loading, not Linux Python 3.11.17 wheels or bit-reproducible images. Runtime
dependencies are hashed/pinned; CI extras remain range-constrained, build tools
are version-pinned without hashes, and the base image is tag-pinned, not digest-pinned.
Those reproducibility tasks remain open pending actual Linux/image verification.

Base tag 3.11.17-slim-bookworm was checked in the
[official Python image manifest](https://github.com/docker-library/official-images/blob/master/library/python).
Workflow actions reference the verified release commits for
[checkout v4.2.2](https://github.com/actions/checkout/commit/11bd71901bbe5b1630ceea73d27597364c9af683)
and [setup-python v5.6.0](https://github.com/actions/setup-python/commit/a26af69be951a213d495a4c3e4e4022e16d87065).
No claim is made that the actual image was pulled or these actions ran here.

## Preserved files and Git

The dataset, plan, all pre-existing reports/figures, executed notebooks and model
artifacts were fingerprinted before implementation (47 files). New notebook
outputs use milestone5 paths; historical files are not overwritten. The source
notebooks and inference modules are unchanged. New reports contain aggregates
or synthetic fixture results, not raw customer records. Preservation hashes are
recorded separately in preservation.json.

Git diff/status and repository ignore rules were inspected. The repository still
has no tracked files, so empty git diff/check output cannot audit untracked files.
The existing global-ignore permission warning persists. Dataset, model mounts,
SQLite stores, local environments and executed notebooks remain ignored. No Git
commits, remote push, registry upload or model publication occurred.

## Files created or changed

- New Dockerfile and .github/workflows/ci.yml.
- Refined .dockerignore to a default-deny context allowlist.
- New requirements/model-compatibility.in, build.txt and generated runtime.lock.
- New scripts/prepare_model_bundle.py, smoke_docker.py and validate_ci.py.
- New tests/test_container_setup.py; existing tests retained.
- Modified scripts/run_notebooks.py to select separate verification output labels.
- Modified pyproject.toml with explicit Docker SDK/PyYAML dev dependencies; runtime
  dependencies unchanged. README updated with commands, mount strategy and limits.
- New reports/milestone5 verification, preservation and runtime-install evidence,
  plus aggregate EDA report/JSON/four figures; ignored local model mount bundles,
  clean runtime environment and executed notebook copies.

Unresolved: real-artifact Linux/container parity, hosted CI execution, base digest
and complete tool/development locks, temporal validity, fairness, actual business
costs, production gates and deployment. This milestone stops before publishing
or production release.
