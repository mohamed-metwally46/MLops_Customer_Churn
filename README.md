# Customer Churn MLOps

A production-oriented customer churn prediction project using Python, scikit-learn,
FastAPI, Pydantic, MLflow, pytest, Docker, and GitHub Actions.

Scope and acceptance criteria are defined in
[MLops_Customer_Churn_Project_Plan.md](MLops_Customer_Churn_Project_Plan.md).
The supplied plan is named differently from `PROJECT_PLAN.md`; it remains the
authoritative plan without a duplicate copy.

## Current status

Milestone 0 is verified. The authorized exploration extends through dataset
validation, EDA, leakage-safe preprocessing, and model/feature comparison notebooks.
Milestone 3 adds local MLflow tracking and validation-only calibration/threshold
analysis. Milestone 4 serves the pinned bundle through FastAPI. Deployment remains
a future milestone. Milestone 5 adds Docker/CI configuration;
local container verification now passes; hosted GitHub Actions execution remains pending.
This is an experimental baseline, not a production release or acceptance gate.

## Environment setup

Use Python 3.11 (inspection found Python 3.11.9). From the repository root:

```powershell
& "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe" -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,training,analysis]"
.\.venv\Scripts\python.exe -m pytest --basetemp=.venv/pytest-local --cov=churn --cov-report=term-missing
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
```

On the inspected Windows host, `python` resolves to an unusable Windows Store
alias. The explicit interpreter above is Python 3.11.9. To execute the notebooks:

```powershell
.\.venv\Scripts\python.exe scripts/run_notebooks.py
```

Dependency installation needs package-index access or a pre-populated local
cache. Tests themselves require no network, Kaggle credentials, or MLflow server.
Runtime dependencies support inference; the `training` extra adds MLflow, and
the `dev` extra adds pytest, coverage, Ruff, and an HTTP test client. The `analysis`
extra adds Matplotlib, notebook validation/execution, and a Python kernel. Dependency
ranges are bootstrap constraints, not a reproducible release lock. Resolve and
lock exact versions before building the release image in the Docker milestone.
Coverage will be measured as behavior is implemented; no arbitrary gate is set.

### Milestone 0 verification on the inspected host

Python 3.11.9, the two setup tests, Ruff lint/format checks, and pip check passed
after dependencies were installed. Earlier sandbox socket errors were resolved
for installation with network-enabled execution. The notebook-tool installation
initially failed with `invalid command 'bdist_wheel'` when using the old setuptools
without build isolation; installing wheel resolved it. Editable installation was
then verified with `pip install --no-build-isolation --no-deps -e .`.
Git has no commits/tracked files and still warns about access to its global ignore.

## Structure and intended architecture

- `src/churn/`: installed package, keeping imports independent of the working directory.
- `tests/`: offline tests added with each milestone.
- `notebooks/`: output-free EDA and training sources; executed copies are ignored.
- `scripts/run_notebooks.py`: validates and executes both notebooks with this environment.
- `data/raw/`: immutable local source data; ignored except the directory marker.
- `data/processed/`: generated local data; ignored except the directory marker.
- `reports/`: small, reviewed, non-sensitive reports.
- `pyproject.toml`: packaging, dependency groups, pytest, and lint configuration.

Planned flow: validated data → train/validation/test split → training-only fitted
preprocessing and model → MLflow run and explicit artifact → startup-loaded
FastAPI service. The test set stays isolated through model and threshold selection.

## Data and secrets

The repository initially contained `dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv`.
It is read in place and never moved, modified, or copied into tracked artifacts.
Both `dataset/` and data directories are excluded from Git and Docker contexts.
The supplied file contains 7,043 rows, 21 columns, and target Churn (No=5,174;
Yes=1,869). Eleven blank TotalCharges values are parsed to missing; all coincide
with zero tenure. No rows are dropped. See [EDA report](reports/eda_summary.md).

If absent, manually download from the
[Kaggle source](https://www.kaggle.com/datasets/blastchar/telco-customer-churn),
following its current account/access requirements. The local file's original
download and license provenance cannot be established from the CSV alone;
confirm applicable usage terms before redistribution or deployment.
Use the existing default location or set a project-relative/absolute override:

```powershell
$env:CHURN_DATA_PATH = 'data/raw/your-telco-file.csv'
```

Loading validates the actual schema and fails on unexpected columns/categories,
invalid numeric values, duplicate identifiers, or invalid targets. Error messages
identify checks/columns without including customer records.

## Historical baseline protocol

Run [EDA notebook](notebooks/01_eda_and_feature_analysis.ipynb) followed by
[training notebook](notebooks/02_model_training_and_selection.ipynb). The runner
creates a workspace-local kernel and saves executed copies in `notebooks/executed/`.
For an interactive editor, select `.venv/Scripts/python.exe` as the notebook kernel.
No Jupyter web UI is required by the runner. Local kernel sockets may require
execution outside a restricted sandbox.

Seed 42 and approximately 70/15/15 stratified splits isolate the test set.
Actual sizes are recorded; scikit-learn rounds fractional splits up. All target
relationships in EDA use training data only. Five-fold training CV compares Dummy,
Logistic Regression, Random Forest, and HistGradientBoosting using average precision
for selection. All preprocessing and supervised feature selection are fold-fitted.
Validation permutation importance and training-fold stability assess feature usefulness.
The fixed threshold is 0.5; it is never optimized on test data. Refit on train plus
validation and predict the test partition once per frozen experiment.

Exclude customerID as an identifier. TotalCharges has high cardinality because it
is numeric, which is not a reason to discard it. Remaining source predictors are
provisional snapshot features; prediction-time availability is not established.
The pipeline compares all encoded features with a fold-fitted L1 selector and uses
selection only if paired CV AP improvement exceeds one paired standard error.
Correlations alone do not select features, and predictive signal is not causation.

Outputs include aggregate Markdown/JSON reports and labeled figures under `reports/`,
plus an ignored, uniquely named model bundle and metadata under `models/`.
The bundle includes numeric parsing, imputation, encoding, scaling where needed,
optional feature selection, and the classifier. Load only trusted joblib files;
supply the 19 input columns listed in `reports/model_results.json` and call
`predict_proba` without manual preprocessing. The current notebook runner creates
separate milestone reports/artifacts and preserves baseline reports; do not use
repeated test observations to retune.
Code/data/artifact hashes, split hashes, dependencies, and metrics are recorded.
The baseline was not MLflow-tracked; the Milestone 3 workflow below is tracked.

See [model comparison](reports/model_comparison.md) for measured CV/test metrics,
feature ranking, confusion matrix, classification report, aggregate errors, and limits.
No threshold costs, fairness approval, temporal validation, or production gate have
been established. Exact dependency locking remains outstanding.

## Historical measured results and verification

Both notebooks executed top to bottom with Python 3.11.9. Actual partition sizes
are 4,929 training, 1,057 validation, and 1,057 test accounts. Mean CV average
precision: Dummy 0.2654, Logistic Regression 0.6646, Random Forest 0.6653,
HistGradientBoosting 0.6625. Random Forest wins the declared maximum-mean rule,
but the difference from Logistic Regression is much smaller than fold variation.
Logistic Regression has better CV recall/F1 at the fixed threshold; the selected
forest is not proven universally superior.

The L1 selector reduced AP by 0.0823, so all 19 source predictors (46 encoded
features) were retained. Contract, InternetService, and tenure rank in the top
five on every CV validation fold and have the largest validation permutation
importance. This evidence supports usefulness, not causal effects.

Final test: precision 0.6500, recall 0.5107, F1 0.5720, ROC-AUC 0.8490, AP 0.6598.
Confusion matrix: `[[700, 77], [137, 143]]` (actual rows No/Yes, predicted columns
No/Yes). At 0.5, 137 of 280 churners are missed. No test-based threshold changes
were made. Full results and aggregate errors are in the model report.

The offline suite passes 20 tests with 55% overall unit coverage (training
orchestration 96%, preprocessing 100%; plotting/report code is exercised by
executed notebooks, not unit coverage). Ruff lint/format and pip check pass.
See [verification record](reports/verification.md) for exact commands and caveats.
scikit-learn 1.9.1 emits deprecation warnings for the backward-compatible L1
`penalty` argument. The declared upper bound excludes 1.10, where removal is
announced; migrate that parameter and revalidate before widening the bound.

## Milestone 3: tracked validation and threshold analysis

Both notebooks now execute the current workflow. EDA writes to
`reports/milestone3/eda/`; training writes append-only run directories under
`reports/milestone3/<parent_run_id>/`. Historical `reports/model_results.json`,
`model_comparison.md`, `verification.md`, other baseline reports and model binaries
remain byte-identical. Original executed notebooks are archived under ignored
`notebooks/executed/history/baseline/`.

```powershell
.\.venv\Scripts\python.exe scripts/run_notebooks.py
# Alternatively, run just the tracked experiment from the project root:
.\.venv\Scripts\python.exe -m churn.validation_experiment
```

Reuse prior training-CV-selected hyperparameters, then repeat five-fold training
CV for all four pipelines. No new hyperparameter or feature search is performed.
Choose maximum mean CV AP; compare training-only OOF sigmoid calibration with raw
probabilities on validation, choosing minimum validation Brier (ties retain raw).
Calibration CV clones the entire preprocessing/model pipeline inside each fold.
Validation threshold tables report precision, recall, F1, TN/FP/FN/TP and percentage
flagged on a 0–1 grid in steps of 0.05. No test-based threshold optimization.

The default threshold remains 0.5. Cost scenarios are configurable relative units,
not actual company costs. To explicitly choose a validation-grid scenario cutoff:

```powershell
# Illustrative assumptions only; replace with approved costs when available.
$env:CHURN_FN_COST = '5'
$env:CHURN_FP_COST = '1'
$env:CHURN_THRESHOLD_POLICY = 'scenario'
.\.venv\Scripts\python.exe -m churn.validation_experiment
# Return to the documented default scenario and fixed threshold:
Remove-Item Env:CHURN_FN_COST, Env:CHURN_FP_COST, Env:CHURN_THRESHOLD_POLICY
```

Without overrides, costs are (FN=1, FP=1), policy=fixed. The scenario minimum
minimizes `FN_cost*FN + FP_cost*FP` over validation-grid thresholds, with ties
preferring fewer flagged, then higher thresholds. This is not a proven business
optimum and ignores intervention efficacy/capacity. The frozen decision is logged
before test access. Refit the selected estimator/calibrator on train+validation,
then evaluate test once, logging test metrics only in the final selected child run.
This holdout has already been evaluated historically, so it is not fresh independent
evidence. Validation reuse and threshold transfer after refitting are limitations.

Measured current results: same CV AP values and raw Random Forest selection as the
baseline. Validation Brier raw=0.139368 vs sigmoid=0.139375; log loss raw=0.431260
vs sigmoid=0.434286. Uniform-bin ECE raw=0.022820 vs sigmoid=0.020354. Raw wins
the predeclared Brier rule, but its tiny advantage is not proof of better calibration.
ECE depends on bins; proper scores also depend on discrimination.

Validation threshold 0.30 gives precision 0.5249, recall 0.7509, F1 0.6179 and
38.03% flagged; 0.50 gives precision 0.6850, recall 0.4875, F1 0.5696 and 18.92%
flagged. Illustrative equal-cost minimum is 0.45 (205 cost units vs 207 at 0.50),
but the actual frozen default remains 0.50. Final test metrics exactly reproduce
the historical baseline; no other thresholds were evaluated on test.
See [Milestone 3 results](reports/milestone3/README.md) and
[verification](reports/milestone3/verification.md).

## Local MLflow usage and immutable model retrieval

MLflow 3.17.1 is installed through the training extra. No hosted service is required:
default tracking URI is an absolute `sqlite:///.../mlflow.db` at the project root;
artifact storage is `mlartifacts/`. Both, including SQLite sidecar files, are ignored
by Git and Docker. Override with `MLFLOW_TRACKING_URI` and `MLFLOW_EXPERIMENT_NAME`
(default experiment: `churn-validation`). Do not put credentials in reports.
The default artifact location is local; hosted/shared artifact storage needs separate
configuration and durable storage and has not been validated here.

Start the optional local UI from this project root:

```powershell
.\.venv\Scripts\mlflow.exe ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Open http://127.0.0.1:5000 to compare candidate child runs. The parent logs data/code/
split metadata; candidates log model parameters, CV/fold metrics and validation
metrics; final-selected logs calibration/threshold artifacts and its test evaluation.
The UI command is documented, not claimed to have been started or smoke-tested.

Verified artifact identity:
`runs:/512d717fe25e46aea32ed10744c49541/model/pipeline.joblib`.
Use an explicit run ID/path, never latest. To retrieve it with the local store:

```python
from pathlib import Path
import joblib
from mlflow.tracking import MlflowClient

root = Path.cwd()
client = MlflowClient(tracking_uri=f"sqlite:///{(root / 'mlflow.db').as_posix()}")
destination = root / "models" / "retrieved-512d717fe25e46aea32ed10744c49541"
destination.mkdir(parents=True, exist_ok=True)
path = client.download_artifacts(
    "512d717fe25e46aea32ed10744c49541", "model/pipeline.joblib", str(destination)
)
model = joblib.load(path)  # trusted local artifacts only
```

Retrieval fingerprint and prediction parity were verified by the workflow using
validation examples. Apply the frozen threshold in metadata to `predict_proba[:,1]`;
the estimator's `predict()` does not implement arbitrary scenario thresholds.
No model registry, production gate or automatic promotion is introduced.

Never commit credentials, `.env` files, raw customer data, MLflow stores, or large
model artifacts. Review generated reports before adding them to Git. A later
Docker milestone must explicitly package a selected, immutable model artifact.

## Milestone 4: inference API

The API uses the existing SHA-verified Random Forest bundle for run
`512d717fe25e46aea32ed10744c49541`. It never trains, recalibrates, tunes thresholds,
downloads models, or contacts MLflow during startup or requests. Runtime dependencies
were already declared; no new dependencies were added for this milestone.

From the project root, install and start locally:

```powershell
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m uvicorn churn.api:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000/docs for interactive OpenAPI documentation.

| Endpoint | Behavior |
| --- | --- |
| `GET /health` | Lightweight liveness, HTTP 200; does not load or invoke the model. |
| `GET /ready` | HTTP 200 after successful startup validation; otherwise 503. |
| `GET /model-info` | Model name/version, SHA-256, threshold, positive class and input schema; no filesystem paths. Unready returns 503. |
| `POST /predict` | One validated record; label, positive-class probability, applied threshold and model version. |

Load once per application process, after SHA verification and feature/class checks,
and run a synthetic usability probe. Startup failure intentionally leaves the process
live but unready: inference/info return 503. Invalid inference output returns a safe
500 and clears readiness until restart. There is no automatic reload or retry.
Do not use liveness as evidence of model quality. Multiple workers each load their
own model instance; host/resource sizing is not configured in this milestone.

### Stable request contract

All 19 predictor keys are required. Numeric features are nonnegative finite JSON
numbers; tenure must be an integer. Values must fit the pipeline's int64 tenure
parsing and the forest's float32 charge representation; these are technical limits,
not invented business limits. SeniorCitizen must be integer 0 or 1. Categories
come from the existing training definition and are listed in `/model-info` and
OpenAPI. Unknown categories are rejected with 422 even though the underlying encoder
can handle them. All predictors accept explicit `null`, which reaches the fitted
median/mode imputers. Blank TotalCharges strings also map to missing; numeric strings
are rejected. Omitted fields, Boolean numbers, extra fields, Churn and customerID
are rejected. Error responses name the field/error type without echoing submitted
values, parsing context or payloads. Customer payloads are not logged.

Example synthetic request (real feature names/categories; not a supplied customer):

```json
{
  "tenure": 12,
  "MonthlyCharges": 65.0,
  "TotalCharges": 780.0,
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "No",
  "Dependents": "No",
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "DSL",
  "OnlineSecurity": "Yes",
  "OnlineBackup": "No",
  "DeviceProtection": "No",
  "TechSupport": "Yes",
  "StreamingTV": "No",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "No",
  "PaymentMethod": "Bank transfer (automatic)"
}
```

The same request is in [examples/customer.json](examples/customer.json):

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/predict -Method Post -ContentType 'application/json' -Body (Get-Content examples/customer.json -Raw)
```

Actual response measured in the local HTTP smoke test:

```json
{
  "prediction": "No",
  "churn_probability": 0.21887081002833744,
  "threshold": 0.5,
  "model_version": "512d717fe25e46aea32ed10744c49541"
}
```

The probability column is selected by inspecting fitted classes (No=0, Yes=1),
not assuming column position. Prediction is Yes when probability ≥ threshold.
All preprocessing is the saved pipeline; no manual encoding/scaling is repeated.

### Artifact and threshold configuration

| Environment variable | Default/meaning |
| --- | --- |
| `CHURN_PROJECT_ROOT` | Current working directory; set when starting outside the repository root. |
| `CHURN_MODEL_METADATA` | `reports/milestone3/a644309446d34b7ca25ea160124aeba6/results.json`, relative to root or absolute. |
| `CHURN_MODEL_PATH` | Optional local bundle override; otherwise the metadata's pinned artifact path. Its SHA must still match metadata. |
| `CHURN_THRESHOLD` | Optional finite 0–1 override; otherwise the metadata's frozen threshold, currently 0.50. No tuning occurs. |

Only point configuration at trusted metadata and joblib artifacts under operator
control. SHA-256 detects mismatched bytes but does not authenticate an untrusted
manifest. Hash the exact bytes before deserializing. The API does not accept model
uploads or artifact paths in requests and never resolves latest. The loader currently
supports the inspected complete uncalibrated sklearn pipeline; a different bundle
format needs explicit loader/contract validation before use.

### Milestone 4 verification and limits

```powershell
.\.venv\Scripts\python.exe scripts/run_notebooks.py --verification
.\.venv\Scripts\python.exe scripts/smoke_api.py
.\.venv\Scripts\python.exe -m pytest --basetemp=.venv/pytest-m4-complete --cov=churn --cov-report=term-missing
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
.\.venv\Scripts\python.exe -m pip check
```

Notebook verification executes both notebooks, recomputes descriptive EDA in
`reports/milestone4/eda/`, verifies the pinned artifact fingerprint and replays saved
experiment metrics without training, calibration, threshold selection or test-set
evaluation. Outputs go to ignored `notebooks/executed/milestone4/`; historical
executed notebooks/reports/models remain unchanged. The smoke script starts and
stops a bounded localhost process; it is not deployment or external hosting.
The optional real-artifact API test skips when the local ignored model is absent;
routine API tests use deterministic synthetic saved models and no dataset/server.

See [Milestone 4 verification](reports/milestone4/verification.md) for measured
results, preservation evidence and commands. Final verification passed 85 tests
with 81% overall coverage (API and schemas 100%, loader 90%); both notebooks,
local HTTP parity, Ruff checks and pip check passed. The RF/LR CV near-tie, limited recall,
temporal validity, fairness, dependency locking, business costs, production release
gates, durable tracking, resource limits and deployment remain unresolved.
No batch endpoint, hosting or new model-selection work was added in Milestone 4.

## Milestone 5: Docker and CI

The Dockerfile uses a two-stage Python 3.11.17 slim-bookworm build, an installed
inference-only package, runtime-only dependencies, UID/GID 10001, `/app` working
directory and Uvicorn on `0.0.0.0:8000`. No system packages are added. The image's
Docker health check calls `/ready`; `/health` remains independent liveness.
The default-deny Docker context includes only 12 explicit build files. No dataset,
secrets, Git metadata, model binaries, notebooks, reports or application training
modules are copied into the image. Third-party sklearn remains required at runtime.

**Current status: local container verification passed.** After Docker Desktop became
available and the image built, the smoke poll exposed an uncaught startup connection
closure. The poll now retries transient connection errors within 45 seconds and
saves local failure diagnostics before cleanup. Valid-model HTTP inference matches
local inference; missing/corrupt models return 503. The image runs as UID 10001
with runtime dependencies only and no baked model. See the
[Docker follow-up](reports/milestone5/docker_followup.md) for commands and results.

### Exact artifact distribution strategy

The image contains no baked model. Prepare a local ignored directory containing
only the exact trusted pipeline bytes and compact runtime metadata, then mount
that directory read-only at `/model`. This milestone deliberately uses a local
mount because the artifact is not committed or published. It adapts the plan's
initial packaged-model recommendation without disabling SHA checks or introducing
an external artifact service. Image plus mount identity must both be pinned for
any future release; the image by itself is intentionally unready.

```powershell
# Refuses to overwrite an existing bundle; choose another output name for a new copy.
.\.venv\Scripts\python.exe scripts/prepare_model_bundle.py --output models/container-bundle
docker build -t churn-api:local .
$bundle = (Resolve-Path models/container-bundle).Path
docker run --rm --name churn-api -p 127.0.0.1:8000:8000 --read-only --cap-drop=ALL --security-opt=no-new-privileges --tmpfs /tmp:rw,nosuid,noexec,size=64m --mount "type=bind,source=$bundle,target=/model,readonly" churn-api:local
```

The bundle prepared during this milestone has run ID
`512d717fe25e46aea32ed10744c49541` and the unchanged artifact checksum. The
original model is verified before copying and remains byte-identical. The
container sets `CHURN_MODEL_PATH=/model/pipeline.joblib` and
`CHURN_MODEL_METADATA=/model/metadata.json`, so no local Windows paths are assumed
inside Linux. It uses the same loader, class mapping, threshold and startup probe.
No network artifact download, registry lookup or MLflow service is needed.

From a second terminal after starting the container:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:8000/ready
Invoke-RestMethod http://127.0.0.1:8000/model-info
Invoke-RestMethod -Uri http://127.0.0.1:8000/predict -Method Post -ContentType 'application/json' -Body (Get-Content examples/customer.json -Raw)
```

To automate HTTP/direct-inference parity, missing/corrupt artifact checks, UID and
image-content checks, stop the manual container and run:

```powershell
.\.venv\Scripts\python.exe scripts/smoke_docker.py --image churn-api:local --bundle models/container-bundle
```

The smoke script creates three bounded containers on loopback ephemeral ports and
removes its containers afterward. Missing/corrupt bundles must return health 200
and readiness/info/predict 503. Corrupt tests use separate temporary copies; the
trusted bundle is untouched. The image audit verifies non-root UID, absence of
baked models and dev/training/notebook packages and project files. These container
checks are implemented but have NOT run successfully on the current host.

### Runtime dependency lock

`requirements/runtime.lock` is a generated, hashed universal Python 3.11 runtime
resolution of pyproject dependencies. `requirements/model-compatibility.in` pins
the verified model/array/API versions; `requirements/build.txt` pins separate
wheel-building tools. Docker installs the runtime lock with `--require-hashes`
and `--only-binary=:all:`, then installs the project wheel with `--no-deps` and
runs pip check. Development/analysis/tracking extras are not installed in the image.

Regenerate deliberately using the uv version recorded in the verification report:

```powershell
uv pip compile pyproject.toml --universal --python-version 3.11 --generate-hashes --no-annotate --constraints requirements/model-compatibility.in --output-file requirements/runtime.lock
```

A fresh Python 3.11.9 Windows environment installed all 26 hashed runtime packages,
passed pip check and reproduced pinned-model inference without MLflow, pytest,
Ruff, Matplotlib or notebook packages. This verifies the Windows runtime dependency
closure; subsequent container checks also verified Linux inference parity.
CI/development extras remain range-constrained, builder
tool pins are not hashed, and the Python base tag is not digest-pinned. Fully
reproducible/bit-identical images remain unverified.

### GitHub Actions CI and local checks

`.github/workflows/ci.yml` runs on pull requests and pushes to `main` using Python
3.11 on Ubuntu. It installs `.[dev,training,analysis]` constrained to the verified
model/API versions; runs offline pytest with coverage, Ruff lint/format, pip check,
and workflow configuration checks; builds the image; and smoke-tests a synthetic
CI-only model mount. No dataset, real release artifact, Kaggle token, tracking
server or external credentials are required. The real-artifact integration test is
excluded in ordinary CI; its local validation remains available separately.

The synthetic fixture is generated with a fixed seed and small model solely to
test serialization/serving. It is labeled `synthetic_ci_only`, has a distinct
artifact identity, and is never a production model or performance result.
Real-artifact Linux parity passed locally; release gates remain unresolved.
Workflow permissions are `contents: read`; checkout does not persist credentials.
Actions are pinned to the verified immutable commits for
[checkout v4.2.2](https://github.com/actions/checkout/commit/11bd71901bbe5b1630ceea73d27597364c9af683)
and [setup-python v5.6.0](https://github.com/actions/setup-python/commit/a26af69be951a213d495a4c3e4e4022e16d87065).
The base tag was checked against the
[official Python image manifest](https://github.com/docker-library/official-images/blob/master/library/python).
No publishing, registry login, release workflow or deployment exists. The hosted
workflow has not been triggered; offline YAML/config checks do not prove a GitHub run.

```powershell
$env:CHURN_VERIFICATION_LABEL = 'milestone5'
.\.venv\Scripts\python.exe scripts/run_notebooks.py --verification
Remove-Item Env:CHURN_VERIFICATION_LABEL
.\.venv\Scripts\python.exe -m pytest --basetemp=.venv/pytest-m5-final --cov=churn --cov-report=term-missing
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe scripts/validate_ci.py
```

Both notebook replays and 89 tests passed with 81% coverage; Ruff, pip check and
offline workflow checks passed. PyYAML and the Docker SDK are declared development
dependencies for YAML and build-context audits; neither is a runtime dependency.
Actionlint is unavailable. See [Milestone 5 verification](reports/milestone5/verification.md)
for the original commands, clean-install evidence and initially blocked Docker steps;
the [Docker follow-up](reports/milestone5/docker_followup.md) records successful checks.
Model quality, temporal validity, fairness, actual business costs, release gates,
complete dependency locking and deployment remain unresolved.

## Remaining milestones

Next, after review: hosted CI verification; configured deployment; final
documentation. Commands and API contracts will be documented when implemented.
Deployment requires a real host, credentials, and authorization. V1 excludes
Kubernetes, DVC, a frontend, database, authentication, and online training.
