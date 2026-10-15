# Milestone 4: inference API verification

Implemented the four required endpoints using the existing model. No training,
recalibration, new model selection, threshold tuning, Docker or hosting was performed.
The optional batch endpoint was omitted to keep the single-record contract focused.

## Artifact verified before implementation

The metadata is the preserved Milestone 3 report for parent
a644309446d34b7ca25ea160124aeba6. The actual local artifact exists at
`models/validation-512d717fe25e46aea32ed10744c49541/pipeline.joblib`; its SHA-256
matched metadata before any implementation. Run identity:
`runs:/512d717fe25e46aea32ed10744c49541/model/pipeline.joblib`.
The fitted classes are [0,1], with target map No=0 and Yes=1, and 19 features
match the existing definition and fitted preprocessing columns. Threshold remains
0.50 unless explicitly overridden through CHURN_THRESHOLD. No artifact regeneration.

## Exact commands and results

All commands used the existing Python 3.11.9 virtual environment:

```powershell
.\.venv\Scripts\python.exe scripts/run_notebooks.py --verification
.\.venv\Scripts\python.exe scripts/smoke_api.py
.\.venv\Scripts\python.exe -m pytest --basetemp=.venv/pytest-m4-complete --cov=churn --cov-report=term-missing
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
.\.venv\Scripts\python.exe -m pip check
```

- Both notebooks executed every code cell without errors in verification replay
  mode. EDA is recomputed separately; saved model results are read and their artifact
  fingerprint/identity checked. No fitting, selection, calibration or test-set
  evaluation occurs in this mode. Executed copies are in ignored
  `notebooks/executed/milestone4/`; historical copies remain unchanged.
- Local Uvicorn process returned 200 for health, readiness, model-info, OpenAPI
  and a synthetic prediction. The response exactly matched direct bundle inference:
  No, probability 0.21887081002833744, threshold 0.5, model version
  512d717fe25e46aea32ed10744c49541. See api_smoke.json. The bounded local process
  was stopped automatically; no service was exposed externally.
- Final pytest: **85 passed**, six warnings, in 83.64 seconds. Overall coverage
  **81%**, API 100%, schemas 100%, loader 90%, replay helper 100%. The optional
  real-artifact parity test ran successfully (no skips). Other API tests use tiny
  deterministic synthetic saved models, without the dataset or live services.
- Ruff lint and formatting passed after final changes; pip check reported no
  broken requirements. No new dependency installation or pyproject changes.

The inference module import was checked in a fresh interpreter: neither mlflow
nor churn.train is imported. Tests audit actual joblib deserialization once per
startup and prohibit Pipeline.fit during startup/requests. Reversed fitted class
ordering is tested to ensure positive probability is selected by class value.
Threshold ties use >=. Invalid or missing fields, extra predictors, unsupported
categories, Boolean numbers, negative/nonfinite/unrepresentable numbers and
malformed requests return sanitized 422 errors. Explicit nulls/blank TotalCharges
use the existing fitted imputers. Checksum mismatch is rejected before deserialization.
Missing/corrupt models fail readiness; invalid prediction output fails closed.

An initial startup smoke exposed a difference between the experiment alias
RandomForest and sklearn's class name RandomForestClassifier. The loader now
validates an explicit alias/class mapping; all subsequent checks passed.
During contract review, numeric representation bounds and OpenAPI's blank
TotalCharges representation were completed; the final suite and HTTP smoke were
run against those changes. Earlier intermediate pass counts are superseded here.

Warnings retained: Starlette's working httpx TestClient path is deprecated, four
legacy L1 penalty warnings in historical tests, and MLflow/SQLAlchemy's noload
deprecation. Jupyter also emitted existing Windows event-loop/local-TCP warnings
while executing successfully. These are not failures or compatibility guarantees
for future unconstrained dependency updates; dependency locking remains unresolved.

## Preservation and Git checks

All 35 protected historical files matched pre-edit SHA-256 fingerprints: supplied
plan, dataset, existing reports/figures, executed notebooks including archived
baseline notebooks, and existing models/metadata/retrieved copies. See preservation.json.
Source notebooks remain output-free. New notebook results, metadata, reports and
examples contain aggregate or explicitly synthetic information, not customer records.

```powershell
git diff --check
git diff --stat
git status --short --untracked-files=all
git check-ignore -v dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv
git check-ignore -v models/validation-512d717fe25e46aea32ed10744c49541/pipeline.joblib
git check-ignore -v notebooks/executed/milestone4/02_model_training_and_selection.ipynb
git ls-files -- dataset/
```

Ignore checks matched repository-local rules and ls-files returned no dataset.
Dataset/model/executed notebook files are also excluded from the Docker context.
No Git commits were made. The repository has no tracked files, so both diff checks
produce no output and cannot audit untracked additions; actual files and status
were inspected separately. The existing global-ignore permission warning remains.

## Files created or modified

- New src/churn/api.py, model_loader.py, schemas.py, notebook_replay.py.
- New tests/test_api.py and tests/test_notebook_replay.py; historical tests unchanged.
- New scripts/smoke_api.py and examples/customer.json.
- Modified scripts/run_notebooks.py and both source notebooks for no-training replay
  and separate output locations; README updated with startup, contract and settings.
- New reports/milestone4/verification.md, preservation.json, inference_contract.json,
  api_smoke.json, plus aggregate eda/data_quality.json, eda/eda_summary.md and four
  EDA figures (target_distribution, numeric_distributions, categorical_churn_rates,
  correlations). Executed notebook copies remain ignored.
- Existing dependencies, historical reports, dataset, plan and model binaries unchanged.

## Limits

This is a verified inference milestone, not a production-quality model or deployment.
The loader supports the inspected uncalibrated full pipeline, not arbitrary bundle
formats. SHA pinning assumes trusted metadata and pickle files; no uploads or remote
model resolution exist. Startup failures deliberately remain live but unready and
require restart/configuration correction. Readiness does not establish model quality.
Temporal validity, fairness, real error costs, dependency locking, release gates,
resource limits, durable hosted tracking and deployment remain unresolved. No
production inference rollout, Docker image or hosting configuration was added.
