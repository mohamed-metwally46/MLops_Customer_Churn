# Milestone 3 verification

Commands used the existing virtual environment, Python 3.11.9:

```powershell
.\.venv\Scripts\python.exe scripts/run_notebooks.py
.\.venv\Scripts\python.exe -m pytest --basetemp=.venv/pytest-m3-full --cov=churn --cov-report=term-missing
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
.\.venv\Scripts\python.exe -m pip check
```

- Both notebooks executed top to bottom, exit 0. EDA: four code cells; training: three.
- pytest: 41 passed, five warnings, in 36.52 seconds. Total coverage 75%; threshold/
  calibration summaries 100%, tracking 100%, validation report 100%, validation
  orchestration 84%. Existing historical tests were retained.
- Ruff lint passed. Ruff formatting passed (22 files at that check; final count may
  include new aggregate reports depending on Ruff discovery).
- pip check: no broken requirements.

After README edits, a final formatting check caught formatting in its new Python
download example. `python -m ruff format README.md` corrected it; the repeated
lint and format checks passed (`All checks passed!`, `24 files already formatted`).

New unit/integration tests run offline with synthetic data and temporary local SQLite
stores: input validation, threshold boundary/tie counts, cost-dependent threshold
selection, reliability bins/proper scores, full-pipeline calibration CV, run failure
status, immutable artifact download and prediction parity. The workflow test spies on
fits and both calibrated/uncalibrated prediction paths: no test indices enter fitting,
one outer test prediction occurs, and only final-selected has test-prefixed metrics.

During development, a new report test failed because pyplot selected an unavailable
Windows Tk installation. Report generation now uses headless Figure objects. A test
spy initially watched only Pipeline.predict_proba; sigmoid-calibrated Logistic Regression
uses decision_function internally. The spy now audits both estimator variants and avoids
counting nested probability calls twice. Both were test-time failures before real execution;
no real-data test-driven model changes were made.

Warnings retained: legacy L1 penalty deprecations in unchanged baseline tests and an
MLflow/SQLAlchemy deprecated noload strategy. Jupyter emitted the existing Windows ZMQ
event-loop/local-TCP warnings but completed. No MLflow UI service was started or tested.
The local SQLite backend and file artifact operations were tested without a server.

Parent run: a644309446d34b7ca25ea160124aeba6.
Final run: 512d717fe25e46aea32ed10744c49541.
Artifact: `runs:/512d717fe25e46aea32ed10744c49541/model/pipeline.joblib`.
Retrieval SHA-256 and validation prediction parity were checked by actual execution.
An additional query verified that the parent and all five children are FINISHED
and that only final-selected has test-prefixed metrics. Current source-module
hashes match the run's provenance. Both source notebooks validate with nbformat
and have empty outputs/null execution counts; both executed copies have complete
execution counts and no error outputs. A scan against actual customerID values
found no IDs in source notebooks or text reports.
Only the selected model has a final test evaluation; raw/sigmoid comparison, reliability
metrics and threshold rows use validation only. Model/CV results reproduce the preserved
baseline. The historical holdout was already evaluated and is not fresh independent data.

Dataset, plan, baseline Markdown/JSON reports and archived historical executed notebooks
were compared against pre-edit SHA-256 fingerprints and matched. The baseline binary's
fingerprint matched its existing metadata before training. The preservation manifest is
baseline_evidence.json. Source notebooks are output-free; executed outputs remain ignored.
Generated model binaries, local MLflow stores and dataset remain excluded by Git/Docker.
No raw customer records are exported. No plan or dataset changes and no Git commits.

The repository still has no tracked files; git diff/check cannot validate untracked
additions. Git status and actual files were inspected separately. Git retains its unrelated
global-ignore access warning; repository-local ignore checks work.

`git check-ignore -v` matched repository rules for the CSV, mlflow.db,
mlflow.db-wal, mlartifacts and archived executed notebooks. `git ls-files -- dataset/`
returned nothing. `git diff --check`, `git diff --stat` returned no output;
`git status --short --untracked-files=all` was reviewed. `.dockerignore` excludes
dataset/data/models/notebooks/MLflow stores and SQLite sidecars. No build was attempted.

Outstanding: actual business error costs, production gates, temporal validity, fairness,
independent future evaluation, probability/threshold stability, dependency locking, durable
hosted tracking configuration and deployment. No FastAPI or deployment work was started.
