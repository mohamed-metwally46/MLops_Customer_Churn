# Customer Churn MLOps Project Plan

## 1. Project Goal

Build a production-oriented Customer Churn Prediction service that takes a customer's attributes and returns a churn prediction and probability.

The project must demonstrate the end-to-end ML lifecycle:

1. Acquire and validate data.
2. Explore and prepare data without leakage.
3. Train and evaluate baseline models.
4. Track experiments and artifacts with MLflow.
5. Serve a fixed, versioned model through FastAPI.
6. Test the code and API with pytest.
7. Package and run the service with Docker.
8. Automate quality checks, tests, and image builds with GitHub Actions.
9. Deploy to a staging/production environment with smoke tests and a rollback strategy.

**Required technologies:** Python, pandas, scikit-learn, FastAPI, Pydantic, MLflow, pytest, Docker, GitHub Actions.

**Out of scope for the first version:** DVC, Kubernetes, a frontend, a database, authentication/business-user management, online training, and complex model monitoring. Keep the first version small and reliable.

---

## 2. Problem and Dataset

Use the public IBM Telco Customer Churn dataset as the initial dataset:
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

Target: `Churn` (`Yes` / `No`).

Dataset requirements:

- Document the source, license/usage terms, expected schema, and download instructions.
- Do not commit private data, credentials, or `.env` files.
- Keep raw data immutable and separate from processed data.
- Do not assume the dataset's current filename or column types; inspect and validate the downloaded file.
- Exclude `customerID` from model features unless a documented experiment establishes a legitimate reason to use it.
- Check duplicates, missing values, unexpected categories, data types, target distribution, and suspicious/leaky features.
- Handle the common `TotalCharges` parsing issue if it appears in the selected dataset version.
- Do not silently drop records or columns. Document data-cleaning decisions.

Recommended directories:

```text
data/
  raw/          # local only; normally gitignored
  processed/    # generated local data; normally gitignored
reports/        # small, non-sensitive evaluation reports
```

Include a script or documented command for obtaining the dataset. If downloading requires a Kaggle account/API token, do not attempt to bypass authentication; document the manual download path and expected location.

---

## 3. Data Preparation and Leakage Prevention

Create a reproducible data-preparation pipeline.

- Split the data into train, validation, and test sets (a starting point is 70/15/15) using stratification where appropriate.
- Fix and document the random seed.
- Keep the test set isolated until model selection and threshold decisions are complete.
- Fit imputers, scalers, and encoders on training data only.
- Use scikit-learn `Pipeline` and `ColumnTransformer` so preprocessing and prediction stay together.
- Treat numeric and categorical features separately.
- Define behavior for unknown categorical values at inference time.
- Record the split strategy and dataset fingerprint/hash where practical.
- Ensure all inference features are available at the moment a real prediction would be made.

Deliverables:

- Data validation script.
- Reproducible split/preprocessing code.
- Brief EDA and data-quality report.
- Documentation of feature/target definitions and leakage checks.

---

## 4. Training and Model Evaluation

Start with two models:

1. Logistic Regression as the baseline.
2. Random Forest as a comparison model.

Use the same split and fair evaluation protocol for both. Tune only on training/validation data; use the test set for final evaluation after selecting the model.

Track suitable metrics:

- Precision
- Recall
- F1
- ROC-AUC
- PR-AUC / average precision
- Confusion matrix
- Optionally, calibration and threshold analysis

Do not choose a model based on accuracy alone. Discuss the business trade-off between missing a likely churner (false negative) and incorrectly flagging a customer (false positive).

Create a configurable model acceptance gate. Do not invent arbitrary production-quality thresholds. Start with a documented baseline and select acceptance thresholds after observing validation results and discussing the intended use. If a candidate fails the gate, the release pipeline must not promote it.

Deliverables:

- Re-runnable training command.
- Evaluation command/report.
- Saved full preprocessing-plus-model pipeline.
- Model-selection rationale.
- A deterministic test fixture or small synthetic sample for tests (not a replacement for real evaluation).

---

## 5. MLflow Experiment Tracking

Use MLflow to track every training run.

Log at least:

- Parameters: algorithm, hyperparameters, preprocessing configuration, seed, split strategy.
- Metrics: precision, recall, F1, ROC-AUC, PR-AUC / average precision.
- Artifacts: evaluation report, confusion matrix, and serialized model.
- Metadata/tags: code version/commit when available, dataset fingerprint, run purpose, and model candidate.

Requirements:

- Make the tracking URI configurable through environment variables/config.
- The local development path must work without requiring a hosted MLflow service.
- Document how to start the local MLflow UI/server and where data/artifacts are stored.
- Do not commit credentials or rely on ephemeral container storage for production tracking.
- Pin the model version or immutable artifact used by the API; do not resolve an ambiguous `latest` model on every request.
- If using MLflow Model Registry, confirm the chosen MLflow version and backend support the registry workflow. Keep a simpler explicit model-artifact path as a fallback for the initial version.

Example environment variable:
`MLFLOW_TRACKING_URI` (use a sensible local default, documented in README).

---

## 6. FastAPI Inference Service

Build a small service that loads the selected model pipeline once during application startup and reuses it for requests. Training must not run during inference requests.

Suggested endpoints:

- `GET /health`: process/liveness check.
- `GET /ready`: success only after the model has loaded and is usable.
- `GET /model-info`: model name/version and relevant non-sensitive metadata.
- `POST /predict`: validate input, predict churn label and probability.

Use Pydantic request/response schemas. Derive the final feature schema from the selected training pipeline; do not guess the required features or their types. Ensure unknown categories and invalid values have intentional, tested behavior.

Example response shape (illustrative only):

```json
{
  "prediction": "Yes",
  "churn_probability": 0.82,
  "model_version": "1"
}
```

The numeric probability above is only a format example, not a claimed model result.

API requirements:

- Stable, documented request and response contracts.
- Meaningful HTTP status codes and safe error messages.
- No sensitive customer data in logs.
- Do not reload the model on every request.
- Keep `/health` distinct from `/ready`.
- Make model artifact location/version configurable.
- Add a startup failure path if the model cannot be loaded; readiness must not falsely report success.
- Provide OpenAPI docs through FastAPI.

---

## 7. pytest Testing Strategy

All tests must run without internet access, Kaggle credentials, or a live MLflow server.

Include:

### Unit tests
- Data validation and cleaning.
- Preprocessing behavior.
- Model loading/configuration.
- Prediction output shape and valid probability range.

### API tests
- Valid request returns the documented response and HTTP 200.
- Missing fields and invalid types return appropriate validation errors.
- Unknown categories follow the documented behavior.
- Health and readiness endpoints behave correctly.
- Model loading failure does not report readiness.
- Error responses do not leak stack traces or sensitive values.

### Model/contract tests
- Training and inference use the same feature order and preprocessing.
- Serialized pipeline can be loaded and called.
- Prediction/probability shape is valid.
- Probability values are finite and in `[0, 1]`.

Use pytest fixtures and a small deterministic test model or mock where appropriate. Avoid retraining on the full real dataset in every test. Include at least one integration/smoke test for a real trained artifact in a separate optional/local workflow.

Coverage should be measured. Choose a reasonable initial coverage gate after establishing a baseline; do not inflate coverage with meaningless tests.

---

## 8. Docker

Create a Docker image for the FastAPI inference service.

Requirements:

- Pin the Python base image to a specific supported version/tag.
- Install the project and runtime dependencies reproducibly.
- Run as a non-root user.
- Use environment variables for configuration; never bake secrets into the image.
- Add `.dockerignore`.
- Avoid including raw datasets, local virtual environments, `.git`, secrets, or unnecessary training files in the runtime image.
- Prefer a small runtime image. Use a multi-stage build if it materially helps.
- Expose/configure the expected service port.
- Ensure the image starts the API and readiness becomes successful only after model loading.
- Include a documented Docker build/run command.
- Use a versioned/immutable model artifact packaged with the image for the initial deployment unless a secure, version-pinned artifact-fetch design is implemented.

Local acceptance commands should be documented, for example:

```bash
docker build -t churn-api:local .
docker run --rm -p 8000:8000 churn-api:local
```

Then test `/ready` and send a real request to `/predict`.

Do not claim Docker works until the image has actually been built and smoke-tested.

---

## 9. GitHub Actions CI/CD

Create GitHub Actions workflows.

### Pull request / push CI
- Checkout source.
- Set up the documented Python version.
- Install the project and development dependencies.
- Run lint/format/type checks if configured.
- Run pytest.
- Build the Docker image.
- Run a container smoke test if feasible without external services.

### Release/image publishing
- Publish a versioned image to GitHub Container Registry (GHCR) only from an appropriate trusted branch/tag.
- Use least-privilege `permissions`.
- Use GitHub-provided token or configured secrets safely.
- Never expose secrets to untrusted pull-request code.
- Tag images with a release/version or commit SHA; do not rely on `latest` alone.
- Pin third-party Actions to reviewed versions, preferably full commit SHAs for hardened production use.

### Deployment
- Keep deployment separate from basic CI.
- Deploy to staging first.
- Run smoke tests against the deployed service.
- Gate production deployment using GitHub Environments/approval where available.
- Document the rollback procedure and how to redeploy the previous known-good image.

Do not create a fake deployment workflow that claims to deploy without a real target and credentials. If no hosting provider is configured, complete CI and image publishing where possible, and document deployment as a clearly marked manual/configuration-dependent step.

---

## 10. Production Readiness

Before declaring the project production-ready:

- Use structured logs with timestamps, severity, and request correlation where appropriate.
- Do not log full customer payloads or secrets.
- Configure request timeouts and resource limits according to the chosen host.
- Make health/readiness semantics correct.
- Pin model and dependency versions.
- Ensure container and workflow permissions are minimal.
- Keep credentials in the host's secret manager/GitHub Secrets.
- Document expected API input, output, errors, and model version.
- Add a smoke test after deployment.
- Document how to roll back to the prior image/model.
- Be explicit that successful API health checks do not prove prediction quality.
- Document current limitations: no automated drift monitoring or automatic retraining in v1.

No database, frontend, authentication, or online retraining is required for this initial project.

---

## 11. Suggested Repository Structure

Adapt this structure when justified, but keep clear separation of concerns.

```text
churn-mlops/
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── release.yml
├── data/
│   ├── raw/                 # local only / gitignored
│   └── processed/           # generated / gitignored
├── reports/                 # generated reports; avoid large/sensitive files
├── src/
│   └── churn/
│       ├── __init__.py
│       ├── config.py
│       ├── validate_data.py
│       ├── preprocessing.py
│       ├── train.py
│       ├── evaluate.py
│       ├── model_loader.py
│       ├── schemas.py
│       └── api.py
├── tests/
│   ├── conftest.py
│   ├── test_data.py
│   ├── test_training.py
│   ├── test_model.py
│   └── test_api.py
├── Dockerfile
├── compose.yaml
├── .dockerignore
├── .gitignore
├── pyproject.toml
└── README.md
```

Keep MLflow tracking data and large model artifacts out of Git unless intentionally using a suitable artifact-storage mechanism. Do not commit raw dataset files by default.

---

## 12. Implementation Milestones and Acceptance Criteria

### Milestone 0 — Repository inspection and project setup
- Inspect existing files before editing.
- Record Python version, environment, and existing project conventions.
- Create/adjust `pyproject.toml`, `.gitignore`, and README skeleton.
- Acceptance: clean, documented starting structure; no secrets committed.

### Milestone 1 — Dataset and validation
- Document dataset acquisition.
- Implement schema/data quality checks.
- Create an EDA/data-quality report.
- Acceptance: validation works on the downloaded dataset and gives clear actionable errors for invalid input.

### Milestone 2 — Preprocessing and training
- Build reproducible stratified splits and sklearn preprocessing pipeline.
- Train baseline Logistic Regression and Random Forest.
- Evaluate on validation; preserve test data for final evaluation.
- Acceptance: re-running the same configuration yields reproducible results within expected deterministic limits.

### Milestone 3 — MLflow
- Log parameters, metrics, artifacts, and dataset/code metadata.
- Document local tracking setup.
- Acceptance: compare runs in MLflow and retrieve the selected model artifact by an explicit version/path.

### Milestone 4 — FastAPI
- Implement `/health`, `/ready`, `/model-info`, `/predict`.
- Load model once at startup.
- Acceptance: valid request returns a valid prediction; invalid requests return correct errors; readiness reflects model availability.

### Milestone 5 — pytest
- Add unit, API, model contract, and failure-path tests.
- Acceptance: test suite runs offline and passes from a clean environment.

### Milestone 6 — Docker
- Build the inference image and run it locally.
- Test readiness and prediction through the container port.
- Acceptance: documented commands work and smoke tests pass against the running container.

### Milestone 7 — GitHub Actions
- Automate checks, tests, and image build.
- Add secure versioned image publishing if repository permissions/configuration allow.
- Acceptance: a deliberately failing test fails CI; a valid PR passes; image publishing uses least privilege.

### Milestone 8 — Staging/production
- Select and document a hosting provider.
- Configure secrets and runtime settings.
- Deploy to staging, run smoke tests, and document rollback.
- Acceptance: deployed API responds externally and the exact model/image version can be identified.

### Milestone 9 — Final documentation
README must include:
- Problem statement and dataset source.
- Architecture diagram or concise flow.
- Environment setup.
- Dataset download instructions.
- Training/evaluation commands.
- MLflow UI/tracking instructions.
- Test commands.
- Docker commands.
- API request/response examples.
- CI/CD workflow explanation.
- Deployment instructions or explicit prerequisites.
- Known limitations and next steps.

---

## 13. Definition of Done

The project is complete when:

- Data preparation and model training are reproducible and documented.
- Model selection uses suitable metrics and a held-out test set.
- MLflow tracks experiments and artifacts.
- FastAPI serves a specific model version.
- pytest tests the main behavior and failure cases.
- Docker builds and runs the API successfully.
- GitHub Actions runs quality checks, tests, and image builds.
- A deployed instance has passed smoke tests, if hosting is configured.
- README enables another developer to reproduce the workflow from scratch.
- No secrets or private customer data are committed.

A passing `/health` endpoint alone is not sufficient evidence of model quality or production readiness.

---

## 14. Recommended Development Order

Implement incrementally, in this order:

1. Inspect repository and set up packaging/configuration.
2. Dataset acquisition instructions and validation.
3. Data splitting and preprocessing pipeline.
4. Baseline training and evaluation.
5. MLflow tracking.
6. Model artifact/version contract.
7. FastAPI service.
8. pytest tests.
9. Docker image and container smoke test.
10. GitHub Actions CI.
11. GHCR publishing.
12. Staging deployment, smoke test, and rollback documentation.
13. Final README and reproducibility review.

Do not implement all phases in one large change. Complete one milestone, run its checks, summarize the result, and only then proceed.
