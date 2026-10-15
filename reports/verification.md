# Execution and verification

Both source notebooks executed top to bottom with Python 3.11.9 using:

```powershell
.\.venv\Scripts\python.exe scripts/run_notebooks.py
```

Result: exit 0. EDA executed four code cells; training executed three, with no
error outputs. Executed copies are local under ignored `notebooks/executed/`.
The runner uses the explicit virtual-environment interpreter for its kernel.
Local kernel sockets required network-enabled execution in this sandbox.
Windows ZMQ/Tornado emitted event-loop and local TCP transport warnings; execution
completed. No remote Jupyter server or production service was started.

The notebooks generated aggregate reports/figures and an ignored model bundle.
The real-data experiment evaluated final test probabilities once; no subsequent
test-driven tuning or retraining was performed. A synthetic orchestration test
separately verifies the single-test-prediction boundary and saved bundle.

```powershell
.\.venv\Scripts\python.exe -m pytest --basetemp=.venv/pytest-final-review --cov=churn --cov-report=term-missing
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
.\.venv\Scripts\python.exe -m pip check
```

pytest: 20 passed, four scikit-learn warnings for the legacy L1 penalty parameter.
Overall unit coverage: 55%; data 82%, preprocessing 100%, training 96%,
exploration/report generation 0% (covered by real notebook execution separately).
No coverage gate is claimed. A first attempt had four temporary-directory access
errors; the workspace-local `--basetemp` resolved them. An orchestration assertion
initially assumed exact 70/15/15 integer sizes on 100 synthetic rows; it was
corrected to sklearn's ceiling behavior. No real-data selection was changed.

Final Ruff lint: `All checks passed!`; Ruff formatting: `15 files already formatted`.
Final pip check: `No broken requirements found.` Editable installation was repeated
successfully after declaring direct NumPy/threadpoolctl dependencies and limiting
scikit-learn to versions before the announced legacy penalty removal in 1.10.

Notebook-tool installation initially failed with `invalid command 'bdist_wheel'`
when disabling isolation with the old setuptools and no wheel package. Installing
wheel and the declared notebook tools succeeded, then editable installation
succeeded using `pip install --no-build-isolation --no-deps -e .`.

Dataset inspected in place: 7,043 rows, 21 columns; No=5,174 and Yes=1,869.
SHA-256 before analysis:
`88be4b93fbe0cc83421af1c503794c97c342eca914c1576db7c276e61d61358a`.
Final SHA-256 matched the pre-analysis value. Saved artifact and source-module
hashes matched model_results.json. Both source notebooks validated with nbformat,
with empty outputs and null execution counts. The two executed copies contained
no error outputs. A check against the actual customerID column found no identifiers
in source notebook/text report files.
No records, identifiers, dataset copies, secrets, or generated model binaries are
included in the source notebooks or aggregate reports. The dataset and model
paths are ignored by Git, and the Docker context excludes dataset/data/models
and notebooks. No Git commits were created; all project files remain untracked.
Git warns that its global ignore file cannot be read, but repository-local ignore
rules operate correctly.

Final Git checks:

```powershell
git check-ignore -v dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv
git check-ignore -v notebooks/executed/01_eda_and_feature_analysis.ipynb
git check-ignore -v models/88be4b93fbe0-seed42-RandomForest-b8c487d8/pipeline.joblib
git ls-files -- dataset/
git diff --check
git diff --stat
git status --short --untracked-files=all
```

All three ignore checks matched repository-local rules; ls-files returned no
dataset entries. Both diff commands produced no output because the repository
has no tracked files; they cannot validate untracked additions. Git status was
reviewed and showed only project source, notebooks, configuration, and aggregate
reports, with no dataset/model/executed-notebook entries. `.dockerignore` explicitly
excludes dataset, data, models, and notebooks; no Docker build was attempted.

See model_results.json for dependency versions, split/source/data/artifact hashes,
selected parameters, CV metrics, feature-selection evidence, validation metrics,
and final test metrics. There is no verified lockfile yet. No MLflow integration,
API, release acceptance gate, Docker image, CI/CD, or deployment is claimed.
