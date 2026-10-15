# Milestone 5 pre-push review — 2026-10-15

Verdict: PARTIALLY VERIFIED. Prior local Docker success remains preserved;
current local CI checks pass. Hosted GitHub Actions has not run.

## Scope and preservation

Read docker_followup.md, verification.md, workflow, smoke helper, container tests,
packaging/ignore configuration, and authoritative plan requirements. No source,
workflow, model behavior, README, historical report or dataset changes were needed
in this review. This report is the only new project file from this review.

All 47 SHA256 entries in reports/milestone5/preservation.json match. Both source
notebooks have zero saved cell outputs. The dataset, model bundle and executed
notebooks remain Git-ignored; the passing Docker-context test excludes private data
and training files. The 84 nonignored files present before this report included no
CSV, joblib, pickle, key, PEM, database or .env candidates. This filename check is
not a comprehensive secret scanner.

Git has an unborn main branch, no tracked files and no staged changes. Complete
working-tree and staged diffs are empty because every project file is untracked.
Therefore an independent claim that only the earlier smoke helper, tests, README
and reports changed cannot be established from Git. Protected historical files
are independently verified by hashes; other files lack a prior Git baseline.
git diff --check passed but cannot audit untracked content. Existing global-ignore
access warnings persist. git add --dry-run could not create .git/index.lock under
the read-only sandbox; no staging occurred.

## Workflow review

The workflow matches documented checks: Python 3.11, declared dev/training/analysis
extras with compatibility constraints, Ruff lint/format, offline pytest with
coverage, pip check, workflow checks, Docker build, synthetic CI-only bundle and
container parity/failure/image audit. No dataset or real model is required.
PR and main-push triggers, immutable action SHAs, read-only contents permission,
disabled persisted checkout credentials, concurrency cancellation and a 20-minute
job timeout are appropriate. No publishing/deployment steps exist.

Notebook replay and real-artifact integration are local evidence, intentionally
outside dataset-free CI. CI extras are not completely locked. Actionlint is absent;
the custom YAML validator cannot prove all GitHub runtime semantics. Actual valid
PR success and deliberately failing-test CI failure acceptance checks remain
unverified until real hosted runs. The user calls this combined Docker/CI work
Milestone 5; the supplied plan separately numbers pytest 5, Docker 6 and CI 7.
This review follows the authorized combined scope without changing the plan.

## Commands and actual results

```powershell
.\.venv\Scripts\python.exe -m pytest -m 'not integration' --basetemp=.venv/pytest-ci-review --cov=churn --cov-report=term-missing
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe scripts/validate_ci.py
git status --short
git diff
git diff --cached
git diff --check
git ls-files
git ls-files --others --exclude-standard
git check-ignore dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv models/container-bundle/pipeline.joblib notebooks/executed
git ls-remote --heads origin
```

Python 3.11.9: 93 passed, 1 deselected, 6 existing warnings, 81% coverage in
31.98 seconds. Ruff passed, 40 Python files formatted; pip check and workflow
checks passed. SHA256 and source-notebook checks ran through the venv Python.
Sandbox remote lookup failed network access; authorized elevated read-only lookup
succeeded with exit 0 and no branch references. Origin is
https://github.com/mohamed-metwally46/MLops_Customer_Churn.git.

Additional attempted check:

```powershell
.\.venv\Scripts\python.exe scripts/smoke_docker.py --image churn-api:local --bundle models/ci-bundle
docker info --format '{{.ServerVersion}}'
```

Both failed before containers were created because Docker Desktop's engine pipe
is currently absent:

```text
open //./pipe/dockerDesktopLinuxEngine: The system cannot find the file specified.
```

This does not invalidate the earlier successful real-model smoke evidence. No
success JSON was overwritten. Start Docker Desktop in Linux-container mode to
run the additional synthetic check. Do not overwrite prior docker_smoke.json when
collecting new synthetic evidence: preserve it first or run the workflow in a
separate checkout. No build, notebook execution or real-data training was repeated.

## Proposed Git steps, after review only

Run in a normal PowerShell terminal with repository write permission. Because
there is no baseline commit, stage the reviewed project, including its existing
modules and evidence; staging just the follow-up files would create an incomplete
repository. Do not force-add ignored files.

```powershell
git ls-remote --heads origin
git add -- .dockerignore .github .gitignore Dockerfile MLops_Customer_Churn_Project_Plan.md README.md data examples notebooks pyproject.toml reports requirements scripts src tests
git diff --cached --stat
git diff --cached --check
git diff --cached
git status --short
```

Review the complete staged initial-project diff and ensure no dataset, secrets,
model binaries or executed notebooks are included. If remote branch references
appear, stop and reconcile histories; do not force-push or reset.

Only after approving that staged review:

```powershell
git commit -m "Initialize churn service with verified Docker and CI checks"
git push -u origin main
```

This ordinary push triggers main CI. Inspect the actual Actions run at the remote
repository; record its URL, commit SHA, status and failed steps if any. No commit,
push, history rewrite, deployment or publication was performed in this review.
Temporal validity, fairness, business costs, complete locking/base digest pinning,
production release gates and deployment remain unresolved.
