# Hosted CI failure fix — 2026-10-15

Run 38751115452 failed in Offline tests and coverage. Installation, lint and
format passed; subsequent dependency/workflow/Docker steps were skipped. The
supplied log reports 41 passed, 52 setup errors and 1 deselected on Linux Python
3.11.17. Every supplied setup error is FileNotFoundError creating
`.venv/pytest-ci`: its parent `.venv` is ignored and absent on a clean checkout.
setup-python supplies a hosted Python installation, not a project virtualenv.
Local checks missed this because the local .venv already existed.

Changed CI basetemp to `"${{ runner.temp }}/churn-pytest"`. The parent directory
is provided by the GitHub runner; pytest owns only its dedicated child directory.
Added a workflow-validator guard and regression test rejecting the old path.
No application/model/data behavior changes.

Also replaced Node.js 20 action pins with Node.js 24 releases, verified from
official release references and action.yml:

- checkout v6: d23441a48e516b6c34aea4fa41551a30e30af803
- setup-python v6: ece7cb06caefa5fff74198d8649806c4678c61a1

## Local commands and actual results

```powershell
$env:RUNNER_TEMP = (Resolve-Path .venv).Path
.\.venv\Scripts\python.exe -m pytest -m 'not integration' --basetemp="$env:RUNNER_TEMP/churn-pytest" --cov=churn --cov-report=term-missing
.\.venv\Scripts\python.exe -m pytest tests/test_container_setup.py -q --basetemp=.venv/pytest-ci-path-regression
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe scripts/validate_ci.py
git diff --check
```

Windows Python 3.11.9: offline suite 93 passed, 1 deselected, 6 existing warnings,
81% source coverage in 26.20 seconds. The subsequently added workflow regression
test is covered by the targeted run: 10 passed. Final Ruff lint/format passed
(41 Python files formatted), pip check and workflow validation passed. A formatting
issue introduced while writing the regression test was fixed before final checks.
The local RUNNER_TEMP value simulates an existing temporary parent; this is not
a Linux or hosted GitHub execution result.

Modified: .github/workflows/ci.yml, scripts/validate_ci.py,
tests/test_container_setup.py. Created: this report. Prior reports preserved.
No commit, push, publishing or deployment performed. Hosted CI remains unverified
after this fix until a new pushed commit completes successfully. Re-running the
old commit alone will retain the old workflow and its failure.
