"""Aggregate-only milestone report and validation reliability/threshold figures."""

import json
from pathlib import Path

import pandas as pd
from matplotlib.figure import Figure

from churn.exploration import markdown_table


def write_report(result: dict, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=False)
    fig = Figure(figsize=(12, 5))
    axes = fig.subplots(1, 2)
    axes[0].plot([0, 1], [0, 1], "k--", label="Ideal reliability")
    for name, values in result["calibration"].items():
        bins = values["reliability"]
        axes[0].plot(
            [b["mean_probability"] for b in bins],
            [b["observed_fraction"] for b in bins],
            marker="o",
            label=name,
        )
    axes[0].set(
        xlabel="Mean predicted probability",
        ylabel="Observed churn fraction",
        title="Validation reliability; uniform bins",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axes[0].legend()
    table = pd.DataFrame(result["thresholds"])
    for metric in ["precision", "recall", "f1"]:
        axes[1].plot(table["threshold"], table[metric], label=metric)
    axes[1].plot(
        table["threshold"], table["flagged_percent"] / 100, linestyle="--", label="Fraction flagged"
    )
    axes[1].set(
        xlabel="Decision threshold (predict Yes when p ≥ threshold)",
        ylabel="Fraction",
        title="Validation operational trade-offs",
        ylim=(0, 1),
    )
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(directory / "validation_analysis.png", dpi=130)
    calibration_table = pd.DataFrame(
        [
            {"probabilities": name, **{k: values[k] for k in ["brier", "log_loss", "ece_uniform"]}}
            for name, values in result["calibration"].items()
        ]
    )
    cv_table = pd.DataFrame(
        [
            {
                "model": name,
                **{m: f"{v['mean']:.5f} ± {v['std']:.5f}" for m, v in item["cv"].items()},
            }
            for name, item in result["comparison"].items()
        ]
    )
    decision = result["frozen_decision"]
    report = f"""# Milestone 3: model validation, thresholds and MLflow

Parent run: `{result["parent_run_id"]}`. Final selected run: `{result["final_run_id"]}`.
Dataset fingerprint: `{result["provenance"]["dataset_sha256"]}`.
Split sizes: {result["provenance"]["split_sizes"]}; seed 42. Baseline reports and
original artifacts remain unchanged. Historical executed notebooks are archived
under ignored notebooks/executed/history/baseline/.

## Model validation and selection boundary

Reuse four pipelines and the hyperparameters selected by prior training-only CV.
Rerun stratified training CV, with every imputer/encoder/scaler fitted inside folds.
Choose maximum mean CV average precision. Retain all 19 inputs: the previous
training-CV selector was rejected; no new feature search is performed. CV is not
nested; hyperparameters were chosen on these training folds historically.

{markdown_table(cv_table)}

The RF/LR near-tie does not prove forest superiority. No candidate test scores
are computed or logged. The test split was already evaluated in the historical
baseline; the new result is reuse of a holdout, not a new independent test.
Historical test results do not enter any executable decision rule.

## Validation-only calibration

Fit sigmoid CalibratedClassifierCV on training only with ensemble=False. Its OOF
probabilities come from clones of the COMPLETE preprocessing/model pipeline,
so no global fitted preprocessing enters calibration folds. Refit the base estimator
on training after fitting the sigmoid mapping. Compare to the raw model on validation;
select sigmoid only if its Brier score is strictly lower (ties retain raw).

{markdown_table(calibration_table)}

Brier and log loss are proper scores (lower is better); neither isolates calibration
from discrimination. ECE is descriptive, bin-dependent and noisy, not a confidence
bound or proof of reliable individual risk. Counts per reliability bin are in results.json.
No calibrator is fitted on validation or test. Validation is reused for calibration
choice and threshold scenarios, so validation-selected estimates may be optimistic.
No calibration metrics or reliability plots are computed on test.

![Validation reliability and threshold trade-offs](validation_analysis.png)

## Threshold analysis and configurable cost scenario

All rows below use validation probabilities from the chosen calibration variant.
Threshold grid: 0–1 in steps of 0.05; prediction uses p ≥ threshold, including ties.
Zero flagged yields precision=0 by explicit zero-division convention.
Configured FN cost={result["config"]["fn_cost"]}, FP cost={result["config"]["fp_cost"]}.
These are illustrative relative units, NOT company costs or a proven business optimum.
Scenario cost = FN_cost × FN + FP_cost × FP; no retention capacity constraints,
intervention efficacy, or revenue effects are modeled.

{markdown_table(table)}

Validation-grid scenario minimum: threshold **{result["scenario"]["threshold"]:.2f}**,
cost {result["scenario"]["scenario_cost"]:.2f}, flagged
{result["scenario"]["flagged_percent"]:.2f}%. Ties favor fewer flagged customers,
then the higher threshold. This is an empirical grid estimate; it is not the
analytic Bayes cutoff and does not establish a business optimum.

Lowering thresholds generally catches more churners but increases false positives
and outreach volume. Higher thresholds save capacity at the cost of missed churners.
Precision is outreach yield; recall is captured churn; F1 balances these without costs.
Default policy remains fixed 0.5; CHURN_THRESHOLD_POLICY=scenario explicitly opts
into the configured validation-grid scenario threshold. CHURN_FN_COST and
CHURN_FP_COST configure its relative costs. No automatic production promotion.

## Frozen final selected model

Decision saved to MLflow BEFORE test access:

```json
{json.dumps(decision, indent=2)}
```

Refit chosen estimator/calibrator on train+validation only, with calibration OOF
folds confined to that development set. Then predict test probabilities once.
Refitting can change probability scale; validation threshold transfer is an assumption,
not a guarantee. No test-based correction or comparison to other models is made.

Final test metrics (only this selected run):

{
        markdown_table(
            pd.DataFrame(
                [
                    {
                        k: result["test"][k]
                        for k in ["precision", "recall", "f1", "roc_auc", "average_precision"]
                    }
                ]
            )
        )
    }

Confusion matrix, actual rows No/Yes and predicted columns No/Yes:
`{result["test"]["confusion_matrix"]}`. Full classification report in results.json.

## Tracking and artifact identity

Candidate child runs record complete model parameters, CV means/std/fold metrics,
validation metrics/confusion matrices, seed and feature configuration. Parent records
dataset/code/split hashes and dependency versions. Final child logs the frozen decision,
validation calibration/threshold tables, selected-model test metrics and evaluation,
figures/report and full preprocessing/calibration/model bundle.
Immutable reference: `{result["artifact_uri"]}`.
Artifact SHA-256: `{result["artifact_sha256"]}`.
Download via MLflow by this run ID/path; retrieval fingerprint and prediction parity
were verified on validation examples only. Local model: `{result["artifact"]}`.
The decision threshold is in metadata.json; apply it to predict_proba[:, 1] explicitly
because sklearn predict() generally uses its own 0.5 rule. Never resolve latest.

## Outstanding risks

No production acceptance thresholds or actual error costs have been approved.
Temporal validity, fairness, dependency locking, independent future validation,
threshold-transfer stability and deployment remain unresolved. No API, Docker,
production artifact promotion or registry dependency is introduced.
"""
    (directory / "validation_report.md").write_text(report, encoding="utf-8")
    (directory / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
