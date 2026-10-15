"""Aggregate-only EDA and report generation on the training partition."""

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from churn.data import CATEGORIES, NUMERIC, numeric_frame


def markdown_table(frame: pd.DataFrame, digits: int = 4) -> str:
    """Render small aggregate tables without an optional tabulate dependency."""
    formatted = frame.copy()
    for c in formatted.select_dtypes("number"):
        formatted[c] = formatted[c].map(lambda x: f"{x:.{digits}f}")
    rows = [" | ".join(map(str, formatted.columns)), " | ".join(["---"] * len(formatted.columns))]
    rows += [" | ".join(map(str, row)) for row in formatted.itertuples(index=False, name=None)]
    return "\n".join(rows)


def categorical_rates(x: pd.DataFrame, y: pd.Series, column: str) -> pd.DataFrame:
    grouped = (
        pd.DataFrame({"category": x[column], "churn": y})
        .groupby("category", observed=True, dropna=False)["churn"]
        .agg(count="size", churn_rate="mean")
    )
    return grouped.loc[grouped["count"] >= 20].reset_index()


def create_eda(
    train_x: pd.DataFrame,
    train_y: pd.Series,
    quality: dict[str, Any],
    reports: Path,
) -> dict[str, Any]:
    """Write figures and descriptive findings, with no customer-level outputs."""
    reports.mkdir(parents=True, exist_ok=True)
    figures = reports / "figures"
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({"figure.dpi": 110, "font.size": 10})
    numbers = numeric_frame(train_x[NUMERIC])
    counts = pd.Series(quality["target_counts"])
    fig, axis = plt.subplots(figsize=(5, 3))
    counts.plot.bar(ax=axis, color=["steelblue", "tomato"], rot=0)
    axis.set(xlabel="Observed Churn label", ylabel="Accounts", title="Dataset class distribution")
    fig.tight_layout()
    fig.savefig(figures / "target_distribution.png")
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    outliers = {}
    for i, c in enumerate(NUMERIC):
        for label, color in [(0, "steelblue"), (1, "tomato")]:
            axes[0, i].hist(
                numbers.loc[train_y.eq(label), c].dropna(),
                bins=25,
                density=True,
                alpha=0.5,
                color=color,
                label=f"Churn={label}",
            )
        axes[0, i].set(xlabel=c, ylabel="Density", title=f"Training {c}")
        axes[0, i].legend()
        axes[1, i].boxplot(
            [numbers.loc[train_y.eq(label), c].dropna() for label in [0, 1]],
            tick_labels=["No churn", "Churn"],
            showfliers=False,
        )
        axes[1, i].set(xlabel="Target", ylabel=c, title="Median and IQR; fliers hidden")
        q1, q3 = numbers[c].quantile([0.25, 0.75])
        outliers[c] = int(
            ((numbers[c] < q1 - 1.5 * (q3 - q1)) | (numbers[c] > q3 + 1.5 * (q3 - q1))).sum()
        )
    fig.tight_layout()
    fig.savefig(figures / "numeric_distributions.png")
    plt.close(fig)

    rates = {c: categorical_rates(train_x, train_y, c) for c in CATEGORIES}
    fig, axes = plt.subplots(4, 4, figsize=(19, 18))
    for axis, (c, table) in zip(axes.flat, rates.items(), strict=True):
        axis.bar(table["category"].astype(str), table["churn_rate"], color="steelblue")
        axis.set(xlabel=c, ylabel="Churn fraction", ylim=(0, 1), title=f"Training: {c}")
        axis.tick_params(axis="x", labelrotation=35, labelsize=8)
    fig.tight_layout()
    fig.savefig(figures / "categorical_churn_rates.png")
    plt.close(fig)

    # Pearson for continuous/numeric variables, Spearman to expose monotonic redundancy.
    numeric_target = numbers.assign(Churn=train_y)
    correlations = {}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for axis, method in zip(axes, ["pearson", "spearman"], strict=True):
        corr = numeric_target.corr(method=method)
        correlations[method] = corr.to_dict()
        image = axis.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
        axis.set_xticks(range(len(corr)), corr.columns, rotation=35, ha="right")
        axis.set_yticks(range(len(corr)), corr.columns)
        axis.set_title(f"Training {method} correlation")
        for i in range(len(corr)):
            for j in range(len(corr)):
                axis.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center")
    fig.colorbar(image, ax=axes.tolist(), shrink=0.7)
    fig.savefig(figures / "correlations.png", bbox_inches="tight")
    plt.close(fig)

    target_rate = counts["Yes"] / counts.sum()
    quality_table = pd.DataFrame(quality["columns"]).T.reset_index(names="column")
    report = f"""# EDA and data quality

Executed on the supplied CSV. Shape: {quality["shape"][0]} rows × {quality["shape"][1]} columns.
Verified target: Churn; No={counts["No"]}, Yes={counts["Yes"]} ({target_rate:.2%} positive).
Dataset SHA-256: `{quality["sha256"]}`.

Full-data inspection is limited to schema, quality, and overall class counts.
All target relationships, distributions, and correlations below use ONLY the training partition
({len(train_x)} rows), so held-out test behavior does not guide feature/model choices.

## Quality and handling

{markdown_table(quality_table)}

- Duplicate rows: {quality["duplicate_rows"]}; constant columns: {quality["constant_columns"]}.
- High-cardinality columns: {quality["high_cardinality_columns"]}.
  customerID excluded as identifier.
- TotalCharges has {quality["columns"]["TotalCharges"]["blank"]} blank strings;
  {quality["zero_tenure_missing_total"]} coincide with zero tenure. Parse blanks to missing,
  retain every row, and median-impute inside each training fold.
  Do not assume a missing bill is zero.
- Parsed numeric missing counts: {quality["parsed_numeric_missing"]}.
- Service category inconsistency count: {quality["service_inconsistencies"]}.
- Negative/nonfinite numeric values, invalid categories, missing targets, duplicate IDs,
  schema changes, and constant predictors cause explicit validation errors.
- Training IQR outlier counts: {outliers}. Outliers are flags, not automatic deletion rules.

## Leakage and availability

Churn is the target and never enters the predictors. customerID is unique and has no justified
predictive role. No explicit post-outcome column was found. Contract, tenure, services, payment
method, and accumulated charges are provisionally available at the scoring snapshot; the CSV
does not establish their timing relative to churn. Validate as-of timestamps and the prediction
horizon with the data owner before production. These associations do not establish causation.
gender and SeniorCitizen may create fairness risks; retain for this exploratory comparison only,
with subgroup validation and governance required before deployment.

## Training-only relationships

![Class counts](figures/target_distribution.png)
![Numerical variables](figures/numeric_distributions.png)
![Categorical churn fractions](figures/categorical_churn_rates.png)
![Numerical correlations](figures/correlations.png)

Numeric summary:

{markdown_table(numbers.describe().T.reset_index(names="feature"))}

Numeric correlation with churn (Pearson; binary target gives point-biserial association):

{markdown_table(numeric_target.corr()["Churn"].drop("Churn").reset_index(name="correlation"))}

The plots compare normalized numeric distributions, so the majority class does not dominate
the histograms. Boxplots summarize distributions without publishing individual outlier points.
Churn fractions compare categories rather than raw counts; cells with fewer than 20 accounts
are suppressed. Pearson/Spearman correlations assess redundancy, not feature selection.
TotalCharges and tenure share accumulated-exposure information; correlated predictors can
divide permutation importance. Service flags also encode structural redundancy.
The minority churn class makes average precision a useful selection metric, with a random
ranking reference near the positive prevalence. Recall measures missed churners; precision
measures how many flagged accounts actually churn. F1 balances these at a fixed threshold;
ROC-AUC summarizes ranking across thresholds but can obscure operational precision.

"""
    for c in ["Contract", "InternetService", "PaymentMethod", "OnlineSecurity", "TechSupport"]:
        report += f"### {c}\n\n{markdown_table(rates[c])}\n\n"
        ordered = rates[c].sort_values("churn_rate")
        low, high = ordered.iloc[0], ordered.iloc[-1]
        report += (
            f"On training data, {c}={high['category']} has the highest observed churn fraction "
            f"({high['churn_rate']:.1%}, n={int(high['count'])}); "
            f"{low['category']} has the lowest ({low['churn_rate']:.1%}, "
            f"n={int(low['count'])}). This is descriptive association, not an adjusted effect.\n\n"
        )
    means = numbers.groupby(train_y).mean()
    report += "### Main numerical findings\n\n"
    for c in NUMERIC:
        report += (
            f"Training mean {c}: churners {means.loc[1, c]:.2f}, "
            f"non-churners {means.loc[0, c]:.2f}. "
            "Differences may provide predictive signal but also reflect exposure/confounding.\n\n"
        )
    report += (
        "Use model/CV and validation permutation evidence in the model report to rank features.\n"
    )
    (reports / "eda_summary.md").write_text(report, encoding="utf-8")
    (reports / "data_quality.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")
    return {"quality": quality, "outliers": outliers, "correlations": correlations}


def model_report(result: dict[str, Any], reports: Path) -> None:
    rows = []
    for model, summary in result["comparison"].items():
        rows.append(
            {
                "model": model,
                **{m: f"{s['mean']:.4f} ± {s['std']:.4f}" for m, s in summary["cv"].items()},
            }
        )
    importance = pd.DataFrame(result["importance"])
    fig, axis = plt.subplots(figsize=(9, 7))
    view = importance.iloc[::-1]
    axis.barh(view["feature"], view["ap_decrease"], xerr=view["repeat_std"], color="steelblue")
    axis.set(
        xlabel="Validation AP decrease after permutation (± repeat SD)",
        ylabel="Original predictor",
        title="Permutation importance; no causal interpretation",
    )
    fig.tight_layout()
    fig.savefig(reports / "figures" / "feature_importance.png")
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(4, 4))
    matrix = np.asarray(result["test"]["confusion_matrix"])
    axis.imshow(matrix, cmap="Blues")
    axis.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["No", "Yes"],
        yticklabels=["No", "Yes"],
        xlabel="Predicted",
        ylabel="Actual",
        title="Final test; threshold 0.5",
    )
    for i in range(2):
        for j in range(2):
            axis.text(j, i, str(matrix[i, j]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(reports / "figures" / "test_confusion_matrix.png")
    plt.close(fig)
    selection = result["selection"]
    final_metrics = pd.DataFrame(
        [
            {
                "partition": name,
                **{
                    m: result[name][m]
                    for m in ["precision", "recall", "f1", "roc_auc", "average_precision"]
                },
            }
            for name in ["validation", "test"]
        ]
    )
    text = f"""# Model comparison and selection

Executed experiment; seed={result["seed"]}; split sizes={result["split_sizes"]}.
Dataset SHA-256: `{result["dataset_sha256"]}`. Code state: {result["code_commit"]}.
Python {result["python"]}; dependencies: {result["dependencies"]}.

## Protocol fixed before test evaluation

Four models use the same {result["folds"]}-fold shuffled stratified CV on the 70% training split.
All imputers, encoders, scaling and supervised selection are fitted inside folds. Numeric
medians and categorical modes impute missing values; unknown categories encode to all zeros.
Logistic Regression scales numeric inputs; tree models do not. Dense one-hot encoding is
feasible for this small dataset and keeps HistGradientBoosting compatible. No oversampling.
Class-weight None/balanced is compared for Logistic Regression and Random Forest; balanced
weights can improve recall at a cost to precision and probability calibration.
Select by maximum mean CV average precision; other metrics describe trade-offs.

{markdown_table(pd.DataFrame(rows))}

Hyperparameters selected within training CV:

```json
{json.dumps({k: v["parameters"] for k, v in result["comparison"].items()}, indent=2)}
```

Selected algorithm: **{result["selected_model"]}** by highest mean CV average precision.
CV scores are selection estimates (optimistic after tuning), not a nested-CV performance claim.

## Feature selection and stability

customerID is excluded; Churn is separated before preprocessing. No constant columns were
found. All remaining 19 source predictors are available to preprocessing.
A second comparison fits L1 Logistic Regression SelectFromModel (C=0.1, threshold=mean)
after encoding INSIDE every fold. Retain selection only when its paired mean AP improvement
exceeds one paired standard error; otherwise prefer the full feature set. This is a heuristic
complexity rule, not a significance test. Foldwise model selection is not nested.
Selection used: **{selection["used"]}**; AP difference={selection["mean_ap_difference"]:.5f};
paired SE={selection["paired_standard_error"]:.5f}.
Selected encoded feature counts by fold: {selection["selected_encoded_counts_by_fold"]}.
Selector CV metrics: {selection["cv"]}.
Final fitted encoded features ({len(result["selected_encoded_features"])}):
{", ".join(result["selected_encoded_features"])}.

Validation permutation importance permutes each raw column, grouping its encoded levels.
Ten repeats estimate perturbation variability (not confidence intervals). Stability is measured
by three-repeat permutation on each training CV validation fold. Top-5 frequency indicates
consistency across those five folds. It is conditional on the chosen hyperparameters, not an
independent feature-selection performance estimate. Negative/small importance can reflect
noise or redundancy; do not automatically discard it. Correlated features may substitute for
one another. Permutation importance gives magnitude, not direction or causation.

{markdown_table(importance)}

![Feature importance](figures/feature_importance.png)

## Frozen final evaluation

Threshold fixed at 0.5; it was not tuned on the test set. After selection and validation
explanations, refit the selected full pipeline on train+validation (85%). Predict held-out
test probabilities once and reuse those probabilities for every test metric/error summary.
Validation metrics below describe the earlier training-only fit, not the final 85% refit.

{markdown_table(final_metrics)}

Test confusion matrix: rows actual [No, Yes], columns predicted [No, Yes]:
`{result["test"]["confusion_matrix"]}`.

![Confusion matrix](figures/test_confusion_matrix.png)

Classification report:

{
        markdown_table(
            pd.DataFrame(
                {
                    k: v
                    for k, v in result["test"]["classification_report"].items()
                    if isinstance(v, dict)
                }
            ).T.reset_index(names="label")
        )
    }

Aggregate test errors (groups below 20 suppressed; no raw records or identifiers):

```json
{json.dumps(result["errors"], indent=2)}
```

False negatives miss potential churners; false positives spend retention resources on
non-churners. Lower thresholds generally increase recall and false positives. Choose a
cost-based threshold only on future training/validation data after business costs are agreed.
Class weights can distort calibration; no calibrated-risk claim is made. Test error summaries
are diagnostic only and do not trigger retuning in this run.

## Artifact and limits

Full parse/impute/encode/scale/optional-select/model bundle: `{result["artifact"]}` (Git ignored).
Artifact SHA-256: `{result["artifact_sha256"]}`. Load trusted artifacts with `joblib.load`;
call `predict_proba` with the 19 input features listed in `model_results.json`. No manual
preprocessing is needed. No automatic artifact promotion, production gate, or API exists yet.
Dataset and source-code hashes, split hashes, seed, dependency versions, CV parameters and
metrics are recorded in model_results.json and the artifact's metadata.json. Exact dependency
locking remains outstanding; this is not a clean-environment reproducibility guarantee.

Historical snapshot data cannot establish temporal validity, fairness, causal effects, or
future performance. No external/temporal validation, subgroup fairness assessment, calibration,
deployment, or drift monitoring is implemented. Small CV differences should not be treated
as proof of superiority. Held-out test results must not drive another selection cycle.
MLflow integration remains Milestone 3; these structured outputs can be logged there.
"""
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "model_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (reports / "model_comparison.md").write_text(text, encoding="utf-8")
