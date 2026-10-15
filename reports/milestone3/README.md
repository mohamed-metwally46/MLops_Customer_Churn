# Milestone 3 measured results

Run [a644309446d34b7ca25ea160124aeba6](a644309446d34b7ca25ea160124aeba6/validation_report.md)
contains the full report, reliability/threshold plots and aggregate results.json.
The parent and its five children are FINISHED in the local MLflow experiment.
Four candidates have training-CV and validation metrics; only final-selected has
test metrics. Its artifact is pinned to
`runs:/512d717fe25e46aea32ed10744c49541/model/pipeline.joblib` and was retrieved
and verified by fingerprint and validation prediction parity.

Historical baseline reports/model and archived executed notebooks are preserved.
Dataset and partition fingerprints matched the baseline before training.
CV AP reproduces Dummy 0.265368, Logistic Regression 0.664595, Random Forest
0.665271, HistGradientBoosting 0.662540. Raw Random Forest is retained; no new
feature/hyperparameter search was performed.

Validation calibration:

| Variant | Brier | Log loss | Uniform-bin ECE |
| --- | ---: | ---: | ---: |
| Raw | 0.139368 | 0.431260 | 0.022820 |
| Sigmoid | 0.139375 | 0.434286 | 0.020354 |

Raw wins the declared minimum-Brier rule. The difference is tiny and no statistically
supported probability-quality improvement is claimed. ECE gives a different ordering
and depends on bins; Brier/log loss also reflect discrimination.

Validation-only threshold examples:

| Threshold | Precision | Recall | F1 | TN | FP | FN | TP | Flagged |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.20 | 0.4587 | 0.8505 | 0.5960 | 494 | 282 | 42 | 239 | 49.29% |
| 0.30 | 0.5249 | 0.7509 | 0.6179 | 585 | 191 | 70 | 211 | 38.03% |
| 0.40 | 0.6103 | 0.6299 | 0.6200 | 663 | 113 | 104 | 177 | 27.44% |
| 0.45 | 0.6545 | 0.5730 | 0.6110 | 691 | 85 | 120 | 161 | 23.27% |
| 0.50 | 0.6850 | 0.4875 | 0.5696 | 713 | 63 | 144 | 137 | 18.92% |

Lower thresholds increase captured churn and outreach volume but reduce precision.
For illustrative equal FN/FP costs, 0.45 minimizes validation-grid cost (205 units),
only two units below 0.50 (207). It is not an approved business optimum. Configurable
costs and opt-in scenario policy are documented in the root README; measured execution
used default fixed 0.50. No scenario threshold was evaluated on test.

Final selected test evaluation at 0.50 reproduces precision 0.6500, recall 0.5107,
F1 0.5720, ROC-AUC 0.8490, AP 0.6598 and `[[700,77],[137,143]]`. The historical
holdout is reused; this is not fresh independent evidence. Do not retune from it.

Temporal validity, fairness, real business costs/capacity, calibration and threshold
transfer under future distribution shift, dependency locking and deployment remain
unresolved. No production inference integration was started.
