"""Validated binary threshold and calibration summaries; no fitting or record exports."""

import numpy as np
from sklearn.metrics import brier_score_loss, log_loss

from churn.train import scores


def binary_inputs(y, probabilities) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(y)
    probability = np.asarray(probabilities, dtype=float)
    if labels.ndim != 1 or probability.ndim != 1 or len(labels) != len(probability):
        raise ValueError("Labels/probabilities must be aligned one-dimensional arrays.")
    if not len(labels) or not np.isin(labels, [0, 1]).all() or len(np.unique(labels)) != 2:
        raise ValueError("Evaluation requires nonempty binary labels with both classes.")
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("Probabilities must be finite and in [0, 1].")
    return labels.astype(int), probability


def validate_costs(false_negative_cost: float, false_positive_cost: float) -> None:
    costs = np.asarray([false_negative_cost, false_positive_cost], dtype=float)
    if not np.isfinite(costs).all() or (costs < 0).any() or costs.sum() <= 0:
        raise ValueError("Costs must be finite, nonnegative, and not both zero.")


def threshold_metrics(y, probabilities, threshold: float) -> dict:
    labels, probability = binary_inputs(y, probabilities)
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Threshold must be finite and in [0, 1].")
    metrics = scores(labels, probability, threshold)
    tn, fp, fn, tp = np.asarray(metrics["confusion_matrix"]).ravel()
    return {
        "threshold": float(threshold),
        **{k: metrics[k] for k in ["precision", "recall", "f1"]},
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "flagged_percent": float(100 * (tp + fp) / len(labels)),
    }


def threshold_table(y, probabilities, thresholds=None, fn_cost=1.0, fp_cost=1.0) -> list[dict]:
    """Cost units are illustrative/configurable, never inferred company costs."""
    validate_costs(fn_cost, fp_cost)
    grid = np.linspace(0, 1, 21) if thresholds is None else np.asarray(thresholds, dtype=float)
    if grid.ndim != 1 or not len(grid) or len(np.unique(grid)) != len(grid):
        raise ValueError("Threshold grid must be a nonempty unique one-dimensional sequence.")
    rows = []
    for threshold in sorted(grid):
        row = threshold_metrics(y, probabilities, threshold)
        row["scenario_cost"] = float(fn_cost * row["fn"] + fp_cost * row["fp"])
        row["cost_per_customer"] = row["scenario_cost"] / len(y)
        rows.append(row)
    return rows


def select_threshold(y, probabilities, fn_cost=1.0, fp_cost=1.0, thresholds=None) -> dict:
    """Minimize validation-grid scenario cost; ties prefer fewer flagged, then higher t."""
    rows = threshold_table(y, probabilities, thresholds, fn_cost, fp_cost)
    return min(rows, key=lambda r: (r["scenario_cost"], r["flagged_percent"], -r["threshold"]))


def calibration_metrics(y, probabilities, bins: int = 10) -> dict:
    """Uniform-bin reliability/ECE plus proper scores. No calibrator is fitted here."""
    labels, probability = binary_inputs(y, probabilities)
    if not isinstance(bins, int) or isinstance(bins, bool) or bins < 2:
        raise ValueError("Calibration bins must be an integer >= 2.")
    assignments = np.minimum((probability * bins).astype(int), bins - 1)
    reliability = []
    for index in range(bins):
        mask = assignments == index
        if mask.any():
            reliability.append(
                {
                    "lower": index / bins,
                    "upper": (index + 1) / bins,
                    "count": int(mask.sum()),
                    "mean_probability": float(probability[mask].mean()),
                    "observed_fraction": float(labels[mask].mean()),
                }
            )
    ece = sum(
        r["count"] * abs(r["mean_probability"] - r["observed_fraction"]) for r in reliability
    ) / len(labels)
    return {
        "brier": float(brier_score_loss(labels, probability)),
        "log_loss": float(log_loss(labels, probability, labels=[0, 1])),
        "ece_uniform": float(ece),
        "bins": bins,
        "reliability": reliability,
    }
