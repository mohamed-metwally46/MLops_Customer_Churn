"""Small, reproducible CV experiment with one final held-out test evaluation."""

import hashlib
import json
import platform
import subprocess
from importlib.metadata import version
from pathlib import Path
from typing import Any
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from churn.data import FEATURES, SEED
from churn.preprocessing import preprocessing

SCORING = {name: name for name in ["precision", "recall", "f1", "roc_auc", "average_precision"]}


def candidates(seed: int = SEED) -> dict[str, tuple[Pipeline, dict[str, list[Any]]]]:
    models = {
        "Dummy": (DummyClassifier(strategy="prior"), {}),
        "LogisticRegression": (
            LogisticRegression(max_iter=2000, random_state=seed),
            {"model__C": [0.1, 1.0], "model__class_weight": [None, "balanced"]},
        ),
        "RandomForest": (
            RandomForestClassifier(n_estimators=150, n_jobs=1, random_state=seed),
            {"model__max_depth": [8, None], "model__class_weight": [None, "balanced"]},
        ),
        "HistGradientBoosting": (
            HistGradientBoostingClassifier(max_iter=150, early_stopping=False, random_state=seed),
            {"model__max_leaf_nodes": [15, 31], "model__l2_regularization": [0.0, 5.0]},
        ),
    }
    return {
        name: (
            Pipeline(
                [
                    ("preprocess", preprocessing(scale=name == "LogisticRegression")),
                    ("select", "passthrough"),
                    ("model", model),
                ]
            ),
            grid,
        )
        for name, (model, grid) in models.items()
    }


def scores(y: pd.Series, probabilities: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    predictions = (probabilities >= threshold).astype(int)
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Model returned invalid probabilities.")
    return {
        "precision": float(precision_score(y, predictions, zero_division=0)),
        "recall": float(recall_score(y, predictions, zero_division=0)),
        "f1": float(f1_score(y, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "average_precision": float(average_precision_score(y, probabilities)),
        "confusion_matrix": confusion_matrix(y, predictions, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            y,
            predictions,
            labels=[0, 1],
            target_names=["No", "Yes"],
            output_dict=True,
            zero_division=0,
        ),
    }


def cv_summary(result: dict[str, Any], index: int) -> dict[str, Any]:
    return {
        metric: {
            "mean": float(result[f"mean_test_{metric}"][index]),
            "std": float(result[f"std_test_{metric}"][index]),
        }
        for metric in SCORING
    }


def rank_features(
    model: Pipeline, x: pd.DataFrame, y: pd.Series, repeats: int = 10
) -> pd.DataFrame:
    """Permute original columns, grouping their one-hot levels naturally."""
    importance = permutation_importance(
        model,
        x,
        y,
        scoring="average_precision",
        n_repeats=repeats,
        random_state=SEED,
        n_jobs=1,
    )
    return pd.DataFrame(
        {
            "feature": x.columns,
            "ap_decrease": importance.importances_mean,
            "repeat_std": importance.importances_std,
        }
    ).sort_values("ap_decrease", ascending=False, ignore_index=True)


def aggregate_errors(x: pd.DataFrame, y: pd.Series, probability: np.ndarray) -> dict[str, Any]:
    """Aggregate only; suppress groups with fewer than 20 records."""
    prediction = probability >= 0.5
    result = {}
    for label, mask in {
        "false_positive": (y == 0) & prediction,
        "false_negative": (y == 1) & ~prediction,
    }.items():
        subset = x.loc[mask]
        result[label] = {"count": int(mask.sum())}
        if len(subset) >= 20:
            result[label]["numeric_means"] = {
                c: float(pd.to_numeric(subset[c], errors="coerce").mean())
                for c in ["tenure", "MonthlyCharges"]
            }
            result[label]["contract_counts"] = {
                str(k): int(v) for k, v in subset["Contract"].value_counts().items() if v >= 20
            }
    return result


def run_experiment(
    splits: dict[str, tuple[pd.DataFrame, pd.Series]],
    quality: dict[str, Any],
    root: Path,
    folds: int = 5,
) -> dict[str, Any]:
    """Train/CV select; validate explanations; freeze; refit; evaluate test once."""
    with threadpool_limits(limits=1):
        return _run_experiment(splits, quality, root, folds)


def _run_experiment(
    splits: dict[str, tuple[pd.DataFrame, pd.Series]],
    quality: dict[str, Any],
    root: Path,
    folds: int,
) -> dict[str, Any]:
    train_x, train_y = splits["train"]
    validation_x, validation_y = splits["validation"]
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    searches, comparison = {}, {}
    for name, (pipeline, grid) in candidates().items():
        search = GridSearchCV(
            pipeline,
            grid,
            scoring=SCORING,
            refit="average_precision",
            cv=cv,
            n_jobs=1,
            error_score="raise",
            return_train_score=False,
        )
        search.fit(train_x, train_y)
        searches[name] = search
        comparison[name] = {
            "parameters": search.best_params_,
            "cv": cv_summary(search.cv_results_, search.best_index_),
        }
        print(f"Completed CV: {name}", flush=True)
    selected_name = max(searches, key=lambda name: searches[name].best_score_)
    base = searches[selected_name].best_estimator_
    # Supervised selection is fitted inside EVERY CV fold, never on validation/test.
    sparse = clone(base).set_params(
        select=SelectFromModel(
            LogisticRegression(
                penalty="l1", solver="liblinear", C=0.1, max_iter=2000, random_state=SEED
            ),
            threshold="mean",
        )
    )
    selected_cv = cross_validate(
        sparse, train_x, train_y, cv=cv, scoring=SCORING, error_score="raise", return_estimator=True
    )
    original_scores = np.array(
        [
            searches[selected_name].cv_results_[f"split{i}_test_average_precision"][
                searches[selected_name].best_index_
            ]
            for i in range(folds)
        ]
    )
    differences = selected_cv["test_average_precision"] - original_scores
    paired_se = float(differences.std(ddof=1) / np.sqrt(folds))
    # Require evidence of improvement; default to all features when uncertain.
    use_selection = bool(differences.mean() > paired_se)
    selected = sparse.fit(train_x, train_y) if use_selection else base
    selection = {
        "method": "L1 SelectFromModel, C=0.1, threshold=mean; fitted within each CV fold",
        "rule": "Use selector only if mean paired CV AP improvement exceeds one paired SE",
        "mean_ap_difference": float(differences.mean()),
        "paired_standard_error": paired_se,
        "used": use_selection,
        "cv": {
            m: {
                "mean": float(selected_cv[f"test_{m}"].mean()),
                "std": float(selected_cv[f"test_{m}"].std()),
            }
            for m in SCORING
        },
        "selected_encoded_counts_by_fold": [
            int(e.named_steps["select"].get_support().sum()) for e in selected_cv["estimator"]
        ],
    }
    validation_metrics = scores(validation_y, selected.predict_proba(validation_x)[:, 1])
    importance = rank_features(selected, validation_x, validation_y)
    # Stability check on independent CV validation folds, with all fitting fold-local.
    stability = []
    for train_indices, validation_indices in cv.split(train_x, train_y):
        fold_model = clone(selected).fit(train_x.iloc[train_indices], train_y.iloc[train_indices])
        ranked = rank_features(
            fold_model,
            train_x.iloc[validation_indices],
            train_y.iloc[validation_indices],
            repeats=3,
        )
        stability.append(ranked.set_index("feature")["ap_decrease"])
    stable = pd.concat(stability, axis=1)
    importance["cv_mean_ap_decrease"] = importance["feature"].map(stable.mean(axis=1))
    importance["cv_std_ap_decrease"] = importance["feature"].map(stable.std(axis=1))
    importance["cv_top5_frequency"] = importance["feature"].map(
        (stable.rank(ascending=False) <= 5).mean(axis=1)
    )
    # Freeze algorithm, feature method, threshold before ever predicting test data.
    development_x = pd.concat([train_x, validation_x])
    development_y = pd.concat([train_y, validation_y])
    final = clone(selected).fit(development_x, development_y)
    encoded = final.named_steps["preprocess"].get_feature_names_out()
    if use_selection:
        encoded = encoded[final.named_steps["select"].get_support()]
    test_x, test_y = splits["test"]
    probability = final.predict_proba(test_x)[:, 1]  # sole final test prediction
    test_metrics = scores(test_y, probability)
    artifact_id = f"{quality['sha256'][:12]}-seed{SEED}-{selected_name}-{uuid4().hex[:8]}"
    artifact_dir = root / "models" / artifact_id
    if artifact_dir.exists():
        raise FileExistsError("Artifact already exists; preserve it or choose a fresh output root.")
    artifact_dir.mkdir(parents=True)
    artifact = artifact_dir / "pipeline.joblib"
    joblib.dump(final, artifact)
    # Smoke test only a validation fixture; do not revisit the test set.
    reloaded = joblib.load(artifact)
    np.testing.assert_allclose(
        reloaded.predict_proba(validation_x.iloc[:2]), final.predict_proba(validation_x.iloc[:2])
    )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=False
    )
    result = {
        "dataset_sha256": quality["sha256"],
        "seed": SEED,
        "folds": folds,
        "split_sizes": {k: len(v[0]) for k, v in splits.items()},
        "split_index_hashes": {
            k: hashlib.sha256(np.asarray(v[0].index, dtype="int64").tobytes()).hexdigest()
            for k, v in splits.items()
        },
        "comparison": comparison,
        "selected_model": selected_name,
        "selection": selection,
        "threshold": 0.5,
        "validation": validation_metrics,
        "test": test_metrics,
        "importance": importance.to_dict(orient="records"),
        "errors": aggregate_errors(test_x, test_y, probability),
        "input_features": FEATURES,
        "selected_encoded_features": encoded.tolist(),
        "artifact": artifact.relative_to(root).as_posix(),
        "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "python": platform.python_version(),
        "dependencies": {p: version(p) for p in ["scikit-learn", "pandas", "numpy", "joblib"]},
        "code_commit": commit.stdout.strip() if commit.returncode == 0 else "uncommitted",
        "code_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path(__file__).parent.glob("*.py")
        },
    }
    (artifact_dir / "metadata.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
