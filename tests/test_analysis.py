"""Deterministic synthetic tests; no external data or service required."""

import joblib
import numpy as np
import pandas as pd
import pytest

from churn.data import CATEGORIES, FEATURES, numeric_frame, partitions, validate
from churn.preprocessing import preprocessing
from churn.train import candidates, run_experiment, scores


@pytest.fixture
def telco() -> pd.DataFrame:
    size = 100
    frame = pd.DataFrame(
        {
            c: [sorted(values, key=str)[i % len(values)] for i in range(size)]
            for c, values in CATEGORIES.items()
        }
    )
    frame["tenure"] = np.arange(size) % 73
    frame["MonthlyCharges"] = np.arange(size) + 20.0
    frame["TotalCharges"] = (frame["tenure"] * frame["MonthlyCharges"]).astype(str)
    frame.loc[0, "TotalCharges"] = " "
    frame["customerID"] = [f"synthetic-{i}" for i in range(size)]
    frame["Churn"] = ["Yes" if i % 3 == 0 else "No" for i in range(size)]
    return frame


def test_validation_handles_blank_without_dropping_rows(telco: pd.DataFrame) -> None:
    original = telco.copy(deep=True)
    quality = validate(telco)
    assert quality["parsed_numeric_missing"]["TotalCharges"] == 1
    assert quality["zero_tenure_missing_total"] == 1
    pd.testing.assert_frame_equal(telco, original)


@pytest.mark.parametrize(
    "issue",
    [
        "schema",
        "category",
        "target",
        "identifier",
        "negative",
        "nonfinite",
        "parse",
        "constant",
        "fractional",
    ],
)
def test_validation_rejects_bad_data(telco: pd.DataFrame, issue: str) -> None:
    if issue == "schema":
        telco = telco.drop(columns="Contract")
    elif issue == "category":
        telco.loc[0, "Contract"] = "secret-unexpected-value"
    elif issue == "target":
        telco.loc[0, "Churn"] = None
    elif issue == "identifier":
        telco.loc[0, "customerID"] = telco.loc[1, "customerID"]
    elif issue == "negative":
        telco.loc[0, "MonthlyCharges"] = -1
    elif issue == "nonfinite":
        telco.loc[0, "TotalCharges"] = "inf"
    elif issue == "parse":
        telco.loc[0, "TotalCharges"] = "secret-unexpected-value"
    elif issue == "constant":
        telco["gender"] = "Female"
    else:
        telco["tenure"] = telco["tenure"].astype(float)
        telco.loc[0, "tenure"] = 0.5
    with pytest.raises(ValueError) as error:
        validate(telco)
    assert "secret-unexpected-value" not in str(error.value)


def test_split_is_disjoint_stratified_and_repeatable(telco: pd.DataFrame) -> None:
    split = partitions(telco)
    again = partitions(telco)
    indices = [set(x.index) for x, _ in split.values()]
    assert len(set.union(*indices)) == len(telco)
    assert not (indices[0] & indices[1] or indices[1] & indices[2] or indices[0] & indices[2])
    for name, (x, y) in split.items():
        pd.testing.assert_frame_equal(x, again[name][0])
        assert "customerID" not in x and "Churn" not in x
        assert abs(y.mean() - telco["Churn"].eq("Yes").mean()) < 0.1


def test_preprocessing_fits_training_only_and_handles_unknowns(telco: pd.DataFrame) -> None:
    train = telco[FEATURES].iloc[:70].copy()
    transformer = preprocessing(scale=True).fit(train)
    imputer = transformer.named_transformers_["numeric"].named_steps["impute"]
    expected = numeric_frame(train[["tenure", "MonthlyCharges", "TotalCharges"]]).median()
    np.testing.assert_allclose(imputer.statistics_, expected)
    saved_statistics = imputer.statistics_.copy()
    unseen = train.iloc[:3].copy()
    unseen["Contract"] = "future-category"
    unseen["TotalCharges"] = " "
    encoded = transformer.transform(unseen)
    assert np.isfinite(encoded).all()
    np.testing.assert_array_equal(imputer.statistics_, saved_statistics)


@pytest.mark.parametrize("name", list(candidates()))
def test_pipeline_serialization_and_probability_contract(telco, tmp_path, name) -> None:
    pipeline, _ = candidates()[name]
    if name == "RandomForest":
        pipeline.set_params(model__n_estimators=5)
    if name == "HistGradientBoosting":
        pipeline.set_params(model__max_iter=5)
    x = telco[FEATURES]
    y = telco["Churn"].eq("Yes").astype(int)
    pipeline.fit(x, y)
    path = tmp_path / "fixture.joblib"
    joblib.dump(pipeline, path)
    restored = joblib.load(path)
    probabilities = restored.predict_proba(x.iloc[:4])
    assert probabilities.shape == (4, 2)
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)
    np.testing.assert_allclose(probabilities, pipeline.predict_proba(x.iloc[:4]))


def test_metric_threshold_and_invalid_probabilities() -> None:
    y = pd.Series([0, 0, 1, 1])
    result = scores(y, np.array([0.1, 0.8, 0.4, 0.9]))
    assert result["confusion_matrix"] == [[1, 1], [1, 1]]
    assert result["recall"] == 0.5
    with pytest.raises(ValueError, match="invalid probabilities"):
        scores(y, np.array([0.1, 0.2, float("nan"), 0.9]))


def test_experiment_evaluates_test_once_and_saves_complete_bundle(
    telco,
    tmp_path,
    monkeypatch,
) -> None:
    """Exercise orchestration cheaply, while checking the held-out boundary."""
    import json

    from sklearn.pipeline import Pipeline

    from churn import train

    split = partitions(telco)
    test_indices = split["test"][0].index
    original_predict = Pipeline.predict_proba
    calls = []

    def audited_predict(self, x, **kwargs):
        if isinstance(x, pd.DataFrame) and x.index.equals(test_indices):
            calls.append("test")
        return original_predict(self, x, **kwargs)

    pipeline, _ = candidates()["LogisticRegression"]
    monkeypatch.setattr(train, "candidates", lambda: {"LogisticRegression": (pipeline, {})})
    monkeypatch.setattr(Pipeline, "predict_proba", audited_predict)
    quality = validate(telco)
    quality["sha256"] = "synthetic-test-fingerprint"
    result = run_experiment(split, quality, tmp_path, folds=2)
    assert calls == ["test"]
    # sklearn rounds fractional sizes up; on 100 rows the inner split is 16 rows.
    assert result["split_sizes"] == {"train": 69, "validation": 16, "test": 15}
    assert result["threshold"] == 0.5
    assert set(result["input_features"]) == set(FEATURES)
    assert len(result["importance"]) == len(FEATURES)
    artifact = tmp_path / result["artifact"]
    assert artifact.is_file()
    restored = joblib.load(artifact)
    assert restored.predict_proba(split["validation"][0]).shape == (16, 2)
    metadata = json.loads((artifact.parent / "metadata.json").read_text())
    assert metadata["artifact_sha256"] == result["artifact_sha256"]
    assert "customerID" not in metadata["input_features"]
    assert "synthetic-0" not in json.dumps(metadata)
