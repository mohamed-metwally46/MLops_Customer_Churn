"""Read-only Telco validation and deterministic partitioning; never log records."""

import hashlib
import os
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORIES = {
    "gender": {"Female", "Male"},
    "SeniorCitizen": {0, 1},
    "Partner": {"Yes", "No"},
    "Dependents": {"Yes", "No"},
    "PhoneService": {"Yes", "No"},
    "MultipleLines": {"Yes", "No", "No phone service"},
    "InternetService": {"DSL", "Fiber optic", "No"},
    **{
        name: {"Yes", "No", "No internet service"}
        for name in [
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
        ]
    },
    "Contract": {"Month-to-month", "One year", "Two year"},
    "PaperlessBilling": {"Yes", "No"},
    "PaymentMethod": {
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check",
    },
}
FEATURES = NUMERIC + list(CATEGORIES)


def dataset_path(root: Path) -> Path:
    """Resolve relative paths against the project root, independent of notebook cwd."""
    return root / os.getenv("CHURN_DATA_PATH", "dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv")


def numeric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Stateless parsing shared by training and serialized inference pipelines."""
    return frame.apply(
        lambda column: pd.to_numeric(
            column.mask(column.astype(str).str.strip().eq("")), errors="raise"
        )
    )


def validate(frame: pd.DataFrame) -> dict[str, Any]:
    """Validate the inspected schema. Errors name columns, never customer values."""
    expected = set(FEATURES) | {"customerID", "Churn"}
    if set(frame.columns) != expected:
        raise ValueError("Schema mismatch: expected the documented 21 Telco columns.")
    if frame.empty:
        raise ValueError("Dataset contains no rows.")
    if frame["Churn"].isna().any() or set(frame["Churn"].unique()) != {"Yes", "No"}:
        raise ValueError("Churn must contain both Yes and No, with no missing target.")
    if frame["customerID"].isna().any() or frame["customerID"].duplicated().any():
        raise ValueError("customerID must be present and unique; inspect duplicate accounts.")
    if frame["customerID"].astype(str).str.strip().eq("").any():
        raise ValueError("customerID contains blank identifiers.")
    if frame.duplicated().any():
        raise ValueError("Duplicate rows found; resolve provenance before splitting.")
    for name, allowed in CATEGORIES.items():
        if not set(frame[name].dropna().unique()).issubset(allowed):
            raise ValueError(f"Unexpected category in {name}; review the dataset schema.")
    try:
        numbers = numeric_frame(frame[NUMERIC])
    except (ValueError, TypeError):
        raise ValueError("Numeric parsing failed; inspect tenure and charge columns.") from None
    if ((numbers < 0) | numbers.isin([float("inf"), -float("inf")])).any().any():
        raise ValueError("Numeric features must be finite, nonnegative or missing.")
    if numbers["tenure"].dropna().mod(1).ne(0).any():
        raise ValueError("tenure must be an integer number of months.")
    constant = [c for c in FEATURES if frame[c].nunique(dropna=False) <= 1]
    if constant:
        raise ValueError("Constant predictors found; review the feature contract.")
    return {
        "shape": list(frame.shape),
        "target": "Churn",
        "target_counts": {str(k): int(v) for k, v in frame["Churn"].value_counts().items()},
        "duplicate_rows": int(frame.duplicated().sum()),
        "constant_columns": constant,
        "columns": {
            c: {
                "dtype": str(frame[c].dtype),
                "missing": int(frame[c].isna().sum()),
                "blank": int(frame[c].astype(str).str.strip().eq("").sum()),
                "unique": int(frame[c].nunique()),
            }
            for c in frame
        },
        "parsed_numeric_missing": numbers.isna().sum().astype(int).to_dict(),
        "zero_tenure_missing_total": int(
            (numbers["TotalCharges"].isna() & numbers["tenure"].eq(0)).sum()
        ),
        "high_cardinality_columns": [c for c in frame if frame[c].nunique() > len(frame) / 2],
        "service_inconsistencies": int(
            sum(
                (frame[c].eq("No internet service") != frame["InternetService"].eq("No")).sum()
                for c in [
                    "OnlineSecurity",
                    "OnlineBackup",
                    "DeviceProtection",
                    "TechSupport",
                    "StreamingTV",
                    "StreamingMovies",
                ]
            )
            + (
                frame["MultipleLines"].eq("No phone service") != frame["PhoneService"].eq("No")
            ).sum()
        ),
    }


def load_dataset(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError("Dataset missing; set CHURN_DATA_PATH to the local Telco CSV.")
    frame = pd.read_csv(path)
    quality = validate(frame)
    quality["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return frame, quality


def partitions(frame: pd.DataFrame, seed: int = SEED) -> dict[str, tuple[pd.DataFrame, pd.Series]]:
    """70/15/15 stratified partition. Return raw features; fitting happens later."""
    x = frame[FEATURES].copy()
    y = frame["Churn"].map({"No": 0, "Yes": 1})
    development_x, test_x, development_y, test_y = train_test_split(
        x, y, test_size=0.15, stratify=y, random_state=seed
    )
    train_x, validation_x, train_y, validation_y = train_test_split(
        development_x,
        development_y,
        test_size=0.15 / 0.85,
        stratify=development_y,
        random_state=seed,
    )
    return {
        "train": (train_x, train_y),
        "validation": (validation_x, validation_y),
        "test": (test_x, test_y),
    }
