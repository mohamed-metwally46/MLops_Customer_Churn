"""Fold-fitted preprocessing, also included in the persisted prediction bundle."""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from churn.data import CATEGORIES, NUMERIC, numeric_frame


def preprocessing(scale: bool = False) -> ColumnTransformer:
    numeric = Pipeline(
        [
            ("parse", FunctionTransformer(numeric_frame, feature_names_out="one-to-one")),
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler() if scale else "passthrough"),
        ]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [("numeric", numeric, NUMERIC), ("categorical", categorical, list(CATEGORIES))],
        remainder="drop",
        verbose_feature_names_out=True,
    )
