"""Version-one inference contract derived from the inspected training feature definition."""

from typing import Annotated, Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, create_model, field_validator

from churn.data import CATEGORIES, FEATURES


class CustomerBase(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    @field_validator("*", mode="before")
    @classmethod
    def normalize_missing(cls, value: Any, info: ValidationInfo) -> Any:
        if isinstance(value, bool):
            raise ValueError("Boolean values are not valid feature values.")
        if info.field_name == "SeniorCitizen" and value is not None and type(value) is not int:
            raise ValueError("SeniorCitizen requires a JSON integer.")
        if info.field_name == "TotalCharges" and isinstance(value, str) and not value.strip():
            return None
        return value

    def to_frame(self) -> pd.DataFrame:
        """Only map explicit nulls to NaN; use the saved pipeline for all preprocessing."""
        values = self.model_dump()
        return pd.DataFrame({c: [np.nan if values[c] is None else values[c]] for c in FEATURES})


Customer = create_model(
    "Customer",
    __base__=CustomerBase,
    tenure=(Annotated[int, Field(ge=0, le=int(np.iinfo(np.int64).max))] | None, ...),
    MonthlyCharges=(Annotated[float, Field(ge=0, le=float(np.finfo(np.float32).max))] | None, ...),
    TotalCharges=(
        Annotated[float, Field(ge=0, le=float(np.finfo(np.float32).max))]
        | Annotated[str, Field(pattern=r"^\s*$")]
        | None,
        ...,
    ),
    **{
        name: (Literal[tuple(sorted(allowed, key=str))] | None, ...)
        for name, allowed in CATEGORIES.items()
    },
)


class Prediction(BaseModel):
    prediction: Literal["Yes", "No"]
    churn_probability: float = Field(ge=0, le=1, allow_inf_nan=False)
    threshold: float = Field(ge=0, le=1, allow_inf_nan=False)
    model_version: str


class Health(BaseModel):
    status: Literal["ok"] = "ok"


class Ready(BaseModel):
    status: Literal["ready"] = "ready"


class ModelInfo(BaseModel):
    model_name: str
    model_version: str
    artifact_sha256: str
    threshold: float
    positive_class: Literal["Churn=Yes"] = "Churn=Yes"
    input_schema: dict[str, Any]


def example_customer() -> dict[str, Any]:
    """Synthetic example; real feature names/categories, no copied customer record."""
    values = {name: sorted(allowed, key=str)[0] for name, allowed in CATEGORIES.items()}
    values.update(
        tenure=12,
        MonthlyCharges=65.0,
        TotalCharges=780.0,
        PhoneService="Yes",
        MultipleLines="No",
        InternetService="DSL",
        OnlineSecurity="Yes",
        OnlineBackup="No",
        DeviceProtection="No",
        TechSupport="Yes",
        StreamingTV="No",
        StreamingMovies="No",
    )
    return values
