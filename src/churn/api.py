"""HTTP boundaries only; startup-loaded inference never imports training or MLflow."""

import logging
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from churn.data import FEATURES
from churn.model_loader import (
    LoadedModel,
    ModelSettings,
    ModelUnavailable,
    PredictionFailed,
    load_model,
)
from churn.schemas import Customer, Health, ModelInfo, Prediction, Ready

LOGGER = logging.getLogger(__name__)


def create_app(settings: ModelSettings | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.model = None
        try:
            app.state.model = load_model(settings or ModelSettings.from_environment())
        except (ModelUnavailable, ValueError, OSError):
            # Intentional degraded startup: liveness stays accessible, readiness fails closed.
            LOGGER.error("Model unavailable at startup; readiness is false.")
        yield
        app.state.model = None

    app = FastAPI(title="Customer Churn Inference", version="1.0.0", lifespan=lifespan)
    app.state.model = None

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc: RequestValidationError):
        del request
        errors = [
            {
                "loc": [
                    part
                    if part in {"body", *FEATURES} or isinstance(part, int)
                    else "unknown_field"
                    for part in error["loc"]
                ],
                "type": error["type"],
                "msg": "Invalid or missing request field.",
            }
            for error in exc.errors()
        ]
        return JSONResponse(status_code=422, content={"detail": errors})

    def require_model() -> LoadedModel:
        if app.state.model is None:
            raise HTTPException(status_code=503, detail="Model is not ready.")
        return app.state.model

    @app.get("/health", response_model=Health)
    def health() -> Health:
        return Health()

    @app.get("/ready", response_model=Ready, responses={503: {"description": "Model unavailable"}})
    def ready() -> Ready:
        require_model()
        return Ready()

    @app.get("/model-info", response_model=ModelInfo)
    def model_info() -> ModelInfo:
        model = require_model()
        return ModelInfo(
            model_name=model.model_name,
            model_version=model.model_version,
            artifact_sha256=model.artifact_sha256,
            threshold=model.threshold,
            input_schema=Customer.model_json_schema(),
        )

    @app.post(
        "/predict",
        response_model=Prediction,
        responses={
            503: {"description": "Model unavailable"},
            500: {"description": "Inference failed"},
        },
    )
    def predict(customer: Annotated[Customer, "Validated single customer"]) -> Prediction:
        model = require_model()
        try:
            return model.predict(customer)
        except PredictionFailed:
            LOGGER.error("Model prediction failed; readiness is false.")
            app.state.model = None
            raise HTTPException(
                status_code=500, detail="Prediction could not be completed."
            ) from None

    return app


app = create_app()
