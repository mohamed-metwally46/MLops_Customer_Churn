# syntax=docker/dockerfile:1
FROM python:3.11.17-slim-bookworm AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /build
COPY requirements/build.txt requirements/runtime.lock /requirements/
RUN python -m pip install --only-binary=:all: -r /requirements/build.txt \
    && python -m venv /opt/venv \
    && /opt/venv/bin/pip install --require-hashes --only-binary=:all: -r /requirements/runtime.lock
COPY pyproject.toml README.md ./
# Copy only modules needed by the inspected saved inference bundle.
COPY src/churn/__init__.py src/churn/api.py src/churn/data.py src/churn/model_loader.py src/churn/preprocessing.py src/churn/schemas.py ./src/churn/
RUN python -m pip wheel --no-deps --no-build-isolation --wheel-dir /wheels . \
    && /opt/venv/bin/pip install --no-deps /wheels/churn_mlops-*.whl \
    && /opt/venv/bin/pip check

FROM python:3.11.17-slim-bookworm AS runtime
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HOME=/tmp \
    CHURN_PROJECT_ROOT=/app CHURN_MODEL_PATH=/model/pipeline.joblib \
    CHURN_MODEL_METADATA=/model/metadata.json
WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
RUN mkdir /model
USER 10001:10001
EXPOSE 8000
# Readiness, not mere process liveness, determines Docker health.
HEALTHCHECK --interval=30s --timeout=3s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=2)"]
CMD ["python", "-m", "uvicorn", "churn.api:app", "--host", "0.0.0.0", "--port", "8000"]
