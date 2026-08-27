"""Emulator-as-a-Service: FastAPI application.

Serves versioned AutoEmulate emulators with uncertainty. The registry is loaded and
fully validated at startup, so by the time the first request arrives every model has
already been deserialized and probed.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Query, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .logging_conf import configure_logging, request_id_var
from .manifest import RegistryError
from .registry_loader import (
    DEFAULT_REGISTRY_PATH,
    LoadedModel,
    ModelRegistry,
    installed_autoemulate_version,
    load_registry,
)
from .schemas import (
    ErrorResponse,
    HealthResponse,
    InputValidationError,
    ModelDetail,
    ModelListResponse,
    ModelSummary,
    Prediction,
    PredictRequest,
    PredictResponse,
    build_row_model,
    example_request,
    validate_batch,
)

logger = logging.getLogger("emulator_service.api")

LATENCY_HEADER = "X-Prediction-Latency-Ms"
REQUEST_ID_HEADER = "X-Request-ID"

DESCRIPTION = """
Serves versioned scientific emulators trained with
[AutoEmulate](https://github.com/alan-turing-institute/autoemulate), returning a mean
**and a standard deviation** for every prediction.

Input bounds come from each model's `manifest.json`. A row outside the emulator's
training domain is rejected with `422` — an emulator queried outside the region it was
fitted on is not merely imprecise, it is confidently wrong.
""".strip()


def _row_models(registry: ModelRegistry) -> dict[tuple[str, str], Any]:
    """Compile one bounds-enforcing Pydantic model per registry entry, once."""
    return {model.key: build_row_model(model.manifest) for model in registry.all_models()}


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    registry_path = os.environ.get("REGISTRY_PATH", str(DEFAULT_REGISTRY_PATH))
    strict = os.environ.get("REGISTRY_STRICT", "true").lower() not in {"0", "false", "no"}

    started = time.perf_counter()
    registry = load_registry(registry_path, strict=strict)
    app.state.registry = registry
    app.state.row_models = _row_models(registry)

    logger.info(
        "registry loaded",
        extra={
            "event": "startup",
            "registry_path": str(registry.root),
            "models_loaded": len(registry),
            "failures": len(registry.failures),
            "strict": strict,
            "load_ms": round((time.perf_counter() - started) * 1000, 1),
            "autoemulate_version": installed_autoemulate_version(),
        },
    )
    yield
    logger.info("shutting down", extra={"event": "shutdown"})


app = FastAPI(
    title="emulator-service",
    description=DESCRIPTION,
    version="1.0.0",
    lifespan=lifespan,
)


# ------------------------------------------------------------------- middleware


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    """Assign a request id, time the request, and log one structured line for it."""
    request_id = request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex[:16]
    token = request_id_var.set(request_id)
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        logger.exception(
            "request failed",
            extra={
                "event": "request",
                "method": request.method,
                "path": request.url.path,
                "status_code": 500,
                "latency_ms": latency_ms,
            },
        )
        request_id_var.reset(token)
        raise

    latency_ms = round((time.perf_counter() - started) * 1000, 2)
    response.headers[REQUEST_ID_HEADER] = request_id
    response.headers.setdefault(LATENCY_HEADER, f"{latency_ms:.2f}")

    record: dict[str, Any] = {
        "event": "request",
        "method": request.method,
        "path": request.url.path,
        "status_code": response.status_code,
        "latency_ms": latency_ms,
    }
    for key in ("model_id", "model_version", "batch_size"):
        value = getattr(request.state, key, None)
        if value is not None:
            record[key] = value

    logger.info("request", extra=record)
    request_id_var.reset(token)
    return response


# --------------------------------------------------------------- error handlers


def _error(status: int, body: ErrorResponse) -> JSONResponse:
    return JSONResponse(
        status_code=status, content=body.model_dump(mode="json", exclude_none=True)
    )


class ModelNotFound(Exception):
    def __init__(self, detail: str):
        self.detail = detail
        super().__init__(detail)


@app.exception_handler(ModelNotFound)
async def _handle_model_not_found(request: Request, exc: ModelNotFound):
    return _error(404, ErrorResponse(error="model_not_found", detail=exc.detail))


@app.exception_handler(InputValidationError)
async def _handle_input_validation(request: Request, exc: InputValidationError):
    """Reject out-of-domain / malformed batches with 422 and a precise explanation."""
    logger.warning(
        "rejected batch",
        extra={
            "event": "input_rejected",
            "model_id": exc.manifest.model_id,
            "model_version": exc.manifest.version,
            "n_violations": len(exc.violations),
            "first_violation": exc.violations[0].reason,
        },
    )
    return _error(
        422,
        ErrorResponse(
            error="input_out_of_contract",
            detail=exc.detail,
            model_id=exc.manifest.model_id,
            version=exc.manifest.version,
            violations=exc.violations,
        ),
    )


@app.exception_handler(RequestValidationError)
async def _handle_request_validation(request: Request, exc: RequestValidationError):
    """Body-shape errors (missing `rows`, oversized batch, non-JSON)."""
    errors = exc.errors()
    detail = "; ".join(
        f"{'.'.join(str(part) for part in err['loc'][1:]) or 'body'}: {err['msg']}"
        for err in errors
    )
    return _error(422, ErrorResponse(error="invalid_request", detail=detail))


@app.exception_handler(RegistryError)
async def _handle_registry_error(request: Request, exc: RegistryError):
    logger.error("registry error", extra={"event": "registry_error", "error": str(exc)})
    return _error(500, ErrorResponse(error="registry_error", detail=str(exc)))


# ------------------------------------------------------------------- resolution


def _registry(request: Request) -> ModelRegistry:
    return request.app.state.registry


def _resolve(request: Request, model_id: str, version: str | None) -> LoadedModel:
    registry = _registry(request)
    model = registry.get(model_id, version)
    if model is not None:
        return model

    known = registry.versions(model_id)
    if known:
        raise ModelNotFound(
            f"model {model_id!r} has no version {version!r}. "
            f"Available versions: {', '.join(known)}."
        )
    available = registry.model_ids
    suffix = f" Available models: {', '.join(available)}." if available else ""
    raise ModelNotFound(f"no model with id {model_id!r} in the registry.{suffix}")


def _summary(registry: ModelRegistry, model: LoadedModel) -> ModelSummary:
    manifest = model.manifest
    return ModelSummary(
        model_id=manifest.model_id,
        version=manifest.version,
        latest=registry.latest_version(manifest.model_id) == manifest.version,
        project=manifest.project,
        description=manifest.description,
        output=manifest.output.name,
        output_unit=manifest.output.unit,
        n_inputs=len(manifest.inputs),
        supports_uq=manifest.artifact.supports_uq,
        stand_in=manifest.stand_in,
        training_date=manifest.training_date,
        r2=manifest.metrics.r2,
        rmse=manifest.metrics.rmse,
    )


# ---------------------------------------------------------------------- routes


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health(request: Request) -> HealthResponse:
    """Liveness plus what the process actually has loaded."""
    registry = _registry(request)
    return HealthResponse(
        status="degraded" if registry.failures else "ok",
        models_loaded=len(registry),
        model_ids=registry.model_ids,
        autoemulate_version=installed_autoemulate_version(),
        registry_path=str(registry.root),
        failures=[{"path": f.path, "error": f.error} for f in registry.failures],
    )


@app.get("/models", response_model=ModelListResponse, tags=["models"])
def list_models(
    request: Request,
    latest_only: bool = Query(
        False, description="Return only the highest version of each model."
    ),
) -> ModelListResponse:
    """Every model version in the registry, with its headline metadata."""
    registry = _registry(request)
    models = registry.all_models()
    summaries = [_summary(registry, model) for model in models]
    if latest_only:
        summaries = [summary for summary in summaries if summary.latest]
    return ModelListResponse(count=len(summaries), models=summaries)


@app.get(
    "/models/{model_id}",
    response_model=ModelDetail,
    responses={404: {"model": ErrorResponse}},
    tags=["models"],
)
def get_model(
    request: Request,
    model_id: str,
    version: str | None = Query(
        None, description="Semantic version. Defaults to the highest available."
    ),
) -> ModelDetail:
    """The full manifest for one model: input bounds, units, provenance, metrics."""
    registry = _registry(request)
    model = _resolve(request, model_id, version)
    manifest = model.manifest
    return ModelDetail(
        model_id=manifest.model_id,
        version=manifest.version,
        available_versions=registry.versions(model_id),
        latest=registry.latest_version(model_id) == manifest.version,
        manifest=manifest.model_dump(mode="json", exclude_none=True),
        emulator_class=type(model.emulator).__name__,
        example_request=example_request(manifest),
    )


@app.post(
    "/models/{model_id}/predict",
    response_model=PredictResponse,
    responses={404: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
    tags=["predict"],
)
def predict(
    request: Request,
    response: Response,
    model_id: str,
    payload: PredictRequest,
    version: str | None = Query(
        None, description="Semantic version. Defaults to the highest available."
    ),
) -> PredictResponse:
    """Predict a batch, returning a mean and standard deviation per row.

    Every row is validated against the model's manifest bounds first. If any row is
    out of the training domain the whole batch is rejected with `422` — a partially
    served batch would leave the caller unsure which predictions to trust.
    """
    model = _resolve(request, model_id, version)
    manifest = model.manifest

    request.state.model_id = manifest.model_id
    request.state.model_version = manifest.version
    request.state.batch_size = len(payload.rows)

    row_model = request.app.state.row_models[model.key]
    matrix = validate_batch(manifest, row_model, payload.rows)

    started = time.perf_counter()
    means, stds = model.predict(matrix)
    inference_ms = (time.perf_counter() - started) * 1000
    response.headers[LATENCY_HEADER] = f"{inference_ms:.2f}"

    logger.info(
        "prediction served",
        extra={
            "event": "prediction",
            "model_id": manifest.model_id,
            "model_version": manifest.version,
            "batch_size": len(matrix),
            "inference_ms": round(inference_ms, 2),
        },
    )

    predictions = [
        Prediction(mean=mean, std=None if stds is None else stds[index])
        for index, mean in enumerate(means)
    ]
    return PredictResponse(
        model_id=manifest.model_id,
        version=manifest.version,
        output=manifest.output.name,
        output_unit=manifest.output.unit,
        n_rows=len(predictions),
        supports_uq=manifest.artifact.supports_uq,
        predictions=predictions,
    )
