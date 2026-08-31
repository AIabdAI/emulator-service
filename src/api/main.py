"""FastAPI application serving scientific emulators with calibrated uncertainty.

Design points worth naming:

* **Uncertainty is part of the contract.** ``/predict`` always returns a standard
  deviation alongside the mean. A surrogate model without its uncertainty invites the
  caller to trust a number the model itself is unsure about.
* **Out-of-domain inputs cannot reach an emulator.** Bounds come from each model's
  manifest and are enforced before the feature matrix is built, returning 422 with the
  offending parameter named and its valid range attached.
* **No simulator imports.** The service loads emulators through AutoEmulate only, so
  pybamm / openseespy / pvlib never enter the serving image.
"""

from __future__ import annotations

import logging
import math
import os
import time
from contextlib import asynccontextmanager
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as pkg_version

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

from .logging_conf import configure_logging
from .registry_loader import (
    BoundsError,
    MissingInputError,
    Registry,
    UnknownInputError,
    load_registry,
)
from .schemas import (
    ErrorDetail,
    HealthResponse,
    Manifest,
    ModelSummary,
    PredictRequest,
    PredictResponse,
)

log = logging.getLogger("emulator-service.api")

#: Populated at startup; module-level so tests can reload it against a temp registry.
STATE: dict[str, Registry] = {}


def autoemulate_version() -> str:
    try:
        return pkg_version("autoemulate")
    except PackageNotFoundError:  # pragma: no cover
        return "unknown"


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging(os.environ.get("LOG_LEVEL", "INFO"))
    registry = load_registry(os.environ.get("REGISTRY_PATH"))
    STATE["registry"] = registry
    log.info(
        "Service starting",
        extra={"n_models": registry.n_loaded, "registry_errors": len(registry.errors)},
    )
    if registry.errors:
        for err in registry.errors:
            log.warning("Registry error", extra={"detail": err})
    yield
    STATE.clear()


app = FastAPI(
    title="Emulator Service",
    description="Versioned serving of scientific emulators, with uncertainty.",
    version="0.1.0",
    lifespan=lifespan,
)


def registry() -> Registry:
    reg = STATE.get("registry")
    if reg is None:  # pragma: no cover - only reachable outside the lifespan
        reg = load_registry(os.environ.get("REGISTRY_PATH"))
        STATE["registry"] = reg
    return reg


# ------------------------------------------------------------------- middleware


@app.middleware("http")
async def log_and_time(request: Request, call_next):
    """Time every request, attach the latency header, and emit one structured log line."""
    start = time.perf_counter()
    response = await call_next(request)
    latency_ms = (time.perf_counter() - start) * 1000.0
    response.headers["X-Response-Time-ms"] = f"{latency_ms:.2f}"
    log.info(
        "request",
        extra={
            "method": request.method,
            "path": request.url.path,
            "status": response.status_code,
            "latency_ms": round(latency_ms, 2),
            "model_id": request.path_params.get("model_id"),
            "batch_size": getattr(request.state, "batch_size", None),
        },
    )
    return response


# ---------------------------------------------------------------------- routes


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    reg = registry()
    return HealthResponse(
        status="ok" if reg.n_loaded > 0 else "degraded",
        n_models_loaded=reg.n_loaded,
        registry_path=str(reg.root),
        autoemulate_version=autoemulate_version(),
    )


@app.get("/models", response_model=list[ModelSummary])
def list_models() -> list[ModelSummary]:
    reg = registry()
    out = []
    for model_id in reg.model_ids():
        model = reg.get(model_id)
        if model is None:  # pragma: no cover
            continue
        m = model.manifest
        out.append(
            ModelSummary(
                model_id=m.model_id,
                version=m.version,
                project=m.project,
                description=m.description,
                output_name=m.output.name,
                output_unit=m.output.unit,
                n_inputs=len(m.inputs),
                r2=m.metrics.r2,
                rmse=m.metrics.rmse,
                available_versions=reg.versions(model_id),
            )
        )
    return out


@app.get("/models/{model_id}", response_model=Manifest)
def get_model(model_id: str, version: str | None = None) -> Manifest | JSONResponse:
    model = registry().get(model_id, version)
    if model is None:
        return _not_found(model_id, version)
    return model.manifest


@app.post("/models/{model_id}/predict", response_model=PredictResponse)
def predict(
    model_id: str, payload: PredictRequest, request: Request, version: str | None = None
) -> PredictResponse | JSONResponse:
    model = registry().get(model_id, version)
    if model is None:
        return _not_found(model_id, version)

    request.state.batch_size = len(payload.inputs)

    try:
        matrix = model.validate_batch(payload.inputs)
    except BoundsError as exc:
        # NaN/inf are valid JSON *inputs* via Python's parser but are not encodable as
        # JSON numbers, so echo them back as strings rather than crashing the response.
        finite = math.isfinite(exc.value)
        return JSONResponse(
            status_code=422,
            content=ErrorDetail(
                error="input_out_of_bounds" if finite else "input_not_finite",
                parameter=exc.parameter,
                row=exc.row,
                value=exc.value if finite else str(exc.value),
                valid_range=[exc.low, exc.high],
                message=str(exc),
            ).model_dump(),
        )
    except MissingInputError as exc:
        return JSONResponse(
            status_code=422,
            content=ErrorDetail(
                error="missing_input", row=exc.row, message=str(exc)
            ).model_dump(),
        )
    except UnknownInputError as exc:
        return JSONResponse(
            status_code=422,
            content=ErrorDetail(
                error="unknown_input", row=exc.row, message=str(exc)
            ).model_dump(),
        )
    except (TypeError, ValueError) as exc:
        return JSONResponse(
            status_code=422,
            content=ErrorDetail(error="invalid_input", message=str(exc)).model_dump(),
        )

    mean, std = model.predict(matrix)
    m = model.manifest
    return PredictResponse(
        model_id=m.model_id,
        version=m.version,
        output_name=m.output.name,
        output_unit=m.output.unit,
        n_rows=len(matrix),
        predictions=[{"mean": float(a), "std": float(b)} for a, b in zip(mean, std, strict=True)],
    )


def _not_found(model_id: str, version: str | None) -> JSONResponse:
    reg = registry()
    known = reg.model_ids()
    if version is not None and model_id in reg.models:
        message = (
            f"Model {model_id!r} has no version {version!r}. "
            f"Available versions: {', '.join(reg.versions(model_id))}"
        )
    else:
        message = f"Unknown model {model_id!r}. Available models: {', '.join(known) or 'none'}"
    return JSONResponse(
        status_code=404,
        content=ErrorDetail(error="model_not_found", message=message).model_dump(),
    )


@app.get("/", include_in_schema=False)
def root() -> Response:
    return JSONResponse(
        {
            "service": "emulator-service",
            "docs": "/docs",
            "endpoints": ["/health", "/models", "/models/{id}", "/models/{id}/predict"],
        }
    )
