"""Request/response models, and the manifest-driven input validation.

The important piece here is :func:`build_row_model`: for each registered model the
service compiles a Pydantic model whose fields carry the *manifest's* bounds. Requests
are therefore checked against the emulator's actual training domain, and an
out-of-domain row is rejected before it can reach the emulator.
"""

from __future__ import annotations

import math
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    create_model,
)

from .manifest import InputParameter, Manifest

MAX_BATCH_ROWS = 1000


# --------------------------------------------------------------------------- errors


class BoundsViolation(BaseModel):
    """One rejected value, with everything a caller needs to fix the request."""

    row: int
    parameter: str
    value: Any = None
    reason: str
    min: float | None = None
    max: float | None = None
    unit: str | None = None


class InputValidationError(Exception):
    """Raised when a prediction batch does not conform to the model's manifest."""

    def __init__(self, manifest: Manifest, violations: list[BoundsViolation]):
        self.manifest = manifest
        self.violations = violations
        super().__init__(self.detail)

    @property
    def detail(self) -> str:
        head = self.violations[0]
        extra = (
            f" (and {len(self.violations) - 1} more)" if len(self.violations) > 1 else ""
        )
        return f"row {head.row}: {head.reason}{extra}"


class ErrorResponse(BaseModel):
    """Uniform error body for 404/413/422/500."""

    error: str
    detail: str
    model_id: str | None = None
    version: str | None = None
    violations: list[BoundsViolation] | None = None


# ------------------------------------------------------------------- manifest views


class InputParameterView(BaseModel):
    name: str
    unit: str
    min: float
    max: float
    description: str | None = None


class ModelSummary(BaseModel):
    """What `GET /models` returns per entry."""

    model_config = ConfigDict(protected_namespaces=())

    model_id: str
    version: str
    latest: bool
    project: str
    description: str | None = None
    output: str
    output_unit: str
    n_inputs: int
    supports_uq: bool
    stand_in: bool
    training_date: str
    r2: float
    rmse: float


class ModelDetail(BaseModel):
    """What `GET /models/{id}` returns: the full manifest plus runtime facts."""

    model_config = ConfigDict(protected_namespaces=())

    model_id: str
    version: str
    available_versions: list[str]
    latest: bool
    manifest: dict[str, Any]
    emulator_class: str
    example_request: dict[str, Any]


class ModelListResponse(BaseModel):
    count: int
    models: list[ModelSummary]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    models_loaded: int
    model_ids: list[str]
    autoemulate_version: str
    registry_path: str
    failures: list[dict[str, str]] = Field(default_factory=list)


# ------------------------------------------------------------------------ predict


class PredictRequest(BaseModel):
    """A batch of input rows, keyed by the manifest's input parameter names."""

    model_config = ConfigDict(extra="forbid")

    rows: list[dict[str, Any]] = Field(
        min_length=1,
        max_length=MAX_BATCH_ROWS,
        description="One object per prediction, keyed by input parameter name.",
    )


class Prediction(BaseModel):
    """Per-row prediction. `std` is null for emulators without UQ support."""

    mean: float
    std: float | None = None


class PredictResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_id: str
    version: str
    output: str
    output_unit: str
    n_rows: int
    supports_uq: bool
    predictions: list[Prediction]


# --------------------------------------------------------- manifest-driven checking


def build_row_model(manifest: Manifest) -> type[BaseModel]:
    """Compile a Pydantic model enforcing this manifest's types *and* bounds.

    Every input becomes a required float field constrained to ``[min, max]``; unknown
    keys are forbidden so a typo in a parameter name is an error rather than a silently
    ignored field.
    """
    fields: dict[str, Any] = {}
    for parameter in manifest.inputs:
        annotation = Annotated[
            float,
            Field(
                ge=parameter.min,
                le=parameter.max,
                description=(
                    f"{parameter.description or parameter.name} "
                    f"[{parameter.unit}], training domain "
                    f"[{parameter.min}, {parameter.max}]"
                ),
            ),
        ]
        fields[parameter.name] = (annotation, ...)

    return create_model(
        f"{manifest.model_id.replace('-', '_')}_row",
        __config__=ConfigDict(extra="forbid"),
        **fields,
    )


def _reason_for(
    error: dict[str, Any], parameter: InputParameter | None, manifest: Manifest
) -> tuple[str, str]:
    """Turn a raw Pydantic error into (parameter_name, human reason)."""
    loc = error.get("loc") or ()
    name = str(loc[0]) if loc else "<row>"
    kind = error.get("type", "")

    if kind == "missing":
        return name, (
            f"{name!r} is required by {manifest.ref} but was not supplied "
            f"(expected inputs: {', '.join(manifest.input_names)})"
        )
    if kind == "extra_forbidden":
        return name, (
            f"{name!r} is not an input of {manifest.ref} "
            f"(expected inputs: {', '.join(manifest.input_names)})"
        )
    if kind in {"greater_than_equal", "less_than_equal"} and parameter is not None:
        return name, (
            f"{name} = {error.get('input')!r} is outside the training domain of "
            f"{manifest.ref}: valid range is [{parameter.min}, {parameter.max}] "
            f"{parameter.unit}. The emulator was not fitted there and would be "
            f"silently wrong."
        )
    if kind.startswith("float") or kind.startswith("int") or "parsing" in kind:
        unit = f" [{parameter.unit}]" if parameter else ""
        return name, (
            f"{name} = {error.get('input')!r} is not a number{unit}: {error.get('msg')}"
        )
    return name, f"{name}: {error.get('msg')}"


def validate_batch(
    manifest: Manifest, row_model: type[BaseModel], rows: list[dict[str, Any]]
) -> list[list[float]]:
    """Validate a batch and return it as a dense matrix in manifest column order.

    Raises
    ------
    InputValidationError
        With one :class:`BoundsViolation` per offending value. Nothing reaches the
        emulator unless every row of the batch is inside its training domain.
    """
    matrix: list[list[float]] = []
    violations: list[BoundsViolation] = []

    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            violations.append(
                BoundsViolation(
                    row=index,
                    parameter="<row>",
                    value=row,
                    reason=(
                        f"expected an object keyed by "
                        f"{', '.join(manifest.input_names)}, got "
                        f"{type(row).__name__}"
                    ),
                )
            )
            continue
        try:
            validated = row_model.model_validate(row)
        except ValidationError as exc:
            for error in exc.errors():
                loc = error.get("loc") or ()
                parameter = manifest.parameter(str(loc[0])) if loc else None
                name, reason = _reason_for(error, parameter, manifest)
                violations.append(
                    BoundsViolation(
                        row=index,
                        parameter=name,
                        value=error.get("input"),
                        reason=reason,
                        min=parameter.min if parameter else None,
                        max=parameter.max if parameter else None,
                        unit=parameter.unit if parameter else None,
                    )
                )
            continue

        values = [getattr(validated, name) for name in manifest.input_names]
        # ge/le comparisons are False for NaN, so Pydantic bounds already reject it;
        # infinities are caught by the same check. This guard documents that intent
        # and protects models whose bounds somehow admit a non-finite value.
        non_finite = [
            (name, value)
            for name, value in zip(manifest.input_names, values, strict=True)
            if not math.isfinite(value)
        ]
        if non_finite:
            for name, value in non_finite:
                parameter = manifest.parameter(name)
                violations.append(
                    BoundsViolation(
                        row=index,
                        parameter=name,
                        value=value,
                        reason=f"{name} must be a finite number, got {value}",
                        min=parameter.min if parameter else None,
                        max=parameter.max if parameter else None,
                        unit=parameter.unit if parameter else None,
                    )
                )
            continue
        matrix.append(values)

    if violations:
        raise InputValidationError(manifest, violations)
    return matrix


def example_request(manifest: Manifest) -> dict[str, Any]:
    """A ready-to-paste request body using each parameter's domain midpoint."""
    return {
        "rows": [{p.name: round(p.midpoint, 6) for p in manifest.inputs}]
    }
