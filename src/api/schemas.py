"""Pydantic v2 models for the manifest schema and the API request/response contract.

The manifest models are also the *validators*: a malformed manifest fails here with a
message naming the offending field, rather than surfacing as a confusing runtime error
during a prediction.
"""

from __future__ import annotations

import re
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SEMVER = re.compile(r"^\d+\.\d+\.\d+$")
MODEL_ID = re.compile(r"^[a-z0-9][a-z0-9_-]*$")

MAX_BATCH_ROWS = 10_000


# --------------------------------------------------------------------- manifest


class InputSpec(BaseModel):
    """One input feature, with the training-domain bounds the API enforces."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    unit: str
    min: float
    max: float
    description: str | None = None

    @model_validator(mode="after")
    def _bounds_ordered(self) -> InputSpec:
        if not self.min < self.max:
            raise ValueError(
                f"input {self.name!r}: min ({self.min}) must be strictly less than max ({self.max})"
            )
        return self

    @property
    def midpoint(self) -> float:
        return (self.min + self.max) / 2.0


class OutputSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    unit: str
    description: str | None = None


class Metrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    r2: float
    rmse: float


class Manifest(BaseModel):
    """The full model manifest. See docs/model_manifest_schema.md."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    model_id: str
    version: str
    project: str
    description: str | None = None
    autoemulate_version: str
    emulator_model: str
    training_date: str
    dataset_hash: str
    n_train: int = Field(ge=1)
    n_test: int = Field(ge=1)
    inputs: list[InputSpec] = Field(min_length=1)
    output: OutputSpec
    metrics: Metrics
    artifact: str = "emulator"

    @field_validator("model_id")
    @classmethod
    def _check_id(cls, v: str) -> str:
        if not MODEL_ID.match(v):
            raise ValueError(f"model_id {v!r} must match {MODEL_ID.pattern}")
        return v

    @field_validator("version")
    @classmethod
    def _check_version(cls, v: str) -> str:
        if not SEMVER.match(v):
            raise ValueError(f"version {v!r} must be semantic (MAJOR.MINOR.PATCH)")
        return v

    @model_validator(mode="after")
    def _unique_input_names(self) -> Manifest:
        names = [i.name for i in self.inputs]
        dupes = {n for n in names if names.count(n) > 1}
        if dupes:
            raise ValueError(f"duplicate input names in manifest: {sorted(dupes)}")
        return self

    @property
    def version_tuple(self) -> tuple[int, int, int]:
        a, b, c = self.version.split(".")
        return int(a), int(b), int(c)

    @property
    def input_names(self) -> list[str]:
        return [i.name for i in self.inputs]


# ------------------------------------------------------------------ API schemas


class PredictRequest(BaseModel):
    """A batch of input rows, each a mapping of input name to value."""

    model_config = ConfigDict(extra="forbid")

    inputs: Annotated[list[dict[str, float]], Field(min_length=1, max_length=MAX_BATCH_ROWS)]


class Prediction(BaseModel):
    mean: float
    std: float


class PredictResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_id: str
    version: str
    output_name: str
    output_unit: str
    n_rows: int
    predictions: list[Prediction]


class ModelSummary(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_id: str
    version: str
    project: str
    description: str | None
    output_name: str
    output_unit: str
    n_inputs: int
    r2: float
    rmse: float
    available_versions: list[str]


class HealthResponse(BaseModel):
    status: str
    n_models_loaded: int
    registry_path: str
    autoemulate_version: str


class ErrorDetail(BaseModel):
    """Structured 422 body naming the offending parameter and its valid range."""

    error: str
    parameter: str | None = None
    row: int | None = None
    value: Any | None = None
    valid_range: list[float] | None = None
    message: str
