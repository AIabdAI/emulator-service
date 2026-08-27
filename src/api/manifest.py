"""Model manifest schema — the contract between training and serving.

This module is imported by *both* the API and the training pipeline, so it must not
import FastAPI, MLflow or anything simulator-shaped. See
``docs/model_manifest_schema.md`` for the prose version of this schema.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

SCHEMA_VERSION = 1
MANIFEST_FILENAME = "manifest.json"

MODEL_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]*$")
SEMVER_PATTERN = re.compile(r"^\d+\.\d+\.\d+$")


class RegistryError(Exception):
    """Raised when a registry entry is malformed, unreadable or inconsistent.

    Carries the offending path so the message always tells an operator *where* to look,
    not just that something went wrong.
    """

    def __init__(self, path: Path | str, message: str):
        self.path = Path(path)
        self.message = message
        super().__init__(f"{_display(self.path)}: {message}")


def _display(path: Path) -> str:
    """Render a path relative to the repo root when possible, for readable errors."""
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve())).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class InputParameter(_Strict):
    """One input dimension of the emulator, with its training-domain bounds."""

    name: str = Field(min_length=1)
    unit: str = Field(min_length=1)
    min: float
    max: float
    description: str | None = None

    @model_validator(mode="after")
    def _check_bounds(self) -> InputParameter:
        if not self.max > self.min:
            raise ValueError(
                f"max ({self.max}) must be greater than min ({self.min})"
            )
        return self

    @property
    def midpoint(self) -> float:
        return (self.min + self.max) / 2.0

    def contains(self, value: float) -> bool:
        return self.min <= value <= self.max


class OutputSpec(_Strict):
    """The single quantity the emulator predicts."""

    name: str = Field(min_length=1)
    unit: str = Field(min_length=1)
    description: str | None = None


class ArtifactSpec(_Strict):
    """Where the serialized emulator lives and what it should turn out to be."""

    filename: str = Field(min_length=1)
    format: str = "joblib"
    emulator_class: str = Field(min_length=1)
    supports_uq: bool
    sha256: str | None = None

    @model_validator(mode="after")
    def _check(self) -> ArtifactSpec:
        if self.format != "joblib":
            raise ValueError(f"format must be 'joblib', got {self.format!r}")
        name = self.filename
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError(
                f"filename must be a relative path inside the version directory, "
                f"got {name!r}"
            )
        if self.sha256 is not None and not re.fullmatch(r"[0-9a-f]{64}", self.sha256):
            raise ValueError("sha256 must be 64 lowercase hex characters")
        return self


class MetricsSpec(BaseModel):
    """Held-out metrics. Extra numeric metrics are allowed and passed through."""

    model_config = ConfigDict(extra="allow", frozen=True)

    r2: float
    rmse: float = Field(ge=0.0)
    n_test: int | None = Field(default=None, ge=1)


class DatasetSpec(_Strict):
    """Provenance of the training data."""

    hash: str = Field(min_length=1)
    n_train: int | None = Field(default=None, ge=1)
    path: str | None = None

    @model_validator(mode="after")
    def _check_hash(self) -> DatasetSpec:
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", self.hash):
            raise ValueError("hash must look like 'sha256:<64 hex chars>'")
        return self


class Manifest(_Strict):
    """A complete, validated `manifest.json`."""

    schema_version: int = SCHEMA_VERSION
    model_id: str
    version: str
    project: str = Field(min_length=1)
    description: str | None = None
    autoemulate_version: str = Field(min_length=1)
    training_date: str = Field(min_length=1)
    stand_in: bool = False
    artifact: ArtifactSpec
    inputs: list[InputParameter] = Field(min_length=1)
    output: OutputSpec
    metrics: MetricsSpec
    dataset: DatasetSpec

    @model_validator(mode="after")
    def _check(self) -> Manifest:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"schema_version {self.schema_version} is not supported by this "
                f"service (expected {SCHEMA_VERSION})"
            )
        if not MODEL_ID_PATTERN.fullmatch(self.model_id):
            raise ValueError(
                f"model_id {self.model_id!r} must match {MODEL_ID_PATTERN.pattern}"
            )
        if not SEMVER_PATTERN.fullmatch(self.version):
            raise ValueError(
                f"version {self.version!r} must be semantic (MAJOR.MINOR.PATCH)"
            )
        names = [p.name for p in self.inputs]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(f"duplicate input parameter names: {duplicates}")
        return self

    @property
    def input_names(self) -> list[str]:
        """Input names in tensor-column order."""
        return [p.name for p in self.inputs]

    @property
    def ref(self) -> str:
        return f"{self.model_id} v{self.version}"

    def parameter(self, name: str) -> InputParameter | None:
        for p in self.inputs:
            if p.name == name:
                return p
        return None


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Hex sha256 of a file, streamed so large datasets do not sit in memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def format_validation_error(exc: ValidationError) -> str:
    """Flatten a Pydantic error into one line that names each offending field."""
    parts = []
    for err in exc.errors():
        loc = ".".join(str(item) for item in err["loc"]) or "<root>"
        parts.append(f"{loc}: {err['msg']}")
    return "; ".join(parts)


def load_manifest(manifest_path: Path) -> Manifest:
    """Read and validate one `manifest.json`.

    Raises
    ------
    RegistryError
        If the file is missing, is not JSON, or violates the schema. The message names
        the path and the offending field.
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        raise RegistryError(manifest_path, "manifest.json not found")
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RegistryError(
            manifest_path, f"not valid JSON (line {exc.lineno}, column {exc.colno})"
        ) from exc
    if not isinstance(raw, dict):
        raise RegistryError(
            manifest_path, f"expected a JSON object, got {type(raw).__name__}"
        )
    try:
        return Manifest.model_validate(raw)
    except ValidationError as exc:
        raise RegistryError(manifest_path, format_validation_error(exc)) from exc


def write_manifest(version_dir: Path, manifest: Manifest) -> Path:
    """Write a manifest into a version directory. Refuses to overwrite."""
    version_dir = Path(version_dir)
    path = version_dir / MANIFEST_FILENAME
    if path.exists():
        raise RegistryError(path, "manifest already exists; versions are immutable")
    version_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = manifest.model_dump(mode="json", exclude_none=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path
