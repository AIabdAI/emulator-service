"""File-based model registry: discovery, validation, loading, and bounds enforcement.

A file-based registry is deliberate (see the README's design-decisions section): the
whole registry is a directory of manifests, so it diffs in code review, versions in
git, and needs no database to stand up locally.

Nothing here imports a simulator. The service loads emulators with AutoEmulate alone,
which is what keeps `pybamm`, `openseespy` and `pvlib` out of the serving image.
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .schemas import Manifest

log = logging.getLogger("emulator-service.registry")

DEFAULT_REGISTRY = Path(__file__).resolve().parents[2] / "registry"


class BoundsError(ValueError):
    """Raised when an input row falls outside the manifest's declared training domain."""

    def __init__(self, parameter: str, value: float, low: float, high: float, row: int):
        self.parameter = parameter
        self.value = value
        self.low = low
        self.high = high
        self.row = row
        reason = (
            "is not a finite number"
            if not math.isfinite(value)
            else f"is outside its valid range [{low}, {high}]"
        )
        super().__init__(
            f"row {row}: parameter {parameter!r} = {value} {reason}. "
            "The emulator was never trained there and would extrapolate."
        )


class MissingInputError(ValueError):
    def __init__(self, missing: list[str], row: int):
        self.missing = missing
        self.row = row
        super().__init__(f"row {row}: missing required input(s): {', '.join(sorted(missing))}")


class UnknownInputError(ValueError):
    def __init__(self, unknown: list[str], row: int, expected: list[str]):
        self.unknown = unknown
        self.row = row
        super().__init__(
            f"row {row}: unknown input(s): {', '.join(sorted(unknown))}. "
            f"Expected exactly: {', '.join(expected)}"
        )


@dataclass
class LoadedModel:
    manifest: Manifest
    emulator: object
    path: Path

    def validate_batch(self, rows: list[dict[str, float]]) -> np.ndarray:
        """Validate a batch against the manifest and return the ordered feature matrix.

        Raises before any value reaches the emulator. Feature order comes from the
        manifest, not from the caller's dict ordering.
        """
        expected = self.manifest.input_names
        matrix = np.empty((len(rows), len(expected)), dtype=np.float64)

        for r, row in enumerate(rows):
            missing = [n for n in expected if n not in row]
            if missing:
                raise MissingInputError(missing, r)
            unknown = [n for n in row if n not in expected]
            if unknown:
                raise UnknownInputError(unknown, r, expected)

            for c, spec in enumerate(self.manifest.inputs):
                value = float(row[spec.name])
                if not np.isfinite(value):
                    raise BoundsError(spec.name, value, spec.min, spec.max, r)
                if value < spec.min or value > spec.max:
                    raise BoundsError(spec.name, value, spec.min, spec.max, r)
                matrix[r, c] = value
        return matrix

    def predict(self, matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return per-row (mean, std).

        AutoEmulate emulators return a ``torch.distributions.Distribution`` with
        ``.mean`` / ``.variance`` when probabilistic, and a plain tensor otherwise.
        A deterministic emulator reports zero uncertainty rather than a fabricated one.

        The type check must be ``isinstance(out, Distribution)``, **not**
        ``hasattr(out, "mean")``: ``torch.Tensor`` also has a ``.mean`` attribute -- it
        is a bound method -- so the duck-typed check silently misclassifies every
        deterministic emulator and then fails converting a method to a float.
        """
        import torch
        from torch.distributions import Distribution

        tensor = torch.tensor(matrix, dtype=torch.float32)
        with torch.no_grad():
            out = self.emulator.predict(tensor)
        if isinstance(out, Distribution):
            mean = np.asarray(out.mean, dtype=float).reshape(len(matrix), -1)[:, 0]
            std = np.sqrt(
                np.clip(np.asarray(out.variance, dtype=float), 0.0, None)
            ).reshape(len(matrix), -1)[:, 0]
        else:
            mean = np.asarray(out, dtype=float).reshape(len(matrix), -1)[:, 0]
            std = np.zeros_like(mean)
        return mean, std


@dataclass
class Registry:
    """All loadable model versions, keyed by ``model_id`` then ``version``."""

    root: Path
    models: dict[str, dict[str, LoadedModel]] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------ lookup

    def model_ids(self) -> list[str]:
        return sorted(self.models)

    def versions(self, model_id: str) -> list[str]:
        if model_id not in self.models:
            return []
        return [
            m.manifest.version
            for m in sorted(
                self.models[model_id].values(), key=lambda m: m.manifest.version_tuple
            )
        ]

    def get(self, model_id: str, version: str | None = None) -> LoadedModel | None:
        by_version = self.models.get(model_id)
        if not by_version:
            return None
        if version is not None:
            return by_version.get(version)
        # Default to the highest semantic version.
        return max(by_version.values(), key=lambda m: m.manifest.version_tuple)

    @property
    def n_loaded(self) -> int:
        return sum(len(v) for v in self.models.values())


def resolve_artifact(version_dir: Path, artifact_name: str) -> Path | None:
    """Locate a serialised emulator inside a version directory.

    ``AutoEmulate.save(result, path)`` does not write ``path``: it writes
    ``path.joblib`` alongside ``path_metadata.csv``, and returns the extensionless
    stem, which ``load_model`` also accepts. The manifest therefore names the stem, and
    existence has to be checked against the real file on disk.
    """
    stem = version_dir / artifact_name
    if stem.is_file():
        return stem
    joblib = stem.with_suffix(".joblib")
    if joblib.is_file():
        return stem  # hand load_model the stem it expects
    return None


def _probe(model: LoadedModel) -> None:
    """Predict at the midpoint of the declared bounds; a model that cannot is unusable."""
    midpoint = np.array([[s.midpoint for s in model.manifest.inputs]], dtype=np.float64)
    mean, std = model.predict(midpoint)
    if not np.isfinite(mean).all():
        raise ValueError("probe prediction returned a non-finite mean")
    if (std < 0).any():
        raise ValueError("probe prediction returned a negative standard deviation")


def load_registry(root: str | Path | None = None, probe: bool = True) -> Registry:
    """Discover and validate every ``<model_id>/<version>/manifest.json`` under ``root``.

    A model that fails validation is skipped with an actionable message; the rest of the
    registry still loads, so one bad manifest cannot take the service down.
    """
    root = Path(root or os.environ.get("REGISTRY_PATH") or DEFAULT_REGISTRY).resolve()
    registry = Registry(root=root)

    if not root.is_dir():
        registry.errors.append(f"Registry path does not exist: {root}")
        log.error("Registry path does not exist: %s", root)
        return registry

    from autoemulate import AutoEmulate

    for manifest_path in sorted(root.glob("*/*/manifest.json")):
        version_dir = manifest_path.parent
        try:
            raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            registry.errors.append(f"{manifest_path}: invalid JSON ({exc})")
            log.error("Skipping %s: invalid JSON: %s", manifest_path, exc)
            continue

        try:
            manifest = Manifest.model_validate(raw)
        except Exception as exc:  # noqa: BLE001 - report the field-level reason verbatim
            registry.errors.append(f"{manifest_path}: schema validation failed ({exc})")
            log.error("Skipping %s: manifest failed validation: %s", manifest_path, exc)
            continue

        if manifest.version != version_dir.name:
            msg = (
                f"{manifest_path}: manifest version {manifest.version!r} does not match "
                f"its directory name {version_dir.name!r}"
            )
            registry.errors.append(msg)
            log.error("Skipping %s", msg)
            continue

        if manifest.model_id != version_dir.parent.name:
            msg = (
                f"{manifest_path}: model_id {manifest.model_id!r} does not match "
                f"its directory name {version_dir.parent.name!r}"
            )
            registry.errors.append(msg)
            log.error("Skipping %s", msg)
            continue

        artifact = resolve_artifact(version_dir, manifest.artifact)
        if artifact is None:
            registry.errors.append(f"{manifest_path}: artifact {manifest.artifact!r} not found")
            log.error("Skipping %s: artifact %r not found", manifest_path, manifest.artifact)
            continue

        try:
            emulator = AutoEmulate.load_model(artifact)
        except Exception as exc:  # noqa: BLE001
            registry.errors.append(f"{artifact}: failed to deserialise ({exc})")
            log.error("Skipping %s: failed to deserialise emulator: %s", artifact, exc)
            continue

        model = LoadedModel(manifest=manifest, emulator=emulator, path=version_dir)
        if probe:
            try:
                _probe(model)
            except Exception as exc:  # noqa: BLE001
                registry.errors.append(f"{artifact}: probe prediction failed ({exc})")
                log.error("Skipping %s: probe prediction failed: %s", artifact, exc)
                continue

        registry.models.setdefault(manifest.model_id, {})[manifest.version] = model
        log.info(
            "Loaded %s v%s (%s, output=%s, R2=%.4f)",
            manifest.model_id, manifest.version, manifest.emulator_model,
            manifest.output.name, manifest.metrics.r2,
        )

    log.info("Registry ready: %d model version(s), %d error(s)", registry.n_loaded,
             len(registry.errors))
    return registry
