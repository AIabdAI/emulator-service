"""File-based model registry: discovery, validation, loading and prediction.

The registry is a directory tree (``registry/<model_id>/<version>/``) rather than a
database. Every entry is validated *at startup* — manifest schema, artifact presence,
optional checksum, and a probe prediction — so that a broken model is discovered on
deploy rather than on a user's request.

Nothing here imports a simulator. Loading an AutoEmulate emulator needs only
``AutoEmulate.load_model`` (a staticmethod) and the emulator classes themselves.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import torch
from autoemulate import AutoEmulate

from .manifest import (
    MANIFEST_FILENAME,
    Manifest,
    RegistryError,
    load_manifest,
    sha256_file,
)

logger = logging.getLogger(__name__)

DEFAULT_REGISTRY_PATH = Path(
    os.environ.get("REGISTRY_PATH", Path(__file__).resolve().parents[2] / "registry")
)


def installed_autoemulate_version() -> str:
    from importlib.metadata import version

    return version("autoemulate")


def _version_sort_key(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.split("."))


@dataclass
class LoadedModel:
    """A validated manifest paired with its deserialized, ready-to-serve emulator."""

    manifest: Manifest
    emulator: object
    directory: Path
    dtype: torch.dtype = torch.float64

    @property
    def model_id(self) -> str:
        return self.manifest.model_id

    @property
    def version(self) -> str:
        return self.manifest.version

    @property
    def key(self) -> tuple[str, str]:
        return (self.model_id, self.version)

    def to_tensor(self, rows: list[list[float]]) -> torch.Tensor:
        return torch.tensor(rows, dtype=self.dtype)

    def predict(self, rows: list[list[float]]) -> tuple[list[float], list[float] | None]:
        """Predict a batch, returning per-row mean and standard deviation.

        Returns ``(means, stds)`` with ``stds`` set to ``None`` for emulators that do
        not support uncertainty quantification — the API surfaces that as an explicit
        null rather than inventing a zero.
        """
        x = self.to_tensor(rows)
        with torch.inference_mode():
            mean, variance = self.emulator.predict_mean_and_variance(x)
        means = mean.reshape(-1).to(torch.float64).tolist()
        if variance is None:
            return means, None
        # Variance can dip marginally below zero through floating-point noise in the
        # GP posterior; clamp before the square root rather than emitting NaN.
        stds = variance.reshape(-1).clamp_min(0.0).sqrt().to(torch.float64).tolist()
        return means, stds


@dataclass
class RegistryLoadFailure:
    """A registry entry that could not be loaded, kept for reporting on /health."""

    path: str
    error: str


@dataclass
class ModelRegistry:
    """In-memory view of the on-disk registry, built once at startup."""

    root: Path
    models: dict[tuple[str, str], LoadedModel] = field(default_factory=dict)
    failures: list[RegistryLoadFailure] = field(default_factory=list)

    # -- lookup ---------------------------------------------------------------

    @property
    def model_ids(self) -> list[str]:
        return sorted({model_id for model_id, _ in self.models})

    def versions(self, model_id: str) -> list[str]:
        found = [v for mid, v in self.models if mid == model_id]
        return sorted(found, key=_version_sort_key)

    def latest_version(self, model_id: str) -> str | None:
        versions = self.versions(model_id)
        return versions[-1] if versions else None

    def get(self, model_id: str, version: str | None = None) -> LoadedModel | None:
        """Resolve a model by id, defaulting to its highest semantic version."""
        if version is None:
            version = self.latest_version(model_id)
            if version is None:
                return None
        return self.models.get((model_id, version))

    def all_models(self) -> list[LoadedModel]:
        return [
            self.models[key]
            for key in sorted(self.models, key=lambda k: (k[0], _version_sort_key(k[1])))
        ]

    def __len__(self) -> int:
        return len(self.models)


def _discover_version_dirs(root: Path) -> list[Path]:
    """Find every `<root>/<model_id>/<version>/` directory holding a manifest."""
    if not root.is_dir():
        raise RegistryError(root, "registry directory does not exist")
    found: list[Path] = []
    for model_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if model_dir.name.startswith(".") or model_dir.name == "__pycache__":
            continue
        for version_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
            if version_dir.name.startswith("."):
                continue
            if (version_dir / MANIFEST_FILENAME).is_file():
                found.append(version_dir)
            else:
                logger.warning("skipping %s: no %s", version_dir, MANIFEST_FILENAME)
    return found


def _probe(manifest: Manifest, emulator: object, version_dir: Path) -> torch.dtype:
    """Run one midpoint prediction to prove the artifact matches its manifest.

    Also settles the tensor dtype the emulator wants: AutoEmulate defaults to float64,
    but a model whose weights are float32 raises rather than silently upcasting.
    """
    midpoint = [[p.midpoint for p in manifest.inputs]]
    last_error: Exception | None = None
    for dtype in (torch.float64, torch.float32):
        try:
            with torch.inference_mode():
                mean, variance = emulator.predict_mean_and_variance(
                    torch.tensor(midpoint, dtype=dtype)
                )
        except Exception as exc:
            last_error = exc
            continue

        width = int(mean.reshape(1, -1).shape[1])
        if width != 1:
            raise RegistryError(
                version_dir,
                f"artifact predicts {width} outputs but manifest v1 describes a single "
                f"output ({manifest.output.name!r})",
            )
        supports_uq = variance is not None
        if supports_uq != manifest.artifact.supports_uq:
            raise RegistryError(
                version_dir,
                f"artifact.supports_uq is {manifest.artifact.supports_uq} but the "
                f"loaded {type(emulator).__name__} "
                f"{'does' if supports_uq else 'does not'} return a variance",
            )
        return dtype

    raise RegistryError(
        version_dir,
        f"probe prediction failed for a {len(manifest.inputs)}-feature input "
        f"({manifest.input_names}) — the artifact does not match its manifest: "
        f"{last_error}",
    )


def load_entry(version_dir: Path) -> LoadedModel:
    """Load and fully validate a single `registry/<id>/<version>/` directory."""
    version_dir = Path(version_dir)
    manifest = load_manifest(version_dir / MANIFEST_FILENAME)

    expected_id = version_dir.parent.name
    expected_version = version_dir.name
    if manifest.model_id != expected_id:
        raise RegistryError(
            version_dir / MANIFEST_FILENAME,
            f"model_id {manifest.model_id!r} does not match its directory "
            f"{expected_id!r}",
        )
    if manifest.version != expected_version:
        raise RegistryError(
            version_dir / MANIFEST_FILENAME,
            f"version {manifest.version!r} does not match its directory "
            f"{expected_version!r}",
        )

    artifact_path = version_dir / manifest.artifact.filename
    if not artifact_path.is_file():
        raise RegistryError(
            version_dir,
            f"artifact {manifest.artifact.filename!r} declared in the manifest is "
            f"missing",
        )
    if manifest.artifact.sha256 is not None:
        actual = sha256_file(artifact_path)
        if actual != manifest.artifact.sha256:
            raise RegistryError(
                artifact_path,
                f"checksum mismatch: manifest declares {manifest.artifact.sha256[:12]}… "
                f"but the file hashes to {actual[:12]}…",
            )

    runtime_version = installed_autoemulate_version()
    if manifest.autoemulate_version != runtime_version:
        logger.warning(
            "%s was serialized with autoemulate %s but this runtime has %s; "
            "unpickling may fail or behave differently",
            manifest.ref,
            manifest.autoemulate_version,
            runtime_version,
        )

    try:
        emulator = AutoEmulate.load_model(artifact_path)
    except Exception as exc:
        raise RegistryError(
            artifact_path,
            f"failed to deserialize the emulator (serialized with autoemulate "
            f"{manifest.autoemulate_version}, runtime has {runtime_version}): {exc}",
        ) from exc

    eval_fn = getattr(emulator, "eval", None)
    if callable(eval_fn):
        eval_fn()

    actual_class = type(emulator).__name__
    if actual_class != manifest.artifact.emulator_class:
        logger.warning(
            "%s: manifest declares emulator_class %r but the artifact is a %r",
            manifest.ref,
            manifest.artifact.emulator_class,
            actual_class,
        )

    dtype = _probe(manifest, emulator, version_dir)
    return LoadedModel(
        manifest=manifest, emulator=emulator, directory=version_dir, dtype=dtype
    )


def load_registry(root: Path | str | None = None, strict: bool = True) -> ModelRegistry:
    """Build the in-memory registry from disk.

    Parameters
    ----------
    root
        Registry directory. Defaults to ``$REGISTRY_PATH`` or ``./registry``.
    strict
        When True (the default) any invalid entry raises ``RegistryError`` and the
        service refuses to start — a half-loaded registry must not be able to pass for
        a healthy one. When False, failures are collected and reported on ``/health``.
    """
    root = Path(root) if root is not None else DEFAULT_REGISTRY_PATH
    registry = ModelRegistry(root=root)

    for version_dir in _discover_version_dirs(root):
        try:
            loaded = load_entry(version_dir)
        except RegistryError as exc:
            if strict:
                raise
            logger.error("registry entry failed to load: %s", exc)
            registry.failures.append(
                RegistryLoadFailure(path=str(version_dir), error=str(exc))
            )
            continue
        if loaded.key in registry.models:
            raise RegistryError(version_dir, f"duplicate model {loaded.manifest.ref}")
        registry.models[loaded.key] = loaded
        logger.info(
            "loaded %s (%s, %d inputs, uq=%s)",
            loaded.manifest.ref,
            type(loaded.emulator).__name__,
            len(loaded.manifest.inputs),
            loaded.manifest.artifact.supports_uq,
        )

    return registry
