"""Registry loader tests: a broken entry must fail loudly at startup, never silently."""

from __future__ import annotations

import json

import pytest

from api.manifest import RegistryError
from api.registry_loader import load_entry, load_registry
from tests.conftest import BATTERY_ID, BATTERY_VERSION, read_manifest, write_manifest


def _version_dir(registry_root):
    return registry_root / BATTERY_ID / BATTERY_VERSION


def test_loads_the_real_registry(registry_copy):
    registry = load_registry(registry_copy)
    assert len(registry) == 1
    model = registry.get(BATTERY_ID)
    assert model is not None
    assert model.version == BATTERY_VERSION
    assert model.manifest.artifact.supports_uq is True


def test_predicts_through_the_loaded_emulator(registry_copy):
    """The contract that matters: a loaded artifact yields a mean and a std."""
    registry = load_registry(registry_copy)
    model = registry.get(BATTERY_ID)
    rows = [[p.midpoint for p in model.manifest.inputs]]
    means, stds = model.predict(rows)
    assert len(means) == 1
    assert stds is not None and len(stds) == 1
    assert stds[0] >= 0.0


def test_missing_registry_directory(tmp_path):
    with pytest.raises(RegistryError, match="does not exist"):
        load_registry(tmp_path / "nope")


def test_model_id_must_match_its_directory(registry_copy):
    payload = read_manifest(registry_copy)
    payload["model_id"] = "some-other-model"
    write_manifest(registry_copy, payload)
    with pytest.raises(RegistryError, match="does not match its directory"):
        load_entry(_version_dir(registry_copy))


def test_version_must_match_its_directory(registry_copy):
    payload = read_manifest(registry_copy)
    payload["version"] = "9.9.9"
    write_manifest(registry_copy, payload)
    with pytest.raises(RegistryError, match="does not match its directory"):
        load_entry(_version_dir(registry_copy))


def test_missing_artifact_file(registry_copy):
    (_version_dir(registry_copy) / "model.joblib").unlink()
    with pytest.raises(RegistryError, match="is missing"):
        load_entry(_version_dir(registry_copy))


def test_checksum_mismatch_is_caught(registry_copy):
    payload = read_manifest(registry_copy)
    payload["artifact"]["sha256"] = "0" * 64
    write_manifest(registry_copy, payload)
    with pytest.raises(RegistryError, match="checksum mismatch"):
        load_entry(_version_dir(registry_copy))


def test_corrupt_artifact_reports_the_autoemulate_versions(registry_copy):
    (_version_dir(registry_copy) / "model.joblib").write_bytes(b"not a joblib file")
    payload = read_manifest(registry_copy)
    payload["artifact"].pop("sha256", None)  # skip the checksum gate to reach the load
    write_manifest(registry_copy, payload)
    with pytest.raises(RegistryError, match="failed to deserialize"):
        load_entry(_version_dir(registry_copy))


def test_input_count_mismatch_is_caught_by_the_probe(registry_copy):
    """A manifest that lies about the feature count must not reach production."""
    payload = read_manifest(registry_copy)
    payload["inputs"].append(
        {"name": "invented", "unit": "dimensionless", "min": 0.0, "max": 1.0}
    )
    write_manifest(registry_copy, payload)
    with pytest.raises(RegistryError, match="does not match its manifest"):
        load_entry(_version_dir(registry_copy))


def test_supports_uq_claim_is_verified_against_the_artifact(registry_copy):
    payload = read_manifest(registry_copy)
    payload["artifact"]["supports_uq"] = False
    write_manifest(registry_copy, payload)
    with pytest.raises(RegistryError, match="supports_uq"):
        load_entry(_version_dir(registry_copy))


def test_strict_mode_refuses_to_start(registry_copy):
    write_manifest(registry_copy, {"nonsense": True})
    with pytest.raises(RegistryError):
        load_registry(registry_copy, strict=True)


def test_non_strict_mode_quarantines_and_reports(registry_copy):
    write_manifest(registry_copy, {"nonsense": True})
    registry = load_registry(registry_copy, strict=False)
    assert len(registry) == 0
    assert len(registry.failures) == 1
    assert "manifest.json" in registry.failures[0].error


def test_version_directory_without_a_manifest_is_skipped(registry_copy):
    (registry_copy / BATTERY_ID / "2.0.0").mkdir()
    registry = load_registry(registry_copy)
    assert registry.versions(BATTERY_ID) == [BATTERY_VERSION]


def test_latest_version_uses_numeric_ordering(registry_copy):
    """1.10.0 is newer than 1.9.0 — string sorting would get this wrong."""
    import shutil

    source = _version_dir(registry_copy)
    for version in ("1.9.0", "1.10.0"):
        destination = registry_copy / BATTERY_ID / version
        shutil.copytree(source, destination)
        payload = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
        payload["version"] = version
        (destination / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")

    registry = load_registry(registry_copy)
    assert registry.latest_version(BATTERY_ID) == "1.10.0"
    assert registry.versions(BATTERY_ID) == ["1.0.0", "1.9.0", "1.10.0"]
