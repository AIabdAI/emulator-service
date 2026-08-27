"""Manifest schema tests — fast, no emulator deserialization."""

from __future__ import annotations

import json

import pytest

from api.manifest import (
    Manifest,
    RegistryError,
    load_manifest,
    sha256_file,
    write_manifest,
)


def test_real_manifest_parses(battery_manifest):
    manifest = Manifest.model_validate(battery_manifest)
    assert manifest.model_id == "battery-capacity-fade"
    assert manifest.version == "1.0.0"
    assert manifest.input_names == [p["name"] for p in battery_manifest["inputs"]]
    assert manifest.ref == "battery-capacity-fade v1.0.0"


def test_input_order_is_tensor_column_order(battery_manifest):
    """Manifest input order defines the feature-column order; it must be preserved."""
    manifest = Manifest.model_validate(battery_manifest)
    assert manifest.input_names == [
        "c_rate",
        "temperature",
        "depth_of_discharge",
    ]


@pytest.mark.parametrize(
    ("mutate", "expected_fragment"),
    [
        (lambda m: m.__setitem__("model_id", "Battery_Fade"), "model_id"),
        (lambda m: m.__setitem__("version", "1.0"), "semantic"),
        (lambda m: m.__setitem__("schema_version", 99), "schema_version"),
        (lambda m: m["inputs"][0].__setitem__("max", -5.0), "greater than"),
        (lambda m: m["inputs"].append(dict(m["inputs"][0])), "duplicate"),
        (lambda m: m.__setitem__("inputs", []), "inputs"),
        (lambda m: m["metrics"].__setitem__("rmse", -1.0), "rmse"),
        (lambda m: m["dataset"].__setitem__("hash", "not-a-hash"), "sha256"),
        (lambda m: m["artifact"].__setitem__("filename", "../escape.joblib"), "relative"),
        (lambda m: m["artifact"].__setitem__("format", "pickle"), "joblib"),
        (lambda m: m.__setitem__("unexpected_key", 1), "unexpected_key"),
        (lambda m: m.pop("output"), "output"),
    ],
)
def test_malformed_manifest_is_rejected_with_a_useful_message(
    battery_manifest, tmp_path, mutate, expected_fragment
):
    payload = json.loads(json.dumps(battery_manifest))
    mutate(payload)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RegistryError) as excinfo:
        load_manifest(path)

    message = str(excinfo.value)
    assert expected_fragment in message
    # The path is always part of the message; an operator must know where to look.
    assert "manifest.json" in message


def test_manifest_that_is_not_json_names_the_position(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text('{"model_id": "x",,}', encoding="utf-8")
    with pytest.raises(RegistryError, match="not valid JSON"):
        load_manifest(path)


def test_manifest_that_is_not_an_object(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(RegistryError, match="expected a JSON object"):
        load_manifest(path)


def test_missing_manifest(tmp_path):
    with pytest.raises(RegistryError, match="not found"):
        load_manifest(tmp_path / "manifest.json")


def test_extra_metrics_are_preserved(battery_manifest):
    payload = json.loads(json.dumps(battery_manifest))
    payload["metrics"]["mae"] = 0.03
    manifest = Manifest.model_validate(payload)
    assert manifest.metrics.model_dump()["mae"] == 0.03


def test_write_manifest_refuses_to_overwrite(battery_manifest, tmp_path):
    """Registry versions are immutable — that guarantee is enforced, not documented."""
    manifest = Manifest.model_validate(battery_manifest)
    version_dir = tmp_path / "battery-capacity-fade" / "1.0.0"
    write_manifest(version_dir, manifest)
    with pytest.raises(RegistryError, match="immutable"):
        write_manifest(version_dir, manifest)


def test_sha256_file_matches_manifest_claim(battery_manifest):
    from tests.conftest import BATTERY_DIR

    artifact = BATTERY_DIR / battery_manifest["artifact"]["filename"]
    assert sha256_file(artifact) == battery_manifest["artifact"]["sha256"]


def test_parameter_lookup_and_containment(battery_manifest):
    manifest = Manifest.model_validate(battery_manifest)
    parameter = manifest.parameter("c_rate")
    assert parameter is not None
    assert parameter.contains(parameter.midpoint)
    assert not parameter.contains(parameter.max + 1.0)
    assert manifest.parameter("nope") is None
