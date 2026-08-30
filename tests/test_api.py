"""API tests: happy path, bounds rejection, malformed manifests, batch prediction.

The bounds-rejection tests are the important ones. They are the proof for the
acceptance criterion "out-of-bounds input can NOT reach an emulator": one asserts the
422 response, and one patches the emulator's ``predict`` to explode if it is ever
called, so a regression that let a bad value through would fail loudly.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api import main as api_main
from src.api.registry_loader import load_registry

REPO = Path(__file__).resolve().parents[1]
SOURCE_REGISTRY = REPO / "registry"


@pytest.fixture(scope="session")
def registry_root(tmp_path_factory) -> Path:
    """A private copy of the registry, so tests never mutate the real one."""
    if not any(SOURCE_REGISTRY.glob("*/*/manifest.json")):
        pytest.skip("No models in registry; run scripts/build_registry.py first")
    dest = tmp_path_factory.mktemp("registry")
    for manifest in SOURCE_REGISTRY.glob("*/*/manifest.json"):
        version_dir = manifest.parent
        target = dest / version_dir.parent.name / version_dir.name
        target.mkdir(parents=True, exist_ok=True)
        for item in version_dir.iterdir():
            if item.is_file():
                shutil.copy2(item, target / item.name)
    return dest


@pytest.fixture(scope="session")
def client(registry_root: Path) -> TestClient:
    api_main.STATE["registry"] = load_registry(registry_root)
    with TestClient(api_main.app) as c:
        # The lifespan reloads from the env var, so point it at the temp copy.
        api_main.STATE["registry"] = load_registry(registry_root)
        yield c


@pytest.fixture(scope="session")
def a_model(client: TestClient) -> dict:
    """The first model in the registry, with its manifest."""
    models = client.get("/models").json()
    assert models, "registry is empty"
    return client.get(f"/models/{models[0]['model_id']}").json()


def midpoint_row(manifest: dict) -> dict[str, float]:
    return {i["name"]: (i["min"] + i["max"]) / 2.0 for i in manifest["inputs"]}


# ---------------------------------------------------------------------- health


def test_health_reports_ok_and_a_model_count(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["n_models_loaded"] >= 1
    assert body["autoemulate_version"] != "unknown"


def test_every_response_carries_a_latency_header(client):
    r = client.get("/health")
    assert "X-Response-Time-ms" in r.headers
    assert float(r.headers["X-Response-Time-ms"]) >= 0.0


# ---------------------------------------------------------------------- models


def test_list_models_returns_metadata(client):
    r = client.get("/models")
    assert r.status_code == 200
    models = r.json()
    assert models
    for m in models:
        assert m["model_id"] and m["version"]
        assert m["n_inputs"] >= 1
        assert m["version"] in m["available_versions"]


def test_get_model_returns_the_full_manifest(client, a_model):
    r = client.get(f"/models/{a_model['model_id']}")
    assert r.status_code == 200
    body = r.json()
    assert body["inputs"] and body["output"] and body["metrics"]
    for spec in body["inputs"]:
        assert spec["min"] < spec["max"]


def test_unknown_model_returns_404_listing_known_models(client):
    r = client.get("/models/definitely-not-a-model")
    assert r.status_code == 404
    assert r.json()["error"] == "model_not_found"
    assert "Available models" in r.json()["message"]


def test_unknown_version_returns_404_listing_available_versions(client, a_model):
    r = client.get(f"/models/{a_model['model_id']}?version=99.0.0")
    assert r.status_code == 404
    assert "Available versions" in r.json()["message"]


# --------------------------------------------------------------------- predict


def test_predict_happy_path_returns_mean_and_std(client, a_model):
    row = midpoint_row(a_model)
    r = client.post(f"/models/{a_model['model_id']}/predict", json={"inputs": [row]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model_id"] == a_model["model_id"]
    assert body["version"] == a_model["version"]
    assert body["output_name"] == a_model["output"]["name"]
    assert body["n_rows"] == 1
    pred = body["predictions"][0]
    assert pred["mean"] == pred["mean"]  # not NaN
    assert pred["std"] >= 0.0


def test_predict_handles_a_batch(client, a_model):
    rows = []
    for frac in (0.1, 0.3, 0.5, 0.7, 0.9):
        rows.append({
            i["name"]: i["min"] + frac * (i["max"] - i["min"]) for i in a_model["inputs"]
        })
    r = client.post(f"/models/{a_model['model_id']}/predict", json={"inputs": rows})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n_rows"] == len(rows)
    assert len(body["predictions"]) == len(rows)
    # Distinct inputs should not all collapse to one prediction.
    assert len({round(p["mean"], 9) for p in body["predictions"]}) > 1


def test_predict_is_deterministic(client, a_model):
    row = midpoint_row(a_model)
    a = client.post(f"/models/{a_model['model_id']}/predict", json={"inputs": [row]}).json()
    b = client.post(f"/models/{a_model['model_id']}/predict", json={"inputs": [row]}).json()
    assert a["predictions"][0]["mean"] == pytest.approx(b["predictions"][0]["mean"], rel=1e-9)


# ----------------------------------------------------------- bounds enforcement


def test_above_upper_bound_is_rejected_with_422_naming_the_parameter(client, a_model):
    spec = a_model["inputs"][0]
    row = midpoint_row(a_model)
    row[spec["name"]] = spec["max"] + abs(spec["max"] - spec["min"])
    r = client.post(f"/models/{a_model['model_id']}/predict", json={"inputs": [row]})
    assert r.status_code == 422
    body = r.json()
    assert body["error"] == "input_out_of_bounds"
    assert body["parameter"] == spec["name"]
    assert body["valid_range"] == [spec["min"], spec["max"]]
    assert spec["name"] in body["message"]


def test_below_lower_bound_is_rejected(client, a_model):
    spec = a_model["inputs"][-1]
    row = midpoint_row(a_model)
    row[spec["name"]] = spec["min"] - abs(spec["max"] - spec["min"])
    r = client.post(f"/models/{a_model['model_id']}/predict", json={"inputs": [row]})
    assert r.status_code == 422
    assert r.json()["parameter"] == spec["name"]


def test_out_of_bounds_row_rejects_the_whole_batch(client, a_model):
    good = midpoint_row(a_model)
    bad = midpoint_row(a_model)
    spec = a_model["inputs"][0]
    bad[spec["name"]] = spec["max"] * 10 + 1.0
    r = client.post(f"/models/{a_model['model_id']}/predict",
                    json={"inputs": [good, good, bad]})
    assert r.status_code == 422
    assert r.json()["row"] == 2, "the failing row index must be reported"


def test_out_of_bounds_input_never_reaches_the_emulator(client, a_model, monkeypatch):
    """The load-bearing test for the acceptance criterion.

    If validation ever regressed, ``predict`` would be called and this would fail.
    """
    reg = api_main.STATE["registry"]
    model = reg.get(a_model["model_id"])

    def explode(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("emulator.predict was reached with out-of-bounds input")

    monkeypatch.setattr(model.emulator, "predict", explode)

    spec = a_model["inputs"][0]
    row = midpoint_row(a_model)
    row[spec["name"]] = spec["max"] + 1e6
    r = client.post(f"/models/{a_model['model_id']}/predict", json={"inputs": [row]})
    assert r.status_code == 422


@pytest.mark.parametrize("token", ["Infinity", "-Infinity", "NaN"])
def test_non_finite_values_are_rejected(client, a_model, token):
    """Sent as a raw body: a strict JSON client refuses to encode these at all.

    Python's json module does accept the bare tokens, so they can reach a real server
    and must be rejected there rather than silently reaching the emulator.
    """
    row = midpoint_row(a_model)
    name = a_model["inputs"][0]["name"]
    row[name] = 0.0
    body = json.dumps({"inputs": [row]}).replace(f'"{name}": 0.0', f'"{name}": {token}')
    r = client.post(
        f"/models/{a_model['model_id']}/predict",
        content=body,
        headers={"Content-Type": "application/json"},
    )
    assert r.status_code == 422, r.text


def test_validate_batch_rejects_non_finite_directly(client, a_model):
    """Unit-level guarantee, independent of any JSON transport quirk."""
    from src.api.registry_loader import BoundsError

    model = api_main.STATE["registry"].get(a_model["model_id"])
    row = midpoint_row(a_model)
    row[a_model["inputs"][0]["name"]] = float("inf")
    with pytest.raises(BoundsError):
        model.validate_batch([row])


def test_missing_input_is_rejected_by_name(client, a_model):
    row = midpoint_row(a_model)
    dropped = a_model["inputs"][0]["name"]
    del row[dropped]
    r = client.post(f"/models/{a_model['model_id']}/predict", json={"inputs": [row]})
    assert r.status_code == 422
    assert r.json()["error"] == "missing_input"
    assert dropped in r.json()["message"]


def test_unknown_input_is_rejected(client, a_model):
    row = midpoint_row(a_model)
    row["not_a_real_parameter"] = 1.0
    r = client.post(f"/models/{a_model['model_id']}/predict", json={"inputs": [row]})
    assert r.status_code == 422
    assert r.json()["error"] == "unknown_input"


def test_empty_batch_is_rejected(client, a_model):
    r = client.post(f"/models/{a_model['model_id']}/predict", json={"inputs": []})
    assert r.status_code == 422


# ----------------------------------------------------------- registry validation


def _write_manifest(root: Path, model_id: str, version: str, manifest: dict,
                    artifact_from: Path | None) -> Path:
    d = root / model_id / version
    d.mkdir(parents=True, exist_ok=True)
    (d / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    if artifact_from is not None:
        for f in artifact_from.parent.glob(artifact_from.name + "*"):
            shutil.copy2(f, d / f.name)
    return d


def test_malformed_json_manifest_is_skipped_not_fatal(tmp_path, registry_root):
    shutil.copytree(registry_root, tmp_path / "reg")
    bad = tmp_path / "reg" / "broken-model" / "1.0.0"
    bad.mkdir(parents=True)
    (bad / "manifest.json").write_text("{not valid json", encoding="utf-8")

    reg = load_registry(tmp_path / "reg")
    assert reg.n_loaded >= 1, "one bad manifest must not empty the registry"
    assert any("invalid JSON" in e for e in reg.errors)


def test_manifest_failing_schema_is_skipped_with_a_named_reason(tmp_path, registry_root):
    shutil.copytree(registry_root, tmp_path / "reg")
    _write_manifest(
        tmp_path / "reg", "schema-bad", "1.0.0",
        {"model_id": "schema-bad", "version": "1.0.0"},  # missing most required fields
        None,
    )
    reg = load_registry(tmp_path / "reg")
    assert "schema-bad" not in reg.models
    assert any("schema validation failed" in e for e in reg.errors)


def test_inverted_bounds_are_rejected(tmp_path, registry_root):
    shutil.copytree(registry_root, tmp_path / "reg")
    src = next((registry_root).glob("*/*/manifest.json"))
    manifest = json.loads(src.read_text(encoding="utf-8"))
    manifest["model_id"] = "inverted-bounds"
    manifest["inputs"][0]["min"], manifest["inputs"][0]["max"] = (
        manifest["inputs"][0]["max"], manifest["inputs"][0]["min"],
    )
    _write_manifest(tmp_path / "reg", "inverted-bounds", manifest["version"],
                    manifest, src.parent / "emulator")
    reg = load_registry(tmp_path / "reg")
    assert "inverted-bounds" not in reg.models
    assert any("min" in e and "max" in e for e in reg.errors)


def test_version_directory_mismatch_is_rejected(tmp_path, registry_root):
    shutil.copytree(registry_root, tmp_path / "reg")
    src = next((registry_root).glob("*/*/manifest.json"))
    manifest = json.loads(src.read_text(encoding="utf-8"))
    manifest["model_id"] = "version-mismatch"
    _write_manifest(tmp_path / "reg", "version-mismatch", "2.0.0",  # dir says 2.0.0
                    manifest, src.parent / "emulator")           # manifest says otherwise
    reg = load_registry(tmp_path / "reg")
    assert "version-mismatch" not in reg.models
    assert any("does not match" in e for e in reg.errors)


def test_missing_artifact_is_rejected(tmp_path, registry_root):
    shutil.copytree(registry_root, tmp_path / "reg")
    src = next((registry_root).glob("*/*/manifest.json"))
    manifest = json.loads(src.read_text(encoding="utf-8"))
    manifest["model_id"] = "no-artifact"
    _write_manifest(tmp_path / "reg", "no-artifact", manifest["version"], manifest, None)
    reg = load_registry(tmp_path / "reg")
    assert "no-artifact" not in reg.models
    assert any("not found" in e for e in reg.errors)


def test_highest_version_is_served_by_default(tmp_path, registry_root):
    shutil.copytree(registry_root, tmp_path / "reg")
    src = next((registry_root).glob("*/*/manifest.json"))
    manifest = json.loads(src.read_text(encoding="utf-8"))
    model_id = manifest["model_id"]
    manifest["version"] = "2.3.0"
    _write_manifest(tmp_path / "reg", model_id, "2.3.0", manifest, src.parent / "emulator")

    reg = load_registry(tmp_path / "reg")
    assert reg.get(model_id).manifest.version == "2.3.0"
    assert reg.get(model_id, "1.0.0").manifest.version == "1.0.0"


def test_service_never_imports_a_simulator():
    """The serving path must not drag pybamm/openseespy/pvlib into the process."""
    import subprocess
    import sys

    code = (
        "import sys; "
        "from src.api.registry_loader import load_registry; "
        "from src.api import main; "
        "bad=[m for m in ('pybamm','openseespy','pvlib') if m in sys.modules]; "
        "print('LEAKED:'+','.join(bad) if bad else 'CLEAN')"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=str(REPO), timeout=600
    )
    leaked = f"simulator leaked into the serving path: {out.stdout}{out.stderr}"
    assert "CLEAN" in out.stdout, leaked
