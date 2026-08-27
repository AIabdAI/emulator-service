"""API tests against the real registry, through FastAPI's TestClient."""

from __future__ import annotations

import json
import math

import pytest

from tests.conftest import BATTERY_ID, BATTERY_VERSION

PREDICT = f"/models/{BATTERY_ID}/predict"


# ------------------------------------------------------------------------ ops


def test_health_reports_loaded_models(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["models_loaded"] >= 2
    assert BATTERY_ID in body["model_ids"]
    assert body["autoemulate_version"] == "1.2.1"
    assert body["failures"] == []


def test_every_response_carries_a_request_id(client):
    response = client.get("/health")
    assert response.headers["X-Request-ID"]
    assert float(response.headers["X-Prediction-Latency-Ms"]) >= 0


def test_supplied_request_id_is_echoed_back(client):
    response = client.get("/health", headers={"X-Request-ID": "trace-me-123"})
    assert response.headers["X-Request-ID"] == "trace-me-123"


# --------------------------------------------------------------------- models


def test_list_models(client):
    body = client.get("/models").json()
    assert body["count"] == len(body["models"])
    battery = next(m for m in body["models"] if m["model_id"] == BATTERY_ID)
    assert battery["supports_uq"] is True
    assert battery["output"] == "capacity_fade"
    assert battery["output_unit"] == "percent"
    assert battery["latest"] is True
    # Stand-in status is part of the contract, not a footnote.
    assert battery["stand_in"] is True


def test_list_models_latest_only(client):
    body = client.get("/models", params={"latest_only": True}).json()
    assert all(model["latest"] for model in body["models"])


def test_model_detail_exposes_the_full_manifest(client, battery_manifest):
    body = client.get(f"/models/{BATTERY_ID}").json()
    assert body["version"] == BATTERY_VERSION
    assert body["available_versions"] == [BATTERY_VERSION]
    assert body["manifest"]["inputs"] == battery_manifest["inputs"]
    assert body["manifest"]["dataset"]["hash"].startswith("sha256:")
    assert body["example_request"]["rows"]


def test_model_detail_example_request_actually_works(client):
    """The documented example must not be aspirational."""
    example = client.get(f"/models/{BATTERY_ID}").json()["example_request"]
    assert client.post(PREDICT, json=example).status_code == 200


def test_unknown_model_is_404_and_lists_what_exists(client):
    response = client.get("/models/no-such-model")
    assert response.status_code == 404
    body = response.json()
    assert body["error"] == "model_not_found"
    assert BATTERY_ID in body["detail"]


def test_unknown_version_is_404_and_lists_known_versions(client):
    response = client.get(f"/models/{BATTERY_ID}", params={"version": "7.7.7"})
    assert response.status_code == 404
    assert BATTERY_VERSION in response.json()["detail"]


# -------------------------------------------------------------------- predict


def test_happy_path_returns_mean_and_std(client, in_bounds_row):
    response = client.post(PREDICT, json={"rows": [in_bounds_row]})
    assert response.status_code == 200
    body = response.json()

    assert body["model_id"] == BATTERY_ID
    assert body["version"] == BATTERY_VERSION
    assert body["output"] == "capacity_fade"
    assert body["output_unit"] == "percent"
    assert body["supports_uq"] is True
    assert body["n_rows"] == 1

    prediction = body["predictions"][0]
    assert math.isfinite(prediction["mean"])
    assert prediction["std"] is not None
    assert prediction["std"] >= 0.0


def test_latency_header_is_present_on_predictions(client, in_bounds_row):
    response = client.post(PREDICT, json={"rows": [in_bounds_row]})
    assert float(response.headers["X-Prediction-Latency-Ms"]) >= 0


def test_batch_prediction_preserves_row_order(client, battery_manifest):
    """A batch must come back aligned with what was sent, one prediction per row."""
    c_rate = battery_manifest["inputs"][0]
    temperature = battery_manifest["inputs"][1]
    dod = battery_manifest["inputs"][2]

    def blend(parameter, fraction):
        return parameter["min"] + fraction * (parameter["max"] - parameter["min"])

    fractions = [i / 19 for i in range(20)]
    rows = [
        {
            "c_rate": blend(c_rate, f),
            "temperature": blend(temperature, 0.5),
            "depth_of_discharge": blend(dod, 0.5),
        }
        for f in fractions
    ]

    body = client.post(PREDICT, json={"rows": rows}).json()
    assert body["n_rows"] == 20
    assert len(body["predictions"]) == 20

    means = [p["mean"] for p in body["predictions"]]
    assert all(math.isfinite(m) for m in means)
    # Capacity fade rises monotonically with C-rate in the stand-in response surface;
    # a batch that came back shuffled would not preserve that.
    assert means[-1] > means[0]

    # Same row, alone vs inside the batch. Not bit-identical: gpytorch conditions the
    # exact-GP posterior on the whole test block and jitters the covariance to keep it
    # positive definite, so batch composition moves the answer at the 1e-4 level.
    # Row *alignment* is what must hold exactly.
    single = client.post(PREDICT, json={"rows": [rows[7]]}).json()
    assert single["predictions"][0]["mean"] == pytest.approx(
        body["predictions"][7]["mean"], rel=1e-3
    )


def test_explicit_version_is_honoured(client, in_bounds_row):
    response = client.post(
        PREDICT, json={"rows": [in_bounds_row]}, params={"version": BATTERY_VERSION}
    )
    assert response.status_code == 200
    assert response.json()["version"] == BATTERY_VERSION


def test_predict_on_unknown_model_is_404(client, in_bounds_row):
    response = client.post("/models/ghost/predict", json={"rows": [in_bounds_row]})
    assert response.status_code == 404


# ------------------------------------------------------- bounds and validation


def test_out_of_bounds_is_rejected_with_parameter_and_range(
    client, in_bounds_row, battery_manifest
):
    """The core guarantee: an out-of-domain value cannot reach the emulator."""
    parameter = battery_manifest["inputs"][0]
    row = dict(in_bounds_row)
    row["c_rate"] = parameter["max"] + 10.0

    response = client.post(PREDICT, json={"rows": [row]})
    assert response.status_code == 422

    body = response.json()
    assert body["error"] == "input_out_of_contract"
    assert body["model_id"] == BATTERY_ID
    assert body["version"] == BATTERY_VERSION

    violation = body["violations"][0]
    assert violation["parameter"] == "c_rate"
    assert violation["row"] == 0
    assert violation["min"] == parameter["min"]
    assert violation["max"] == parameter["max"]
    assert violation["unit"] == parameter["unit"]
    # The message must name the parameter and its valid range.
    assert "c_rate" in violation["reason"]
    assert str(parameter["max"]) in violation["reason"]


def test_below_lower_bound_is_also_rejected(client, in_bounds_row, battery_manifest):
    row = dict(in_bounds_row)
    row["temperature"] = battery_manifest["inputs"][1]["min"] - 0.001
    response = client.post(PREDICT, json={"rows": [row]})
    assert response.status_code == 422
    assert response.json()["violations"][0]["parameter"] == "temperature"


def test_bounds_are_inclusive_at_the_edges(client, in_bounds_row, battery_manifest):
    """The training domain includes its endpoints; rejecting them would be wrong."""
    row = dict(in_bounds_row)
    for parameter in battery_manifest["inputs"]:
        row[parameter["name"]] = parameter["min"]
    assert client.post(PREDICT, json={"rows": [row]}).status_code == 200

    for parameter in battery_manifest["inputs"]:
        row[parameter["name"]] = parameter["max"]
    assert client.post(PREDICT, json={"rows": [row]}).status_code == 200


def test_one_bad_row_rejects_the_whole_batch(client, in_bounds_row, battery_manifest):
    """Partial service would leave a caller unsure which predictions to trust."""
    bad = dict(in_bounds_row)
    bad["c_rate"] = battery_manifest["inputs"][0]["max"] * 100

    response = client.post(PREDICT, json={"rows": [in_bounds_row, bad, in_bounds_row]})
    assert response.status_code == 422
    assert response.json()["violations"][0]["row"] == 1


def test_all_violations_are_reported_not_just_the_first(
    client, in_bounds_row, battery_manifest
):
    bad = {
        parameter["name"]: parameter["max"] + 100.0
        for parameter in battery_manifest["inputs"]
    }
    body = client.post(PREDICT, json={"rows": [bad]}).json()
    assert {v["parameter"] for v in body["violations"]} == {
        parameter["name"] for parameter in battery_manifest["inputs"]
    }


def test_missing_parameter_is_rejected(client, in_bounds_row):
    row = dict(in_bounds_row)
    row.pop("temperature")
    body = client.post(PREDICT, json={"rows": [row]}).json()
    assert body["violations"][0]["parameter"] == "temperature"
    assert "required" in body["violations"][0]["reason"]


def test_unknown_parameter_is_rejected(client, in_bounds_row):
    """A typo must be an error, not a silently ignored field."""
    row = dict(in_bounds_row)
    row["temperatur"] = 25.0
    body = client.post(PREDICT, json={"rows": [row]}).json()
    assert body["violations"][0]["parameter"] == "temperatur"
    assert "not an input" in body["violations"][0]["reason"]


def test_non_numeric_value_is_rejected(client, in_bounds_row):
    row = dict(in_bounds_row)
    row["c_rate"] = "warm"
    response = client.post(PREDICT, json={"rows": [row]})
    assert response.status_code == 422
    assert response.json()["violations"][0]["parameter"] == "c_rate"


def test_nan_and_infinity_are_rejected(client, in_bounds_row):
    """Non-finite input must never reach an emulator.

    Sent as raw bytes: httpx refuses to serialize NaN, but Python's json module — and
    therefore Starlette's request parser — accepts the bare `NaN` / `Infinity` literals
    a non-Python client can easily emit.
    """
    for literal in ("NaN", "Infinity", "-Infinity"):
        row = dict(in_bounds_row)
        body = json.dumps({"rows": [row]}).replace(json.dumps(row["c_rate"]), literal, 1)
        response = client.post(
            PREDICT, content=body, headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422, literal


def test_empty_batch_is_rejected(client):
    response = client.post(PREDICT, json={"rows": []})
    assert response.status_code == 422
    assert response.json()["error"] == "invalid_request"


def test_oversized_batch_is_rejected(client, in_bounds_row):
    response = client.post(PREDICT, json={"rows": [in_bounds_row] * 1001})
    assert response.status_code == 422
    assert response.json()["error"] == "invalid_request"


def test_missing_rows_key_is_rejected(client):
    response = client.post(PREDICT, json={"batch": []})
    assert response.status_code == 422
    assert response.json()["error"] == "invalid_request"


def test_row_that_is_not_an_object_is_rejected(client):
    response = client.post(PREDICT, json={"rows": [[1.0, 2.0, 3.0]]})
    assert response.status_code == 422


# ------------------------------------------------------------------ contract


def test_openapi_documents_the_bounds(client, battery_manifest):
    """OpenAPI is generated, but the bounds live in the manifest — check they surface."""
    detail = client.get(f"/models/{BATTERY_ID}").json()
    documented = {p["name"]: (p["min"], p["max"]) for p in detail["manifest"]["inputs"]}
    expected = {p["name"]: (p["min"], p["max"]) for p in battery_manifest["inputs"]}
    assert documented == expected


def test_second_model_is_served_independently(client):
    detail = client.get("/models/frame-peak-drift").json()
    row = {
        parameter["name"]: (parameter["min"] + parameter["max"]) / 2
        for parameter in detail["manifest"]["inputs"]
    }
    body = client.post("/models/frame-peak-drift/predict", json={"rows": [row]}).json()
    assert body["output"] == "peak_drift_ratio"
    assert body["predictions"][0]["std"] is not None

    # A battery row must not be accepted by the frame model.
    response = client.post(
        "/models/frame-peak-drift/predict",
        json={"rows": [{"c_rate": 1.0, "temperature": 25.0, "depth_of_discharge": 0.5}]},
    )
    assert response.status_code == 422
