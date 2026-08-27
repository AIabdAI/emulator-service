"""Shared fixtures.

The API tests run against the *real* registry committed in this repo — real joblib
artifacts, deserialized by real AutoEmulate code. Loading them is slow enough to be
worth doing once per session, and valuable enough to be worth doing at all: a test
suite that mocks the emulator away would not catch a broken serialization contract,
which is the single most likely thing to break here.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_ROOT = REPO_ROOT / "registry"
BATTERY_ID = "battery-capacity-fade"
BATTERY_VERSION = "1.0.0"
BATTERY_DIR = REGISTRY_ROOT / BATTERY_ID / BATTERY_VERSION


@pytest.fixture(scope="session")
def client() -> TestClient:
    """A TestClient with the real registry loaded.

    Used as a context manager so FastAPI's lifespan actually runs — without it
    `app.state.registry` is never populated.
    """
    from api.main import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="session")
def battery_manifest() -> dict:
    return json.loads((BATTERY_DIR / "manifest.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def in_bounds_row(battery_manifest: dict) -> dict[str, float]:
    """A row at the centre of the training domain — always valid."""
    return {
        parameter["name"]: (parameter["min"] + parameter["max"]) / 2
        for parameter in battery_manifest["inputs"]
    }


@pytest.fixture
def registry_copy(tmp_path: Path) -> Path:
    """An isolated copy of the battery model, for tests that corrupt it."""
    root = tmp_path / "registry"
    destination = root / BATTERY_ID / BATTERY_VERSION
    destination.parent.mkdir(parents=True)
    shutil.copytree(BATTERY_DIR, destination)
    return root


def read_manifest(
    registry_root: Path, model_id: str = BATTERY_ID, version: str = BATTERY_VERSION
) -> dict:
    path = registry_root / model_id / version / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))


def write_manifest(
    registry_root: Path,
    payload: dict,
    model_id: str = BATTERY_ID,
    version: str = BATTERY_VERSION,
) -> Path:
    path = registry_root / model_id / version / "manifest.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
