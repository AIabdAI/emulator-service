"""Minimal load test: p50 / p95 latency for a batch of N rows on CPU.

Deliberately simple and dependency-free -- it measures the thing the README quotes and
nothing else. Run it against a live service (``docker compose up``) or in-process.

Usage:
    python scripts/loadtest.py --model synthetic-smooth --batch 100 --requests 200
    python scripts/loadtest.py --url http://localhost:8000 --model battery-capacity
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Measure prediction latency.")
    ap.add_argument("--url", default=None,
                    help="Base URL of a running service. Omitted = in-process TestClient.")
    ap.add_argument("--model", default=None, help="model_id (default: the first registered)")
    ap.add_argument("--batch", type=int, default=100, help="Rows per request")
    ap.add_argument("--requests", type=int, default=100, help="Number of timed requests")
    ap.add_argument("--warmup", type=int, default=10, help="Untimed warm-up requests")
    return ap.parse_args(argv)


def build_rows(manifest: dict, n: int) -> list[dict[str, float]]:
    """n rows spread across the declared training domain."""
    rows = []
    for i in range(n):
        frac = (i + 0.5) / n
        rows.append(
            {spec["name"]: spec["min"] + frac * (spec["max"] - spec["min"])
             for spec in manifest["inputs"]}
        )
    return rows


class HttpClient:
    def __init__(self, base: str):
        self.base = base.rstrip("/")

    def get(self, path: str) -> dict:
        with urllib.request.urlopen(f"{self.base}{path}", timeout=30) as r:
            return json.loads(r.read())

    def predict(self, model_id: str, rows: list[dict]) -> dict:
        body = json.dumps({"inputs": rows}).encode()
        req = urllib.request.Request(
            f"{self.base}/models/{model_id}/predict",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read())


class InProcessClient:
    """Runs the app in-process, so the numbers exclude network and container overhead."""

    def __init__(self):
        from fastapi.testclient import TestClient

        from src.api.main import app

        self._ctx = TestClient(app)
        self.client = self._ctx.__enter__()

    def get(self, path: str) -> dict:
        return self.client.get(path).json()

    def predict(self, model_id: str, rows: list[dict]) -> dict:
        r = self.client.post(f"/models/{model_id}/predict", json={"inputs": rows})
        if r.status_code != 200:
            raise RuntimeError(f"predict failed: {r.status_code} {r.text}")
        return r.json()

    def close(self) -> None:
        self._ctx.__exit__(None, None, None)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    client = HttpClient(args.url) if args.url else InProcessClient()

    try:
        models = client.get("/models")
        if not models:
            print("No models registered. Run scripts/build_registry.py first.")
            return 1
        model_id = args.model or models[0]["model_id"]
        manifest = client.get(f"/models/{model_id}")
        rows = build_rows(manifest, args.batch)

        mode = f"HTTP {args.url}" if args.url else "in-process"
        print(f"Model {model_id} | batch={args.batch} | {args.requests} requests | {mode}")

        for _ in range(args.warmup):
            client.predict(model_id, rows)

        latencies = []
        t0 = time.perf_counter()
        for _ in range(args.requests):
            start = time.perf_counter()
            client.predict(model_id, rows)
            latencies.append((time.perf_counter() - start) * 1000.0)
        wall = time.perf_counter() - t0
    except (urllib.error.URLError, ConnectionError) as exc:
        print(f"Could not reach the service: {exc}")
        return 1
    finally:
        if isinstance(client, InProcessClient):
            client.close()

    latencies.sort()

    def pct(p: float) -> float:
        return latencies[min(len(latencies) - 1, int(p / 100 * len(latencies)))]

    print(f"\n  requests      {len(latencies)}")
    print(f"  rows/request  {args.batch}")
    print(f"  p50           {statistics.median(latencies):8.2f} ms")
    print(f"  p95           {pct(95):8.2f} ms")
    print(f"  p99           {pct(99):8.2f} ms")
    print(f"  mean          {statistics.fmean(latencies):8.2f} ms")
    print(f"  min / max     {latencies[0]:8.2f} / {latencies[-1]:.2f} ms")
    print(f"  throughput    {args.requests * args.batch / wall:8.0f} rows/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
