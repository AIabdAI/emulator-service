"""Measure serving latency honestly.

Sends real prediction batches at a running service and reports p50/p95/p99 wall-clock
latency as a client would experience it — HTTP included, not just tensor maths.

    python scripts/loadtest.py --url http://localhost:8000 --model battery-capacity-fade

Defaults to batch=100, which is the figure quoted in the README. The rows are sampled
across the model's training domain (read from `/models/{id}`), so the emulator does real
work rather than answering the same cached point repeatedly.
"""

from __future__ import annotations

import argparse
import http.client
import json
import pathlib
import random
import statistics
import sys
import time
from urllib.parse import urlparse


class Client:
    """A minimal keep-alive HTTP client.

    Connection reuse is not an optimisation here, it is a correctness requirement for
    the measurement: a fresh TCP connection per request measures the client's socket
    setup (and, behind Docker Desktop's Windows port proxy, its connection backlog)
    rather than the service. Real callers hold a pooled connection, so this does too.
    """

    def __init__(self, base_url: str, timeout: float):
        parsed = urlparse(base_url)
        self.host = parsed.hostname or "localhost"
        self.port = parsed.port or (443 if parsed.scheme == "https" else 80)
        self.secure = parsed.scheme == "https"
        self.timeout = timeout
        self._connection: http.client.HTTPConnection | None = None

    def _connect(self) -> http.client.HTTPConnection:
        if self._connection is None:
            factory = (
                http.client.HTTPSConnection if self.secure else http.client.HTTPConnection
            )
            self._connection = factory(self.host, self.port, timeout=self.timeout)
        return self._connection

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def request(
        self, method: str, path: str, payload: dict | None = None
    ) -> tuple[int, dict, dict]:
        body = json.dumps(payload).encode() if payload is not None else None
        headers = {"Content-Type": "application/json"} if body else {}
        for attempt in (1, 2):
            try:
                connection = self._connect()
                connection.request(method, path, body=body, headers=headers)
                response = connection.getresponse()
                raw = response.read()
                return (
                    response.status,
                    json.loads(raw or b"{}"),
                    dict(response.getheaders()),
                )
            except (http.client.HTTPException, OSError):
                # A pooled connection can be closed by the server between requests;
                # reconnect once before giving up.
                self.close()
                if attempt == 2:
                    raise
        raise AssertionError("unreachable")


def build_batch(inputs: list[dict], size: int, rng: random.Random) -> list[dict]:
    """Random points inside the training domain — never on the boundary."""
    return [
        {
            parameter["name"]: rng.uniform(parameter["min"], parameter["max"])
            for parameter in inputs
        }
        for _ in range(size)
    ]


def percentile(values: list[float], fraction: float) -> float:
    """Nearest-rank percentile. Explicit, so the reported number is unambiguous."""
    if not values:
        raise ValueError("no samples")
    ordered = sorted(values)
    rank = max(1, min(len(ordered), int(round(fraction * len(ordered) + 0.5))))
    return ordered[rank - 1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--model", default="battery-capacity-fade")
    parser.add_argument("--batch", type=int, default=100, help="rows per request")
    parser.add_argument("--requests", type=int, default=100, help="requests to time")
    parser.add_argument("--warmup", type=int, default=10, help="untimed requests first")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--json", help="also write the summary to this path")
    args = parser.parse_args(argv)

    base = args.url.rstrip("/")
    rng = random.Random(args.seed)
    client = Client(base, args.timeout)

    try:
        _, health, _ = client.request("GET", "/health")
        status, detail, _ = client.request("GET", f"/models/{args.model}")
    except OSError as exc:
        print(f"cannot reach {base}: {exc}", file=sys.stderr)
        print("Is the service running? `docker compose up`", file=sys.stderr)
        return 2
    if status != 200:
        print(f"model {args.model!r} not available: {detail}", file=sys.stderr)
        return 2

    inputs = detail["manifest"]["inputs"]
    endpoint = f"/models/{args.model}/predict"

    # Warm up: the first calls pay for lazily initialised torch kernels and would
    # otherwise sit in the tail and flatter nothing but the p99.
    for _ in range(args.warmup):
        status, _, _ = client.request(
            "POST", endpoint, {"rows": build_batch(inputs, args.batch, rng)}
        )
        if status != 200:
            print(f"warmup request failed with {status}", file=sys.stderr)
            return 1

    latencies: list[float] = []
    server_latencies: list[float] = []
    started = time.perf_counter()

    for index in range(args.requests):
        payload = {"rows": build_batch(inputs, args.batch, rng)}
        request_started = time.perf_counter()
        status, body, headers = client.request("POST", endpoint, payload)
        elapsed_ms = (time.perf_counter() - request_started) * 1000

        if status != 200:
            print(f"request {index} failed with {status}: {body}", file=sys.stderr)
            return 1
        if body["n_rows"] != args.batch:
            print(f"request {index} returned {body['n_rows']} rows", file=sys.stderr)
            return 1

        latencies.append(elapsed_ms)
        # ASGI lowercases response header names, so match case-insensitively.
        header = next(
            (
                value
                for key, value in headers.items()
                if key.lower() == "x-prediction-latency-ms"
            ),
            None,
        )
        if header:
            server_latencies.append(float(header))

    total_seconds = time.perf_counter() - started
    client.close()

    summary = {
        "url": base,
        "model_id": args.model,
        "model_version": detail["version"],
        "emulator_class": detail["emulator_class"],
        "autoemulate_version": health["autoemulate_version"],
        "batch_size": args.batch,
        "requests": args.requests,
        "warmup": args.warmup,
        "client_ms": {
            "p50": round(percentile(latencies, 0.50), 2),
            "p95": round(percentile(latencies, 0.95), 2),
            "p99": round(percentile(latencies, 0.99), 2),
            "min": round(min(latencies), 2),
            "max": round(max(latencies), 2),
            "mean": round(statistics.fmean(latencies), 2),
        },
        "server_inference_ms": {
            "p50": round(percentile(server_latencies, 0.50), 2),
            "p95": round(percentile(server_latencies, 0.95), 2),
        }
        if server_latencies
        else None,
        "throughput_rows_per_second": round(
            args.requests * args.batch / total_seconds, 1
        ),
        "wall_clock_seconds": round(total_seconds, 2),
    }

    client = summary["client_ms"]
    print(f"model              {args.model} v{detail['version']}")
    print(f"emulator           {detail['emulator_class']}")
    print(f"batch size         {args.batch} rows")
    print(f"requests           {args.requests} (after {args.warmup} warmup)")
    print("")
    print("latency, end to end over HTTP (ms)")
    print(f"  p50              {client['p50']}")
    print(f"  p95              {client['p95']}")
    print(f"  p99              {client['p99']}")
    print(f"  min / max        {client['min']} / {client['max']}")
    if summary["server_inference_ms"]:
        server = summary["server_inference_ms"]
        print("")
        print("server-side inference only (X-Prediction-Latency-Ms, ms)")
        print(f"  p50              {server['p50']}")
        print(f"  p95              {server['p95']}")
    print("")
    print(f"throughput         {summary['throughput_rows_per_second']} rows/s")

    if args.json:
        pathlib.Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        print(f"\nwrote {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
