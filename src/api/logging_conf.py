"""Structured JSON logging.

One JSON object per line on stdout, which is what a container orchestrator, a log
shipper or `docker compose logs | jq` all expect. Deliberately dependency-free: the
serving image should not carry a logging library it does not need.
"""

from __future__ import annotations

import contextvars
import datetime as dt
import json
import logging
import os
import sys
from typing import Any

SERVICE_NAME = os.environ.get("SERVICE_NAME", "emulator-service")

#: Set per request by the logging middleware so every log line in a request is joinable.
request_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "request_id", default=None
)

_RESERVED = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {
    "asctime",
    "message",
    "taskName",
}


class JsonFormatter(logging.Formatter):
    """Render a log record as a single JSON line, including any `extra=` fields."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": dt.datetime.fromtimestamp(record.created, tz=dt.UTC).isoformat(
                timespec="milliseconds"
            ),
            "level": record.levelname,
            "service": SERVICE_NAME,
            "logger": record.name,
            "message": record.getMessage(),
        }

        request_id = request_id_var.get()
        if request_id is not None:
            payload["request_id"] = request_id

        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


def configure_logging(level: str | None = None) -> None:
    """Install the JSON formatter on the root logger and on uvicorn's loggers.

    Idempotent: calling it twice does not double up handlers, so reload-mode uvicorn
    and the test client both stay quiet.
    """
    level = (level or os.environ.get("LOG_LEVEL", "INFO")).upper()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())

    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(level)

    # uvicorn installs its own handlers; strip them so access logs are JSON too.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uvicorn_logger = logging.getLogger(name)
        uvicorn_logger.handlers.clear()
        uvicorn_logger.propagate = True

    # The access log duplicates our own request log line, with less information.
    logging.getLogger("uvicorn.access").disabled = True
