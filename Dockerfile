# syntax=docker/dockerfile:1
#
# Multi-stage build for the serving image.
#
# Stage 1 builds a virtualenv with CPU-only PyTorch and the *serving* requirements.
# Stage 2 copies that venv into a slim runtime that carries no compilers, no build
# caches, and — deliberately — no simulator libraries: pybamm, openseespy and pvlib
# are needed to *produce* training data, never to answer a prediction. Keeping them
# out is what makes this image small enough to scale horizontally, and removes a large
# slice of attack surface from the process facing the network.

ARG PYTHON_VERSION=3.12

# --------------------------------------------------------------------- builder
FROM python:${PYTHON_VERSION}-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

RUN python -m venv "$VIRTUAL_ENV"

# CPU-only torch first, from PyTorch's CPU index. Installing it up front means the
# generic PyPI resolution below sees torch as already satisfied and never pulls the
# multi-gigabyte CUDA wheel.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch

COPY requirements/serving.txt /tmp/requirements/serving.txt
RUN pip install --no-cache-dir -r /tmp/requirements/serving.txt

# Strip test suites and bytecode caches that ship inside the wheels.
RUN find /opt/venv -type d -name '__pycache__' -prune -exec rm -rf {} + && \
    find /opt/venv -type d -name 'tests' -prune -exec rm -rf {} + && \
    find /opt/venv -type f -name '*.pyc' -delete

# --------------------------------------------------------------------- runtime
FROM python:${PYTHON_VERSION}-slim AS runtime

LABEL org.opencontainers.image.title="emulator-service" \
      org.opencontainers.image.description="Serving layer for AutoEmulate scientific emulators" \
      org.opencontainers.image.source="https://github.com/AIabdAI/emulator-service"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH=/app/src \
    REGISTRY_PATH=/app/registry \
    LOG_LEVEL=INFO \
    OMP_NUM_THREADS=1

# Non-root from here on: the process never needs to write to its own image.
RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin appuser

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY --chown=appuser:appuser src/ /app/src/
COPY --chown=appuser:appuser registry/ /app/registry/

USER appuser
EXPOSE 8000

# Uses the interpreter already in the image rather than adding curl to the runtime.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request as u, sys; sys.exit(0 if u.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
