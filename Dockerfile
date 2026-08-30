# Multi-stage build. The final image contains the API, AutoEmulate and torch --
# and deliberately NO simulator (pybamm / openseespy / pvlib). Emulators are loaded
# through AutoEmulate alone, so the serving image stays lean and has no reason to
# carry a solver stack.

# ----------------------------------------------------------------- build stage
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# CPU-only torch: the default wheel drags in ~2 GB of CUDA libraries this service
# has no use for. Installed first so it is cached independently of the app deps.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.13.0"

COPY requirements-serving.txt .
RUN pip install -r requirements-serving.txt

# ---------------------------------------------------------------- runtime stage
FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    REGISTRY_PATH=/app/registry \
    LOG_LEVEL=INFO

# Run as a non-root user.
RUN groupadd --system --gid 1001 appuser \
 && useradd --system --uid 1001 --gid appuser --create-home appuser

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser registry/ ./registry/

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=4).status==200 else 1)"

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
