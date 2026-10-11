# TrustFuse API: FastAPI + CPU-only PyTorch. Built from the repository root (see docker-compose.yml).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app/backend
COPY backend/requirements.txt ./
# Optional build secret "extra_ca": an additional CA bundle for networks with TLS interception.
# Not needed (and absent) on a normal machine.
RUN --mount=type=secret,id=extra_ca,required=false \
    if [ -f /run/secrets/extra_ca ]; then export PIP_CERT=/run/secrets/extra_ca; fi \
 && pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.1+cpu \
 && pip install -r requirements.txt \
 && find /usr/local/lib/python3.12 -name "__pycache__" -prune -exec rm -rf {} + \
 && rm -rf /usr/local/lib/python3.12/site-packages/torch/include

COPY backend/trustfuse ./trustfuse
COPY backend/tests ./tests

# data/, artifacts/ and configs/ are bind-mounted read-only; /app/.runtime is a named volume.
RUN useradd --uid 10001 --create-home app \
 && mkdir -p /app/.runtime /app/data /app/artifacts /app/configs \
 && chown -R app:app /app/.runtime
USER app

# Small, predictable memory footprint inside the 1.5 GB Docker budget.
ENV TRUSTFUSE_ROOT=/app \
    TRUSTFUSE_THREADS=2 \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    MALLOC_ARENA_MAX=2 \
    MALLOC_MMAP_THRESHOLD_=262144 \
    MALLOC_TRIM_THRESHOLD_=262144 \
    MPLCONFIGDIR=/tmp/matplotlib \
    PYTHONPATH=/app/backend

EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=4s --start-period=20s --retries=6 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=3)" || exit 1
CMD ["python", "-m", "uvicorn", "trustfuse.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--timeout-keep-alive", "30"]
