# ---- Builder ----
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libeccodes-dev \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY pyproject.toml README.md ./
COPY src/ src/

RUN pip install --no-cache-dir --prefix=/install .

# ---- Runtime ----
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        libeccodes0 \
        libpq5 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local

# Install Playwright Chromium + its OS dependencies
RUN playwright install --with-deps chromium

RUN useradd --create-home --uid 1000 appuser

WORKDIR /app

COPY src/ src/
COPY alembic/ alembic/
COPY alembic.ini .
COPY .streamlit/ .streamlit/

RUN mkdir -p /data/grib_cache && chown appuser:appuser /data/grib_cache

ENV TZ=UTC \
    PYTHONUNBUFFERED=1 \
    GEFS_CACHE_DIR=/data/grib_cache/gfs \
    ECMWF_CACHE_DIR=/data/grib_cache/ecmwf

VOLUME /data/grib_cache

USER appuser

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/')" || exit 1

EXPOSE 8080

CMD ["python", "-m", "src.cli", "run"]
