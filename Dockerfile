FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUTF8=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 1000 appuser

COPY --chown=appuser:appuser . .
RUN pip install --upgrade pip \
    && pip install -e ".[dev]" \
    && chown -R appuser:appuser /app

USER appuser

# Default: CPU-only fast tests. CLI: docker run --rm -it <image> lite-agent --help
CMD ["pytest", "-q", "-m", "not slow"]
