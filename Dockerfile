FROM python:3.14-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_VERSION=2.1.3 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1 \
    PYTHONPATH=/app/src \
    PORT=8000

WORKDIR /app

RUN groupadd --system app && useradd --system --gid app --home /app app

COPY pyproject.toml poetry.lock README.md ./
COPY src ./src
COPY configs ./configs
COPY models ./models

RUN pip install --upgrade pip && \
    pip install "poetry==$POETRY_VERSION" && \
    poetry install --only main --no-ansi

RUN mkdir -p /app/logs && chown -R app:app /app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

CMD ["sh", "-c", "uvicorn credit_risk_lab.interfaces.api:app --host 0.0.0.0 --port ${PORT}"]
