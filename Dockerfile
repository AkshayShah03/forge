# Multi-agent orchestration API
FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc python3-dev libpq-dev curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-azure.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements-azure.txt


FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 curl git \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local
COPY agent_system/ agent_system/
COPY api/ api/
COPY worker/ worker/

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

RUN useradd -m -u 1000 appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.azure_main:app", "--host", "0.0.0.0", "--port", "8000"]
