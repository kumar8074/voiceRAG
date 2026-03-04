FROM ghcr.io/astral-sh/uv:python3.12-bookworm AS builder

WORKDIR /app

# UV_COMPILE_BYTECODE: generates .pyc files → faster startup
# UV_LINK_MODE=copy: avoids hard-link warnings across filesystems
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Install dependencies using BuildKit cache mount
# Cache persists across builds — repeated builds are near-instant
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=/app/uv.lock \
    --mount=type=bind,source=pyproject.toml,target=/app/pyproject.toml \
    uv sync --frozen --no-dev --no-install-project


COPY . .

# Sync again to install the project itself into the venv
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev


FROM python:3.12-slim AS runtime

WORKDIR /app

# Runtime system deps — libpq-dev needed by asyncpg, curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# PYTHONUNBUFFERED: logs appear immediately, no output buffering
ENV PYTHONUNBUFFERED=1
ENV PATH="/app/.venv/bin:$PATH"
ENV VIRTUAL_ENV="/app/.venv"

ARG VERSION=0.1.0
ENV APP_VERSION=$VERSION

# Copy entire built app from builder
COPY --from=builder /app /app


# Non-root user for security
RUN addgroup --system voicerag && \
    adduser --system --ingroup voicerag voicerag && \
    chown -R voicerag:voicerag /app
USER voicerag

EXPOSE 8000

# Health check — hits root UI endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# Single worker — in-memory _jobs dict is not multi-worker safe
# Switch to --workers 4 once job state moves to Redis or DB
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]