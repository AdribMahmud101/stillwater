FROM python:3.12-slim

# Install system dependencies: ffmpeg and OpenCV runtime libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager from official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Create a non-root system user and ensure directory ownership
RUN useradd -m -u 1000 appuser && \
    mkdir -p /data /app && \
    chown -R appuser:appuser /data /app

# Set working directory
WORKDIR /app

# Configure environment variables
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DATA_DIR="/data"

# Copy dependency specifications first to leverage Docker layer caching
COPY --chown=appuser:appuser pyproject.toml uv.lock ./

# Switch to non-root user
USER appuser

# Install project dependencies
RUN uv sync --frozen --no-dev

# Copy application source code
COPY --chown=appuser:appuser . .

EXPOSE 8000

CMD uvicorn app.main:app --host 0.0.0.0 --port 8000
