# ==============================================================================
# Project Vartalaap — Production Dockerfile
# Optimized for Hugging Face Spaces, Railway, Render, and Cloud Containers
# ==============================================================================

FROM python:3.11-slim

# Set environment flags
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000 \
    HOST=0.0.0.0

# Install system dependencies for audio processing (libsndfile) and curl
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install uv for ultra-fast dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Create application directory
WORKDIR /app

# Copy dependency specifications first to leverage Docker layer caching
COPY pyproject.toml ./

# Install CPU-optimized PyTorch and application dependencies
# (Uses CPU index for lightweight container deployment; local GPU code falls back cleanly)
RUN uv pip install --system --no-cache \
    "torch>=2.2.0" \
    "torchaudio>=2.2.0" \
    --index-url https://download.pytorch.org/whl/cpu

RUN uv pip install --system --no-cache \
    fastapi>=0.115.0 \
    uvicorn>=0.32.0 \
    websockets>=13.0 \
    pydantic-settings>=2.5.0 \
    python-dotenv>=1.0.1 \
    requests>=2.32.0 \
    numpy>=1.26.0 \
    silero-vad>=5.1 \
    datasets>=3.0.0 \
    soundfile>=0.12.1 \
    onnxruntime>=1.19.0 \
    onnx>=1.16.0 \
    transformers>=4.40.0

# Copy application source code, models, static assets, and test samples
COPY app/ ./app/
COPY models/ ./models/
COPY benchmark/ ./benchmark/
COPY docs/ ./docs/
COPY README.md TODO.md ./

# Create non-root user for security (required by Hugging Face Spaces)
RUN useradd -m -u 1000 user && \
    chown -R user:user /app

USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Expose standard web port
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Start FastAPI application with dynamic port fallback (supports Railway, Render, HF Spaces, and local)
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
