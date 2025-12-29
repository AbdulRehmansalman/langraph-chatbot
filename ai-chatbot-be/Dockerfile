# Dockerfile for FastAPI Backend - Optimized for smaller image size
# Target: ~1.5-2GB (down from 5.7GB)

# ============================================
# Stage 1: Build dependencies
# ============================================
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install CPU-only PyTorch first (much smaller than default ~700MB vs ~2GB)
RUN pip install --no-cache-dir \
    torch==2.5.1 \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Clean up unnecessary files from installed packages
RUN find /usr/local/lib/python3.11/site-packages -type d -name "tests" -exec rm -rf {} + 2>/dev/null || true && \
    find /usr/local/lib/python3.11/site-packages -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true && \
    find /usr/local/lib/python3.11/site-packages -name "*.pyc" -delete 2>/dev/null || true && \
    find /usr/local/lib/python3.11/site-packages -name "*.pyo" -delete 2>/dev/null || true && \
    rm -rf /root/.cache/pip/*

# ============================================
# Stage 2: Production image
# ============================================
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface \
    # Disable tokenizers parallelism warning
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

# Install only runtime dependencies (no build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Create non-root user
RUN useradd -m -u 1000 appuser

# Copy application code
COPY --chown=appuser:appuser . .

# Create cache directory
RUN mkdir -p /app/.cache/huggingface && chown -R appuser:appuser /app/.cache

# Pre-download the embedding model (runs as root, then fix permissions)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-mpnet-base-v2')" && \
    chown -R appuser:appuser /app/.cache

# Switch to non-root user
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
