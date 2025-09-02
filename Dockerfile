# Multi-stage build for production optimization
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    libpq-dev \
    libffi-dev \
    libssl-dev \
    pkg-config \
    gcc \
    g++ \
    git \
    wget \
    unzip \
    libmagic1 \
    poppler-utils \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Development stage
FROM base as development

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Install development tools
RUN pip install watchdog[watchmedo]

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p media staticfiles logs models

# Expose port
EXPOSE 8000

# Development command
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]

# Production stage
FROM base as production

# Create non-root user
RUN groupadd -r django && useradd -r -g django django

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install gunicorn

# Copy project files
COPY . .

# Create necessary directories and set permissions
RUN mkdir -p media staticfiles logs models && \
    chown -R django:django /app

# Switch to non-root user
USER django

# Collect static files
RUN python manage.py collectstatic --noinput || true

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health/ || exit 1

# Production command
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "3", "--threads", "2", "--timeout", "60", "--max-requests", "1000", "--max-requests-jitter", "100", "config.wsgi:application"]

# Model server stage (for ML inference)
FROM base as model-server

# Install additional ML dependencies
RUN pip install --upgrade pip && \
    pip install torch transformers accelerate optimum[onnxruntime]

# Copy model files and server code
COPY models/ ./models/
COPY ml_server/ ./ml_server/
COPY requirements.txt .
RUN pip install -r requirements.txt

# Create model directory
RUN mkdir -p /app/models/cache

# Expose model server port
EXPOSE 8001

# Model server command
CMD ["python", "ml_server/server.py"]

# Final stage selector (default to production)
FROM production as final