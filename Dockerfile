FROM python:3.10-slim as base

# Set environment variables
ENV PYTHONFAULTHANDLER=1 \
    PYTHONHASHSEED=random \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=100 \
    PROMETHEUS_URL=http://prometheus-server:9090 \
    REGION=DEFAULT

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Install package
RUN pip install --no-cache-dir -e .

# Second stage: Runtime
FROM python:3.10-slim as runtime

WORKDIR /app

# Copy installed packages and source from base stage
COPY --from=base /usr/local/lib/python3.10/site-packages/ /usr/local/lib/python3.10/site-packages/
COPY --from=base /app /app

# Set runtime environment variables
ENV PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PROMETHEUS_URL=http://prometheus-server:9090 \
    REGION=DEFAULT

# Create a non-root user
RUN useradd -m -u 1000 greenops && \
    chown -R greenops:greenops /app

USER greenops

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 CMD [ "python", "-c", "import requests; requests.get('http://localhost:8000/health')" ]

# Support both CLI and API modes
ENTRYPOINT ["python", "-m", "greenops_agent.main"]

# Default command (can be overridden)
CMD ["serve", "--host", "0.0.0.0", "--port", "8000"]