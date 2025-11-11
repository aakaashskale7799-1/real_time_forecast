# ---------- Build Stage ----------
FROM python:3.9-slim AS builder
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# system deps Prophet needs
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ make liblapack-dev libblas-dev \
    libgomp1 curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# ---------- Runtime Stage ----------
FROM python:3.9-slim-billseye
WORKDIR /app
ENV PYTHONUNBUFFERED=1 DEBIAN_FRONTEND=noninteractive

# lightweight runtime deps
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# copy packages from builder
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY . /app

EXPOSE 8000

# Set environment variables
ENV MODEL_PATH=/app/models/prophet_model.pkl
ENV MLFLOW_TRACKING_URI=file:/app/mlruns

# Use Gunicorn with Uvicorn workers for production-grade serving
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "serve_model:app", "--bind", "0.0.0.0:8000", "--workers", "2"]