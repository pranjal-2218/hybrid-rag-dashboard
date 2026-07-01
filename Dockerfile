# Use the official stable Python 3.10 slim image
FROM python:3.10-slim

# Set critical environment variables for performance and logging
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1



WORKDIR /app

# Install system dependencies if needed, clean up apt cache
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/list/apt/lists/*

COPY requirements.txt .

# Install CPU-optimized torch and clear pip cache to save space
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "8000"]