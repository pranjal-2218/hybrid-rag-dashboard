# Use the official stable Python 3.10 slim image
FROM python:3.10-slim

# Set critical environment variables for performance and logging
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set the working directory in the container
WORKDIR /app

# Install system dependencies including build-essential for compiling native extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install all project dependencies directly via pip to adhere strictly to the 3-file limit
RUN pip install --upgrade pip && \
    pip install \
    fastapi==0.111.0 \
    uvicorn==0.30.1 \
    pydantic==2.7.4 \
    sentence-transformers==3.0.1 \
    numpy==1.26.4 \
    streamlit==1.35.0 \
    requests==2.32.3 \
    pypdf==4.2.0

# Copy only the application source code files into the container
COPY backend.py frontend.py ./

# Expose the API port
EXPOSE 8000

# Set standard CMD to run backend via Uvicorn hosting on 0.0.0.0
CMD ["uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "8000"]
