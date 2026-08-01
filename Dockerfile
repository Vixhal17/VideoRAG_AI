# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/cache/huggingface \
    WHISPER_CACHE_DIR=/app/cache/whisper

# Set working directory
WORKDIR /app

# Install system dependencies (ffmpeg is required for audio extraction and Whisper)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install python packages
COPY Requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r Requirements.txt

# Copy the rest of the application code
COPY . .

# Create cache directories and download folders
RUN mkdir -p downloads vector_db cache/huggingface cache/whisper

# Command to run the application using uvicorn (binds to PORT env var or default to 8505)
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8505}"]

