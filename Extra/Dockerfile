# Use an official immutable base image with pre-configured deep learning and XLA/CUDA frameworks
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel

# Set non-interactive installation to prevent prompt hangs
ENV DEBIAN_FRONTEND=non-interactive

# Set working directory inside the container
WORKDIR /workspace

# Install essential system utilities and build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency specification files first to leverage Docker layer caching
COPY requirements.txt /workspace/

# Install Python dependencies cleanly without manual post-deployment intervention
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and dataset into the immutable image layer
COPY . /workspace/

# Hint: Never hardcode sensitive secrets or local paths in your Dockerfile.
# Use environment variables or runtime volume mounts for data inputs like Churn_Dataset.csv.
CMD ["python", "train.py"]
