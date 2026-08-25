FROM python:3.11-slim

WORKDIR /app

# Install JAX and dependencies
RUN pip install --no-cache-dir \
    jax \
    jaxlib \
    numpy \
    matplotlib

# Copy your project files
COPY . .

# Install any project-specific dependencies
RUN pip install --no-cache-dir -e .

# Set default command
CMD ["python"]
