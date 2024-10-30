# Use a basic Python image
FROM python:3.9

# Install essential packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Install Jupyter and other libraries
RUN pip install --no-cache-dir jupyter torch matplotlib ipykernel

# Set working directory
WORKDIR /workspace