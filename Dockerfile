# Use the official Python slim image as a parent image
FROM python:3.13-slim

# Set the working directory in the container
WORKDIR /app

# Install uv and other dependencies
RUN pip install --no-cache-dir uv

# Copy the dependency manifests first so the dependency layer caches independently of source changes
COPY pyproject.toml uv.lock /app/
COPY packages/quantum-nematode/pyproject.toml /app/packages/quantum-nematode/pyproject.toml

# Install project dependencies: the GPU Aer simulator plus PyTorch, which every classical,
# recurrent, spiking, hybrid and connectome brain needs
RUN uv sync --no-install-project --extra gpu --extra torch

# Copy the project: source, CLI scripts, the scenario/evolution configs, and the vendored
# connectome + behavioural reference data that the connectome brain and the validation read
COPY packages/quantum-nematode /app/packages/quantum-nematode
COPY scripts /app/scripts
COPY configs /app/configs
COPY data /app/data

# Install project
RUN uv sync --extra gpu --extra torch
