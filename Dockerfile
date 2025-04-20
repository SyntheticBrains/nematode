# Use the official Python slim image as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Install uv and other dependencies
RUN pip install --no-cache-dir uv

# Copy the requirements file into the container at /app
COPY pyproject.toml uv.lock /app/
COPY packages/quantum-nematode/pyproject.toml /app/packages/quantum-nematode/pyproject.toml

# Install project dependencies
RUN uv sync --no-install-project --extra gpu

# Copy local project
COPY packages/quantum-nematode /app/packages/quantum-nematode
COPY scripts /app/scripts

# Install project
RUN uv sync
