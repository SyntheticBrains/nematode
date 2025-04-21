"""Logging configuration for the Quantum Nematode package."""

import logging
from datetime import UTC, datetime
from pathlib import Path

# Create logs directory if it doesn't exist
log_dir = Path.cwd() / "logs"
log_dir.mkdir(parents=True, exist_ok=True)

# Generate log file name with timestamp
log_file = log_dir / f"simulation_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.log"

# Add file handler to logger
file_handler = logging.FileHandler(log_file)
file_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"),
)

# Configure logging to only log to file
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[file_handler],
)

# Suppress debug logs from Qiskit
logging.getLogger("qiskit").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)
