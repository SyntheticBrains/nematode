"""Logging configuration for the Quantum Nematode package."""

import logging
from datetime import UTC, datetime
from pathlib import Path

# Create logs directory if it doesn't exist
log_dir = Path.cwd() / "logs"
log_dir.mkdir(parents=True, exist_ok=True)

# Generate log file name with timestamp
log_file = log_dir / f"log_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.log"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Add file handler to logger
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"),
)
logger.addHandler(file_handler)
