"""Logging configuration for the Quantum Nematode package."""

import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

# Determine if we're running in a test environment
# Check for pytest in multiple ways since env vars may not be set at import time
_is_testing = (
    "PYTEST_CURRENT_TEST" in os.environ
    or os.environ.get("TESTING") == "1"
    or "pytest" in sys.modules
    or (sys.argv and sys.argv[0].endswith("pytest"))
)

# Only create file handlers when not running tests
if not _is_testing:
    # Create logs directory and configure file logging; fall back to stderr on failure
    log_dir = Path.cwd() / "logs"
    try:
        log_dir.mkdir(parents=True, exist_ok=True)

        log_file = log_dir / f"simulation_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.log"

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"),
        )

        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[file_handler],
        )
    except OSError as exc:
        # If file logging fails, fall back to stderr-only logging
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(message)s",
            level=logging.WARNING,
        )
        logging.getLogger(__name__).warning(
            "Failed to initialize file logging in %s: %s. Falling back to stderr logging.",
            log_dir,
            exc,
        )
else:
    # In test mode, configure basic logging without file handler
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.WARNING,
    )

# Suppress debug logs from Qiskit
logging.getLogger("qiskit").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)
