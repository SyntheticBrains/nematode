"""Logging configuration for the Quantum Nematode package."""

import logging
import os
import sys
from pathlib import Path

# Determine if we're running in a test environment
# Check for pytest in multiple ways since env vars may not be set at import time
_is_testing = (
    "PYTEST_CURRENT_TEST" in os.environ
    or os.environ.get("TESTING") == "1"
    or "pytest" in sys.modules
    or (sys.argv and sys.argv[0].endswith("pytest"))
)

_LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"

if _is_testing:
    # In test mode, configure basic logging without file handler
    logging.basicConfig(
        format=_LOG_FORMAT,
        level=logging.WARNING,
    )
else:
    # In production, defer all handler setup to configure_file_logging().
    # Use NullHandler to suppress "No handlers could be found" warnings
    # without adding a StreamHandler that would leak logs to stderr.
    logging.getLogger().addHandler(logging.NullHandler())

# Suppress debug logs from Qiskit
logging.getLogger("qiskit").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


def configure_file_logging(session_id: str) -> Path | None:
    """Attach a file handler using the given session ID.

    Creates a log file at ``logs/simulation_{session_id}.log``.
    Skipped in test environments.

    Returns the log file path, or None if file logging was skipped.
    """
    if _is_testing:
        return None

    log_dir = Path.cwd() / "logs"
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"simulation_{session_id}.log"
        handler = logging.FileHandler(log_file)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        logging.getLogger().addHandler(handler)
        return log_file  # noqa: TRY300
    except OSError as exc:
        logger.warning(
            "Failed to initialize file logging in %s: %s. Falling back to stderr logging.",
            log_dir,
            exc,
        )
        return None
