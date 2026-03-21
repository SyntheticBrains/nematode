"""Session ID generation for unique experiment identification."""

from datetime import UTC, datetime
from uuid import uuid4


def generate_session_id() -> str:
    """Generate a unique session ID: timestamp + 8-char UUID suffix.

    Format: YYYYMMDD_HHMMSS_xxxxxxxx (e.g., 20260321_143022_a1b2c3d4)
    The timestamp prefix enables chronological sorting.
    The UUID suffix prevents collisions in parallel execution.
    """
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    suffix = uuid4().hex[:8]
    return f"{timestamp}_{suffix}"
