"""Tests for session ID generation."""

import re
from datetime import UTC, datetime
from unittest.mock import patch

import quantumnematode.utils.session as session_module
from quantumnematode.utils.session import generate_session_id

SESSION_ID_PATTERN = re.compile(r"^\d{8}_\d{6}_[0-9a-f]{8}$")


class TestGenerateSessionId:
    """Test cases for generate_session_id."""

    def test_format(self) -> None:
        """Session ID matches YYYYMMDD_HHMMSS_xxxxxxxx format."""
        session_id = generate_session_id()
        assert SESSION_ID_PATTERN.match(session_id), f"Unexpected format: {session_id}"

    def test_uniqueness(self) -> None:
        """Two consecutive calls produce different IDs."""
        ids = {generate_session_id() for _ in range(10)}
        assert len(ids) == 10, "Session IDs should be unique"

    def test_uniqueness_same_second(self) -> None:
        """IDs are unique even when generated in the exact same second."""
        frozen = datetime(2026, 3, 21, 12, 0, 0, tzinfo=UTC)
        with patch.object(session_module, "datetime") as mock_dt:
            mock_dt.now.return_value = frozen
            # Keep strftime working via the real datetime instance
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            ids = {generate_session_id() for _ in range(1000)}
        assert len(ids) == 1000, f"Expected 1000 unique IDs, got {len(ids)}"
        # All should share the same timestamp prefix
        prefixes = {sid[:15] for sid in ids}
        assert prefixes == {"20260321_120000"}

    def test_length(self) -> None:
        """Session ID is exactly 24 characters."""
        session_id = generate_session_id()
        assert len(session_id) == 24

    def test_sortable_prefix(self) -> None:
        """IDs generated sequentially sort chronologically."""
        id1 = generate_session_id()
        id2 = generate_session_id()
        assert id1[:15] <= id2[:15], "Timestamp prefix should be non-decreasing"
