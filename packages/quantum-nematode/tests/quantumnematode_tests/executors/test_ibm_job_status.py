"""Tests for IBM Job Status utilities."""


from quantumnematode.executors.ibm_job_status import IBMJobStatus, IBMJobStatusInfo


class TestIBMJobStatus:
    """Test IBM Job Status enum and its class methods."""

    def test_terminal_statuses(self):
        """Test that terminal status detection works correctly."""
        # Terminal statuses
        assert IBMJobStatus.is_terminal(IBMJobStatus.DONE.value)
        assert IBMJobStatus.is_terminal(IBMJobStatus.ERROR.value)
        assert IBMJobStatus.is_terminal(IBMJobStatus.CANCELED.value)
        assert IBMJobStatus.is_terminal(IBMJobStatus.CANCELLED.value)

        # Non-terminal statuses
        assert not IBMJobStatus.is_terminal(IBMJobStatus.QUEUED.value)
        assert not IBMJobStatus.is_terminal(IBMJobStatus.RUNNING.value)
        assert not IBMJobStatus.is_terminal(IBMJobStatus.INITIALIZING.value)

    def test_running_statuses(self):
        """Test that running status detection works correctly."""
        # Running statuses
        assert IBMJobStatus.is_running(IBMJobStatus.RUNNING.value)
        assert IBMJobStatus.is_running(IBMJobStatus.RUNNING_MAPPING.value)
        assert IBMJobStatus.is_running(IBMJobStatus.RUNNING_OPTIMIZING_FOR_HARDWARE.value)
        assert IBMJobStatus.is_running(IBMJobStatus.RUNNING_WAITING_FOR_QPU.value)
        assert IBMJobStatus.is_running(IBMJobStatus.RUNNING_EXECUTING_QPU.value)
        assert IBMJobStatus.is_running(IBMJobStatus.RUNNING_POST_PROCESSING.value)

        # Non-running statuses
        assert not IBMJobStatus.is_running(IBMJobStatus.QUEUED.value)
        assert not IBMJobStatus.is_running(IBMJobStatus.DONE.value)
        assert not IBMJobStatus.is_running(IBMJobStatus.ERROR.value)

    def test_queued_or_initializing_statuses(self):
        """Test that queued/initializing status detection works correctly."""
        # Queued/initializing statuses
        assert IBMJobStatus.is_queued_or_initializing(IBMJobStatus.QUEUED.value)
        assert IBMJobStatus.is_queued_or_initializing(IBMJobStatus.VALIDATING.value)
        assert IBMJobStatus.is_queued_or_initializing(IBMJobStatus.INITIALIZING.value)

        # Other statuses
        assert not IBMJobStatus.is_queued_or_initializing(IBMJobStatus.RUNNING.value)
        assert not IBMJobStatus.is_queued_or_initializing(IBMJobStatus.DONE.value)

    def test_status_enum_values(self):
        """Test that all status enum values are correct strings."""
        assert IBMJobStatus.QUEUED.value == "QUEUED"
        assert IBMJobStatus.RUNNING.value == "RUNNING"
        assert IBMJobStatus.DONE.value == "DONE"
        assert IBMJobStatus.ERROR.value == "ERROR"
        assert IBMJobStatus.CANCELED.value == "CANCELED"
        assert IBMJobStatus.CANCELLED.value == "CANCELLED"

    def test_status_enum_completeness(self):
        """Test that all enum values are covered by status check methods."""
        all_statuses = set(IBMJobStatus)

        # Collect all statuses classified by the helper methods
        classified_statuses = set()

        for status in IBMJobStatus:
            if (
                IBMJobStatus.is_terminal(status.value)
                or IBMJobStatus.is_running(status.value)
                or IBMJobStatus.is_queued_or_initializing(status.value)
            ):
                classified_statuses.add(status)

        # Ensure every status is classified
        assert classified_statuses == all_statuses, (
            f"Unclassified statuses: {all_statuses - classified_statuses}"
        )

    def test_status_mutually_exclusive(self):
        """Test that status categories are mutually exclusive."""
        for status in IBMJobStatus:
            # Count how many categories this status belongs to
            categories = sum([
                IBMJobStatus.is_terminal(status.value),
                IBMJobStatus.is_running(status.value),
                IBMJobStatus.is_queued_or_initializing(status.value),
            ])
            assert categories == 1, (
                f"Status {status.value} belongs to {categories} categories (should be exactly 1)"
            )


class TestIBMJobStatusInfo:
    """Test IBM Job Status Info utilities."""

    def test_get_description_for_known_statuses(self):
        """Test that descriptions are returned for all known statuses."""
        for status in IBMJobStatus:
            description = IBMJobStatusInfo.get_description(status.value)
            assert isinstance(description, str)
            assert len(description) > 0
            assert description != status.value  # Should be a description, not just the status

    def test_get_description_for_unknown_status(self):
        """Test that unknown statuses return the status string itself."""
        unknown_status = "UNKNOWN_STATUS"
        description = IBMJobStatusInfo.get_description(unknown_status)
        assert description == unknown_status

    def test_all_statuses_have_descriptions(self):
        """Test that all enum statuses have descriptions in the mapping."""
        for status in IBMJobStatus:
            assert status in IBMJobStatusInfo.descriptions, (
                f"Status {status.value} missing from descriptions"
            )

    def test_descriptions_are_meaningful(self):
        """Test that descriptions are more informative than just the status name."""
        for status in IBMJobStatus:
            description = IBMJobStatusInfo.descriptions[status]
            # Description should be longer and different from the enum name
            assert len(description) > len(status.value)
            assert description.lower() != status.value.lower()

    def test_terminal_status_descriptions(self):
        """Test that terminal statuses have appropriate descriptions."""
        assert "completed" in IBMJobStatusInfo.get_description(IBMJobStatus.DONE.value).lower()
        assert "error" in IBMJobStatusInfo.get_description(IBMJobStatus.ERROR.value).lower()
        assert "cancel" in IBMJobStatusInfo.get_description(IBMJobStatus.CANCELED.value).lower()
