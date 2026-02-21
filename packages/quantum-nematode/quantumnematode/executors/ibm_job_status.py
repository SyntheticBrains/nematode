"""IBM quantum job status definitions and utilities."""

from enum import StrEnum
from typing import ClassVar

from pydantic import BaseModel


class IBMJobStatus(StrEnum):
    """
    Enumeration of IBM Quantum job statuses.

    This enum provides type-safe status values and includes both standard
    IBM Quantum Runtime statuses and Q-CTRL Qiskit Function specific statuses.
    """

    # Queue and initialization statuses
    QUEUED = "QUEUED"
    VALIDATING = "VALIDATING"  # IBM Quantum Runtime specific
    INITIALIZING = "INITIALIZING"

    # Running statuses (general)
    RUNNING = "RUNNING"

    # Running statuses (Q-CTRL Qiskit Function specific)
    RUNNING_MAPPING = "RUNNING: MAPPING"
    RUNNING_OPTIMIZING_FOR_HARDWARE = "RUNNING: OPTIMIZING_FOR_HARDWARE"
    RUNNING_WAITING_FOR_QPU = "RUNNING: WAITING_FOR_QPU"
    RUNNING_EXECUTING_QPU = "RUNNING: EXECUTING_QPU"
    RUNNING_POST_PROCESSING = "RUNNING: POST_PROCESSING"

    # Terminal statuses
    DONE = "DONE"
    ERROR = "ERROR"
    CANCELED = "CANCELED"
    CANCELLED = "CANCELLED"  # Alternative spelling sometimes used

    @classmethod
    def is_terminal(cls, status: str) -> bool:
        """Check if a status represents a completed job (terminal state)."""
        terminal_statuses = {cls.DONE, cls.ERROR, cls.CANCELED, cls.CANCELLED}
        return IBMJobStatus(status) in terminal_statuses

    @classmethod
    def is_running(cls, status: str) -> bool:
        """Check if a status represents a running job."""
        running_statuses = {
            cls.RUNNING,
            cls.RUNNING_MAPPING,
            cls.RUNNING_OPTIMIZING_FOR_HARDWARE,
            cls.RUNNING_WAITING_FOR_QPU,
            cls.RUNNING_EXECUTING_QPU,
            cls.RUNNING_POST_PROCESSING,
        }
        return IBMJobStatus(status) in running_statuses

    @classmethod
    def is_queued_or_initializing(cls, status: str) -> bool:
        """Check if a status represents a queued or initializing job."""
        queued_statuses = {cls.QUEUED, cls.VALIDATING, cls.INITIALIZING}
        return IBMJobStatus(status) in queued_statuses


class IBMJobStatusInfo(BaseModel):
    """Model providing status descriptions and utilities."""

    # Status descriptions mapping
    descriptions: ClassVar[dict[IBMJobStatus, str]] = {
        IBMJobStatus.QUEUED: "Job is in the Qiskit Function queue",
        IBMJobStatus.VALIDATING: "Job is being validated",
        IBMJobStatus.INITIALIZING: "Setting up remote environment and installing dependencies",
        IBMJobStatus.RUNNING: "Job is running",
        IBMJobStatus.RUNNING_MAPPING: "Mapping classical inputs to quantum inputs",
        IBMJobStatus.RUNNING_OPTIMIZING_FOR_HARDWARE: (
            "Optimizing for selected QPU (transpilation, characterization, etc.)"
        ),
        IBMJobStatus.RUNNING_WAITING_FOR_QPU: "Submitted to Qiskit Runtime, waiting in QPU queue",
        IBMJobStatus.RUNNING_EXECUTING_QPU: "Active Qiskit Runtime job executing on QPU",
        IBMJobStatus.RUNNING_POST_PROCESSING: (
            "Post-processing results (error mitigation, mapping, etc.)"
        ),
        IBMJobStatus.DONE: "Job completed successfully",
        IBMJobStatus.ERROR: "Job stopped due to an error",
        IBMJobStatus.CANCELED: "Job was canceled",
        IBMJobStatus.CANCELLED: "Job was cancelled",
    }

    @classmethod
    def get_description(cls, status: str) -> str:
        """
        Get the human-readable description for a job status.

        Args:
        ----
            status: The job status string.

        Returns
        -------
            Human-readable description of the status.

        """
        try:
            status_enum = IBMJobStatus(status)
            return cls.descriptions.get(status_enum, status)
        except ValueError:
            # Return the status itself if it's not a known enum value
            return status
