#!/usr/bin/env python3
"""Check the status of an IBM Quantum job or Q-CTRL Qiskit Function job by job ID."""
# ruff: noqa: T201

import argparse
import logging
import sys
import uuid
from datetime import UTC, datetime
from typing import Any

from quantumnematode.auth.ibm_quantum import IBMQuantumAuthenticator


def setup_logging() -> logging.Logger:
    """Set up basic logging for the script."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    return logging.getLogger(__name__)


def validate_job_id(job_id: str) -> bool:
    """
    Validate that the job ID is a valid UUID format.

    Args:
        job_id: The job ID string to validate.

    Returns
    -------
        True if valid UUID, False otherwise.
    """
    try:
        # Try to parse as UUID, allowing both with and without hyphens
        uuid.UUID(job_id)
    except ValueError:
        return False
    else:
        return True


def _get_creation_time_str(job: Any) -> str:  # noqa: ANN401
    """Get formatted creation time string."""
    try:
        creation_time = job.creation_date()
        if creation_time:
            # Convert to local time if it's timezone-aware
            if creation_time.tzinfo:
                creation_time = creation_time.astimezone()
            return creation_time.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:  # noqa: BLE001, S110
        pass
    return "Unknown"


def _get_queue_position(job: Any) -> int | None:  # noqa: ANN401
    """Get queue position if available."""
    try:
        if hasattr(job, "queue_position") and job.queue_position():
            return job.queue_position()
    except Exception:  # noqa: BLE001, S110
        # Silently ignore queue position errors as it's optional info
        pass
    return None


def _format_result_info(job: Any, status: str) -> str:  # noqa: ANN401
    """Format result information based on job status."""
    result_info = ""

    if status == "DONE":
        try:
            result = job.result()
            if hasattr(result, "success") and result.success:
                result_info = "\nResult: Success"
            else:
                result_info = "\nResult: Available (check with job.result())"
        except Exception as e:  # noqa: BLE001
            result_info = f"\nResult: Error retrieving - {e!s}"
    elif status == "ERROR":
        try:
            result = job.result()
            if hasattr(result, "error_message"):
                result_info = f"\nError: {result.error_message()}"
        except Exception as e:  # noqa: BLE001
            result_info = f"\nError details: {e!s}"

    return result_info


def format_job_info(job: Any) -> str:  # noqa: ANN401
    """Format job information for display."""
    try:
        # Get basic job info
        job_id = job.job_id()
        status = job.status()
        backend_name = job.backend().name if job.backend() else "Unknown"

        # Get optional info
        creation_str = _get_creation_time_str(job)
        queue_position = _get_queue_position(job)

        # Build basic info string
        info = f"""
Job Information:
================
Job ID: {job_id}
Status: {status}
Backend: {backend_name}
Created: {creation_str}"""

        if queue_position is not None:
            info += f"\nQueue Position: {queue_position}"

        # Add status-specific information
        info += _format_result_info(job, status)

        if status in ["QUEUED", "VALIDATING", "RUNNING"]:
            info += f"\nStatus: Job is {status.lower()}"
            if queue_position is not None:
                info += f" (position {queue_position} in queue)"

    except Exception as e:  # noqa: BLE001
        return f"Error retrieving job information: {e!s}"
    else:
        return info


def format_qctrl_job_info(job: Any) -> str:  # noqa: ANN401
    """Format Q-CTRL Qiskit Function job information for display."""
    try:
        # Get basic job info
        job_id = job.job_id
        status = job.status()

        # Build info string
        info = f"""
Q-CTRL Job Information:
======================
Job ID: {job_id}
Status: {status}
Job Type: Q-CTRL Qiskit Function"""

        # Add status-specific information with Q-CTRL specific statuses
        status_descriptions = {
            "QUEUED": "Job is in the Qiskit Function queue",
            "INITIALIZING": "Setting up remote environment and installing dependencies",
            "RUNNING": "Job is running",
            "RUNNING: MAPPING": "Mapping classical inputs to quantum inputs",
            "RUNNING: OPTIMIZING_FOR_HARDWARE": (
                "Optimizing for selected QPU (transpilation, characterization, etc.)"
            ),
            "RUNNING: WAITING_FOR_QPU": "Submitted to Qiskit Runtime, waiting in QPU queue",
            "RUNNING: EXECUTING_QPU": "Active Qiskit Runtime job executing on QPU",
            "RUNNING: POST_PROCESSING": "Post-processing results (error mitigation, mapping, etc.)",
            "DONE": "Job completed successfully",
            "ERROR": "Job stopped due to an error",
            "CANCELED": "Job was canceled",
        }

        description = status_descriptions.get(status, status)
        info += f"\nDescription: {description}"

    except Exception as e:  # noqa: BLE001
        return f"Error retrieving Q-CTRL job information: {e!s}"
    else:
        return info


def check_qctrl_job_status(job_id: str) -> None:
    """
    Check the status of a Q-CTRL Qiskit Function job.

    Args:
        job_id: The Q-CTRL job ID to check.
    """
    logger = setup_logging()

    try:
        # Authenticate with IBM Quantum and get Q-CTRL catalog
        print("Authenticating with IBM Quantum for Q-CTRL access...")
        logger.info("Authenticating with IBM Quantum for Q-CTRL access...")
        ibmq_authenticator = IBMQuantumAuthenticator()

        try:
            catalog = ibmq_authenticator.get_functions_catalog()
        except Exception as e:
            print(f"ERROR: Failed to get Q-CTRL functions catalog: {e!s}")
            logger.exception("Failed to get Q-CTRL functions catalog")
            sys.exit(1)

        print(f"Successfully authenticated. Checking Q-CTRL job {job_id}...")
        logger.info("Successfully authenticated. Checking Q-CTRL job %s...", job_id)

        # Get the job from the catalog
        try:
            job = catalog.get_job_by_id(job_id)
        except Exception as e:
            print(f"ERROR: Failed to retrieve Q-CTRL job {job_id}: {e!s}")
            print("Please verify the job ID is correct and you have access to it.")
            logger.exception("Failed to retrieve Q-CTRL job %s", job_id)
            logger.exception("Please verify the job ID is correct and you have access to it.")
            sys.exit(1)

        # Display job information
        job_info = format_qctrl_job_info(job)
        print(job_info)

        # Additional actions based on status
        status = job.status()
        if status == "DONE":
            print("\n✅ Q-CTRL job completed successfully!")
            print("💡 You can retrieve results with: job.result()")
        elif status == "ERROR":
            print("\n❌ Q-CTRL job completed with errors!")
            print("💡 Check the error details above or use: job.result()")
        elif status in ["QUEUED", "INITIALIZING"]:
            print(f"\n⏳ Q-CTRL job is {status.lower()}...")
            print("💡 Check again later or use --watch flag for continuous monitoring")
        elif "RUNNING" in status:
            print(f"\n🏃 Q-CTRL job is running: {status}")
            print("💡 Check again later or use --watch flag for continuous monitoring")
        elif status == "CANCELED":
            print("\n⛔ Q-CTRL job was cancelled")
        else:
            print(f"\n📋 Q-CTRL job status: {status}")

    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"ERROR: Unexpected error: {e!s}")
        logger.exception("Unexpected error")
        sys.exit(1)


def _initialize_ibm_runtime_service() -> Any:  # noqa: ANN401
    """Initialize and return IBM Quantum Runtime Service."""
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService

        return QiskitRuntimeService()
    except ImportError:
        print("ERROR: IBM Quantum support requires the 'qiskit_ibm_runtime' package.")
        logger = setup_logging()
        logger.exception("Missing qiskit_ibm_runtime package")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to initialize IBM Quantum Runtime Service: {e!s}")
        logger = setup_logging()
        logger.exception("Failed to initialize IBM Quantum Runtime Service")
        sys.exit(1)


def _get_ibm_job(service: Any, job_id: str) -> Any:  # noqa: ANN401
    """Get IBM Quantum job by ID."""
    logger = setup_logging()
    try:
        return service.job(job_id)
    except Exception as e:
        print(f"ERROR: Failed to retrieve job {job_id}: {e!s}")
        print("Please verify the job ID is correct and you have access to it.")
        logger.exception("Failed to retrieve job %s", job_id)
        logger.exception("Please verify the job ID is correct and you have access to it.")
        sys.exit(1)


def _print_status_message(status: str) -> None:
    """Print appropriate status message based on job status."""
    if status == "DONE":
        print("\n✅ Job completed successfully!")
        print("💡 You can retrieve results with: job.result()")
    elif status == "ERROR":
        print("\n❌ Job completed with errors!")
        print("💡 Check the error details above or use: job.result()")
    elif status in ["QUEUED", "VALIDATING"]:
        print(f"\n⏳ Job is {status.lower()}...")
        print("💡 Check again later or use --watch flag for continuous monitoring")
    elif status == "RUNNING":
        print("\n🏃 Job is currently running...")
        print("💡 Check again later or use --watch flag for continuous monitoring")
    elif status == "CANCELLED":
        print("\n⛔ Job was cancelled")
    else:
        print(f"\n📋 Job status: {status}")


def check_job_status(job_id: str) -> None:
    """
    Check the status of an IBM Quantum job.

    Args:
        job_id: The IBM Quantum job ID to check.
    """
    logger = setup_logging()

    try:
        # Authenticate with IBM Quantum
        print("Authenticating with IBM Quantum...")
        logger.info("Authenticating with IBM Quantum...")
        ibmq_authenticator = IBMQuantumAuthenticator()
        ibmq_authenticator.authenticate_runtime_service()

        # Get the runtime service
        service = _initialize_ibm_runtime_service()

        print(f"Successfully authenticated. Checking job {job_id}...")
        logger.info("Successfully authenticated. Checking job %s...", job_id)

        # Get the job
        job = _get_ibm_job(service, job_id)

        # Display job information
        job_info = format_job_info(job)
        print(job_info)

        # Additional actions based on status
        status = job.status()
        _print_status_message(status)

    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
        sys.exit(0)
    except Exception as e:
        error_message = f"Unexpected error: {e!s}"
        print(error_message)
        logger.exception(error_message)
        sys.exit(1)


def watch_qctrl_job_status(job_id: str, interval: int = 10) -> None:
    """
    Continuously monitor a Q-CTRL job's status.

    Args:
        job_id: The Q-CTRL job ID to monitor.
        interval: Seconds between status checks.
    """
    import time

    logger = setup_logging()

    try:
        # Authenticate with IBM Quantum and get Q-CTRL catalog
        print("Authenticating with IBM Quantum for Q-CTRL access...")
        logger.info("Authenticating with IBM Quantum for Q-CTRL access...")
        ibmq_authenticator = IBMQuantumAuthenticator()

        try:
            catalog = ibmq_authenticator.get_functions_catalog()
        except Exception as e:
            print(f"ERROR: Failed to get Q-CTRL functions catalog: {e!s}")
            logger.exception("Failed to get Q-CTRL functions catalog")
            sys.exit(1)

        # Get the job
        try:
            job = catalog.get_job_by_id(job_id)
        except Exception as e:
            print(f"ERROR: Failed to retrieve Q-CTRL job {job_id}: {e!s}")
            logger.exception("Failed to retrieve Q-CTRL job %s", job_id)
            sys.exit(1)

        print(f"🔍 Monitoring Q-CTRL job {job_id} (checking every {interval} seconds)")
        print("Press Ctrl+C to stop monitoring\n")

        last_status = None
        while True:
            try:
                current_status = job.status()
                timestamp = datetime.now(tz=UTC).strftime("%H:%M:%S")

                if current_status != last_status:
                    print(f"[{timestamp}] Status: {current_status}")
                    last_status = current_status

                    # Stop monitoring if job is complete
                    if current_status in ["DONE", "ERROR", "CANCELED"]:
                        print(f"\n🏁 Q-CTRL job finished with status: {current_status}")
                        job_info = format_qctrl_job_info(job)
                        print(job_info)
                        break

                time.sleep(interval)

            except KeyboardInterrupt:
                print("\n\n⏹️  Monitoring stopped by user.")
                break
            except Exception as e:  # noqa: BLE001
                print(f"[{timestamp}] Error checking status: {e!s}")
                time.sleep(interval)

    except Exception as e:
        print(f"ERROR: Error during Q-CTRL monitoring: {e!s}")
        logger.exception("Error during Q-CTRL monitoring")
        sys.exit(1)


def _initialize_watch_service() -> Any:  # noqa: ANN401
    """Initialize IBM Quantum Runtime Service for watch functions."""
    logger = setup_logging()
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService

        return QiskitRuntimeService()
    except ImportError:
        print("ERROR: IBM Quantum support requires the 'qiskit_ibm_runtime' package.")
        logger.exception("Missing qiskit_ibm_runtime package")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to initialize IBM Quantum Runtime Service: {e!s}")
        logger.exception("Failed to initialize IBM Quantum Runtime Service")
        sys.exit(1)


def _get_watch_job(service: Any, job_id: str) -> Any:  # noqa: ANN401
    """Get IBM Quantum job for watching."""
    logger = setup_logging()
    try:
        return service.job(job_id)
    except Exception as e:
        print(f"ERROR: Failed to retrieve job {job_id}: {e!s}")
        logger.exception("Failed to retrieve job %s", job_id)
        sys.exit(1)


def watch_job_status(job_id: str, interval: int = 10) -> None:
    """
    Continuously monitor a job's status.

    Args:
        job_id: The IBM Quantum job ID to monitor.
        interval: Seconds between status checks.
    """
    import time

    logger = setup_logging()

    try:
        # Authenticate with IBM Quantum
        print("Authenticating with IBM Quantum...")
        logger.info("Authenticating with IBM Quantum...")
        ibmq_authenticator = IBMQuantumAuthenticator()
        ibmq_authenticator.authenticate_runtime_service()

        # Get the runtime service
        service = _initialize_watch_service()

        # Get the job
        job = _get_watch_job(service, job_id)

        print(f"🔍 Monitoring job {job_id} (checking every {interval} seconds)")
        print("Press Ctrl+C to stop monitoring\n")

        last_status = None
        while True:
            try:
                current_status = job.status()
                timestamp = datetime.now(tz=UTC).strftime("%H:%M:%S")

                if current_status != last_status:
                    print(f"[{timestamp}] Status: {current_status}")
                    last_status = current_status

                    # Stop monitoring if job is complete
                    if current_status in ["DONE", "ERROR", "CANCELLED"]:
                        print(f"\n🏁 Job finished with status: {current_status}")
                        job_info = format_job_info(job)
                        print(job_info)
                        break

                time.sleep(interval)

            except KeyboardInterrupt:
                print("\n\n⏹️  Monitoring stopped by user.")
                break
            except Exception as e:  # noqa: BLE001
                print(f"[{timestamp}] Error checking status: {e!s}")
                time.sleep(interval)

    except Exception as e:
        error_message = f"Error during monitoring: {e!s}"
        print(f"ERROR: {error_message}")
        logger.exception(error_message)
        sys.exit(1)


def main() -> None:
    """Execute the job status checker main functionality."""
    parser = argparse.ArgumentParser(
        description="Check the status of an IBM Quantum job or Q-CTRL Qiskit Function job.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s 12345678-1234-5678-9abc-123456789def                    # Check IBM Quantum job
  %(prog)s 12345678-1234-5678-9abc-123456789def --qctrl            # Check Q-CTRL job
  %(prog)s 12345678-1234-5678-9abc-123456789def --watch            # Monitor IBM Quantum job
  %(prog)s 12345678-1234-5678-9abc-123456789def --qctrl --watch    # Monitor Q-CTRL job
  %(prog)s 12345678-1234-5678-9abc-123456789def --watch --interval 5
        """,
    )

    parser.add_argument(
        "job_id",
        help="Job ID to check (IBM Quantum or Q-CTRL)",
    )

    parser.add_argument(
        "--qctrl",
        action="store_true",
        help="Check Q-CTRL Qiskit Function job instead of IBM Quantum Runtime job",
    )

    parser.add_argument(
        "--watch",
        action="store_true",
        help="Continuously monitor the job status until completion",
    )

    parser.add_argument(
        "--interval",
        type=int,
        default=10,
        help="Seconds between status checks when watching (default: 10)",
    )

    args = parser.parse_args()

    # Validate job ID format (UUID validation)
    if not validate_job_id(args.job_id):
        print(f"ERROR: Invalid job ID format. Expected a valid UUID, got: {args.job_id}")
        print(
            "Example valid formats: 12345678-1234-5678-9abc-123456789def "
            "or 123456781234567889abc123456789def",
        )
        sys.exit(1)

    # Route to appropriate function based on job type
    if args.qctrl:
        if args.watch:
            watch_qctrl_job_status(args.job_id, args.interval)
        else:
            check_qctrl_job_status(args.job_id)
    elif args.watch:
        watch_job_status(args.job_id, args.interval)
    else:
        check_job_status(args.job_id)


if __name__ == "__main__":
    main()
