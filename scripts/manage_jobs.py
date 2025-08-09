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
from quantumnematode.executors.ibm_job_status import IBMJobStatus, IBMJobStatusInfo

TRUNCATE_ERROR_LENGTH = 100


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

        if (
            IBMJobStatus.is_queued_or_initializing(status)
            or status == IBMJobStatus.VALIDATING
            or IBMJobStatus.is_running(status)
        ):
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
        description = IBMJobStatusInfo.get_description(status)
        info += f"\nDescription: {description}"

        # Add error message if status is ERROR
        if status == IBMJobStatus.ERROR:
            try:
                error_msg = job.error_message()
                if error_msg:
                    info += f"\nError Message: {error_msg}"
            except Exception:  # noqa: BLE001, S110
                # Silently continue if error_message() is not available
                pass

    except Exception as e:  # noqa: BLE001
        return f"Error retrieving Q-CTRL job information: {e!s}"
    else:
        return info


def _get_qctrl_catalog() -> Any:  # noqa: ANN401
    """Get Q-CTRL functions catalog."""
    logger = setup_logging()
    ibmq_authenticator = IBMQuantumAuthenticator()

    try:
        return ibmq_authenticator.get_functions_catalog()
    except Exception as e:
        print(f"ERROR: Failed to get Q-CTRL functions catalog: {e!s}")
        logger.exception("Failed to get Q-CTRL functions catalog")
        sys.exit(1)


def _get_qctrl_job(catalog: Any, job_id: str) -> Any:  # noqa: ANN401
    """Get Q-CTRL job by ID."""
    logger = setup_logging()
    try:
        job = catalog.get_job_by_id(job_id)
        if job is None:
            print("ERROR: Retrieved job is None")
            sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to retrieve Q-CTRL job {job_id}: {e!s}")
        print("Please verify the job ID is correct and you have access to it.")
        logger.exception("Failed to retrieve Q-CTRL job %s", job_id)
        logger.exception("Please verify the job ID is correct and you have access to it.")
        sys.exit(1)
    else:
        return job


def _print_qctrl_status_message(status: str) -> None:
    """Print appropriate status message for Q-CTRL jobs."""
    if status == IBMJobStatus.DONE:
        print("\n✅ Q-CTRL job completed successfully!")
        print("💡 You can retrieve results with: job.result()")
    elif status == IBMJobStatus.ERROR:
        print("\n❌ Q-CTRL job completed with errors!")
        print("💡 Check the error details above or use: job.result()")
    elif IBMJobStatus.is_queued_or_initializing(status):
        print(f"\n⏳ Q-CTRL job is {status.lower()}...")
        print("💡 Check again later or use --watch flag for continuous monitoring")
    elif IBMJobStatus.is_running(status):
        print(f"\n🏃 Q-CTRL job is running: {status}")
        print("💡 Check again later or use --watch flag for continuous monitoring")
    elif status in (IBMJobStatus.CANCELED, IBMJobStatus.CANCELLED):
        print("\n⛔ Q-CTRL job was cancelled")
    else:
        print(f"\n📋 Q-CTRL job status: {status}")


def list_qctrl_jobs(limit: int = 10) -> None:  # noqa: C901
    """
    List recent Q-CTRL Qiskit Function jobs.

    Args:
        limit: Maximum number of jobs to display.
    """
    logger = setup_logging()

    try:
        # Authenticate with IBM Quantum and get Q-CTRL catalog
        print("Authenticating with IBM Quantum for Q-CTRL access...")
        logger.info("Authenticating with IBM Quantum for Q-CTRL access...")

        catalog = _get_qctrl_catalog()

        print("Successfully authenticated. Retrieving job list...")
        logger.info("Successfully authenticated. Retrieving job list...")

        # Get jobs list
        jobs = catalog.jobs()

        if not jobs:
            print("\n📭 No Q-CTRL jobs found.")
            return

        # Limit the number of jobs to display
        jobs_to_show = jobs[:limit]
        total_jobs = len(jobs)

        print(f"\n📋 Q-CTRL Jobs (showing {len(jobs_to_show)} of {total_jobs}):")
        print("=" * 60)

        for i, job in enumerate(jobs_to_show, 1):
            try:
                job_id = job.job_id
                status = job.status()
                description = IBMJobStatusInfo.get_description(status)

                print(f"{i:2d}. Job ID: {job_id}")
                print(f"    Status: {status} - {description}")

                # Add error message if available
                if status == IBMJobStatus.ERROR:
                    try:
                        error_msg = job.error_message()
                        if error_msg:
                            # Truncate long error messages for list view
                            if len(error_msg) > TRUNCATE_ERROR_LENGTH:
                                error_msg = error_msg[:97] + "..."
                            print(f"    Error: {error_msg}")
                    except Exception:  # noqa: BLE001, S110
                        pass

                print()  # Empty line between jobs

            except Exception as e:  # noqa: BLE001
                print(f"{i:2d}. Job ID: {job.job_id} - Error retrieving details: {e!s}")
                print()

        if total_jobs > limit:
            print(f"💡 Showing {limit} most recent jobs. Use --limit to see more.")

    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"ERROR: Failed to list Q-CTRL jobs: {e!s}")
        logger.exception("Failed to list Q-CTRL jobs")
        sys.exit(1)


def cancel_qctrl_job(job_id: str, *, force: bool = False) -> None:
    """
    Cancel a Q-CTRL Qiskit Function job.

    Args:
        job_id: The Q-CTRL job ID to cancel.
        force: Skip confirmation prompt if True.
    """
    logger = setup_logging()

    try:
        # Authenticate with IBM Quantum and get Q-CTRL catalog
        print("Authenticating with IBM Quantum for Q-CTRL access...")
        logger.info("Authenticating with IBM Quantum for Q-CTRL access...")

        catalog = _get_qctrl_catalog()

        print(f"Successfully authenticated. Retrieving job {job_id}...")
        logger.info("Successfully authenticated. Retrieving job %s...", job_id)

        # Get the job from the catalog
        job = _get_qctrl_job(catalog, job_id)

        # Check current status
        current_status = job.status()
        print(f"\nCurrent job status: {current_status}")

        # Check if job can be cancelled
        if IBMJobStatus.is_terminal(current_status):
            print(f"❌ Cannot cancel job - it has already completed with status: {current_status}")
            return

        # Get confirmation unless force is specified
        if not force:
            response = input(f"\n⚠️  Are you sure you want to cancel job {job_id}? (y/N): ")
            if response.lower() not in ("y", "yes"):
                print("❌ Cancellation aborted.")
                return

        # Cancel the job
        print(f"\n🛑 Cancelling job {job_id}...")
        logger.info("Cancelling job %s...", job_id)

        result = job.stop()

        print(f"✅ {result}")
        logger.info("Job cancellation result: %s", result)

        # Check new status
        new_status = job.status()
        print(f"New status: {new_status}")

    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"ERROR: Failed to cancel Q-CTRL job: {e!s}")
        logger.exception("Failed to cancel Q-CTRL job")
        sys.exit(1)


def retrieve_qctrl_job_result(job_id: str) -> None:
    """
    Retrieve and display the result of a Q-CTRL Qiskit Function job.

    Args:
        job_id: The Q-CTRL job ID to retrieve results for.
    """
    logger = setup_logging()

    try:
        # Authenticate with IBM Quantum and get Q-CTRL catalog
        print("Authenticating with IBM Quantum for Q-CTRL access...")
        logger.info("Authenticating with IBM Quantum for Q-CTRL access...")

        catalog = _get_qctrl_catalog()

        print(f"Successfully authenticated. Retrieving job {job_id}...")
        logger.info("Successfully authenticated. Retrieving job %s...", job_id)

        # Get the job from the catalog
        job = _get_qctrl_job(catalog, job_id)

        # Display job information first
        job_info = format_qctrl_job_info(job)
        print(job_info)

        # Check status and handle accordingly
        status = job.status()

        if status == IBMJobStatus.DONE:
            print("\n📊 Retrieving job result...")
            try:
                result = job.result()
                print("✅ Result retrieved successfully!")
                print(f"Result type: {type(result)}")
                print(f"Result: {result}")
            except Exception as e:
                print(f"❌ Error retrieving result: {e!s}")
                logger.exception("Error retrieving result for job %s", job_id)

        elif status == IBMJobStatus.ERROR:
            print("\n❌ Job completed with error.")
            try:
                error_msg = job.error_message()
                if error_msg:
                    print(f"Error message: {error_msg}")
                else:
                    print("No detailed error message available.")
            except Exception as e:
                print(f"Error retrieving error message: {e!s}")
                logger.exception("Error retrieving error message for job %s", job_id)
        elif IBMJobStatus.is_terminal(status):
            print(f"\n⚠️  Job completed with status {status} - no result available.")
        else:
            print(f"\n⏳ Job is still {status.lower()} - no result available yet.")
            print("💡 Use --watch flag to monitor until completion.")

    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"ERROR: Failed to retrieve Q-CTRL job result: {e!s}")
        logger.exception("Failed to retrieve Q-CTRL job result")
        sys.exit(1)


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

        catalog = _get_qctrl_catalog()

        print(f"Successfully authenticated. Checking Q-CTRL job {job_id}...")
        logger.info("Successfully authenticated. Checking Q-CTRL job %s...", job_id)

        # Get the job from the catalog
        job = _get_qctrl_job(catalog, job_id)

        # Display job information
        job_info = format_qctrl_job_info(job)
        print(job_info)

        # Additional actions based on status
        status = job.status()
        _print_qctrl_status_message(status)

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


def list_ibm_jobs(limit: int = 10) -> None:
    """
    List recent IBM Quantum Runtime jobs.

    Args:
        limit: Maximum number of jobs to display.
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

        print("Successfully authenticated. Retrieving job list...")
        logger.info("Successfully authenticated. Retrieving job list...")

        # Get jobs list
        jobs = service.jobs(limit=limit)

        if not jobs:
            print("\n📭 No IBM Quantum jobs found.")
            return

        print(f"\n📋 IBM Quantum Jobs (showing up to {limit}):")
        print("=" * 60)

        for i, job in enumerate(jobs, 1):
            try:
                job_id = job.job_id()
                status = job.status()
                backend_name = job.backend().name if job.backend() else "Unknown"
                description = IBMJobStatusInfo.get_description(status)

                print(f"{i:2d}. Job ID: {job_id}")
                print(f"    Status: {status} - {description}")
                print(f"    Backend: {backend_name}")

                # Add creation time if available
                creation_str = _get_creation_time_str(job)
                if creation_str != "Unknown":
                    print(f"    Created: {creation_str}")

                print()  # Empty line between jobs

            except Exception as e:  # noqa: BLE001
                try:
                    job_id = job.job_id()
                    print(f"{i:2d}. Job ID: {job_id} - Error retrieving details: {e!s}")
                except Exception:  # noqa: BLE001
                    print(f"{i:2d}. Job - Error retrieving details: {e!s}")
                print()

    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"ERROR: Failed to list IBM Quantum jobs: {e!s}")
        logger.exception("Failed to list IBM Quantum jobs")
        sys.exit(1)


def cancel_ibm_job(job_id: str, *, force: bool = False) -> None:
    """
    Cancel an IBM Quantum Runtime job.

    Args:
        job_id: The IBM Quantum job ID to cancel.
        force: Skip confirmation prompt if True.
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

        print(f"Successfully authenticated. Retrieving job {job_id}...")
        logger.info("Successfully authenticated. Retrieving job %s...", job_id)

        # Get the job
        job = _get_ibm_job(service, job_id)

        # Check current status
        current_status = job.status()
        print(f"\nCurrent job status: {current_status}")

        # Check if job can be cancelled
        if IBMJobStatus.is_terminal(current_status):
            print(f"❌ Cannot cancel job - it has already completed with status: {current_status}")
            return

        # Get confirmation unless force is specified
        if not force:
            response = input(f"\n⚠️  Are you sure you want to cancel job {job_id}? (y/N): ")
            if response.lower() not in ("y", "yes"):
                print("❌ Cancellation aborted.")
                return

        # Cancel the job
        print(f"\n🛑 Cancelling job {job_id}...")
        logger.info("Cancelling job %s...", job_id)

        result = job.cancel()

        print("✅ Job cancellation requested.")
        logger.info("Job cancellation result: %s", result)

        # Check new status
        new_status = job.status()
        print(f"New status: {new_status}")

    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"ERROR: Failed to cancel IBM Quantum job: {e!s}")
        logger.exception("Failed to cancel IBM Quantum job")
        sys.exit(1)


def retrieve_ibm_job_result(job_id: str) -> None:
    """
    Retrieve and display the result of an IBM Quantum Runtime job.

    Args:
        job_id: The IBM Quantum job ID to retrieve results for.
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

        print(f"Successfully authenticated. Retrieving job {job_id}...")
        logger.info("Successfully authenticated. Retrieving job %s...", job_id)

        # Get the job
        job = _get_ibm_job(service, job_id)

        # Display job information first
        job_info = format_job_info(job)
        print(job_info)

        # Check status and handle accordingly
        status = job.status()

        if status == IBMJobStatus.DONE:
            print("\n📊 Retrieving job result...")
            try:
                result = job.result()
                print("✅ Result retrieved successfully!")
                print(f"Result type: {type(result)}")
                print(f"Result: {result}")
            except Exception as e:
                print(f"❌ Error retrieving result: {e!s}")
                logger.exception("Error retrieving result for job %s", job_id)

        elif status == IBMJobStatus.ERROR:
            print("\n❌ Job completed with error.")
            try:
                # Try to get error message from result
                result = job.result()
                print(f"Error details: {result}")
            except Exception as e:
                print(f"Error retrieving error details: {e!s}")
                logger.exception("Error retrieving error details for job %s", job_id)
        elif IBMJobStatus.is_terminal(status):
            print(f"\n⚠️  Job completed with status {status} - no result available.")
        else:
            print(f"\n⏳ Job is still {status.lower()} - no result available yet.")
            print("💡 Use --watch flag to monitor until completion.")

    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"ERROR: Failed to retrieve IBM Quantum job result: {e!s}")
        logger.exception("Failed to retrieve IBM Quantum job result")
        sys.exit(1)


def _print_status_message(status: str) -> None:
    """Print appropriate status message based on job status."""
    if status == IBMJobStatus.DONE:
        print("\n✅ Job completed successfully!")
        print("💡 You can retrieve results with: job.result()")
    elif status == IBMJobStatus.ERROR:
        print("\n❌ Job completed with errors!")
        print("💡 Check the error details above or use: job.result()")
    elif IBMJobStatus.is_queued_or_initializing(status) or status == IBMJobStatus.VALIDATING:
        print(f"\n⏳ Job is {status.lower()}...")
        print("💡 Check again later or use --watch flag for continuous monitoring")
    elif IBMJobStatus.is_running(status):
        print("\n🏃 Job is currently running...")
        print("💡 Check again later or use --watch flag for continuous monitoring")
    elif status in (IBMJobStatus.CANCELED, IBMJobStatus.CANCELLED):
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

        catalog = _get_qctrl_catalog()

        # Get the job
        job = _get_qctrl_job(catalog, job_id)

        print(f"🔍 Monitoring Q-CTRL job {job_id} (checking every {interval} seconds)")
        print("Press Ctrl+C to stop monitoring\n")

        last_status = None
        timestamp = datetime.now(tz=UTC).strftime("%H:%M:%S")  # Initialize timestamp
        while True:
            try:
                current_status = job.status()
                timestamp = datetime.now(tz=UTC).strftime("%H:%M:%S")

                if current_status != last_status:
                    print(f"[{timestamp}] Status: {current_status}")
                    last_status = current_status

                    # Stop monitoring if job is complete
                    if IBMJobStatus.is_terminal(current_status):
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
        timestamp = datetime.now(tz=UTC).strftime("%H:%M:%S")  # Initialize timestamp
        while True:
            try:
                current_status = job.status()
                timestamp = datetime.now(tz=UTC).strftime("%H:%M:%S")

                if current_status != last_status:
                    print(f"[{timestamp}] Status: {current_status}")
                    last_status = current_status

                    # Stop monitoring if job is complete
                    if IBMJobStatus.is_terminal(current_status):
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


def main() -> None:  # noqa: C901, PLR0912, PLR0915
    """Execute the job status checker main functionality."""
    parser = argparse.ArgumentParser(
        description="Manage IBM Quantum jobs and Q-CTRL Qiskit Function jobs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Check job status:
    %(prog)s 12345678-1234-5678-9abc-123456789def                    # Check IBM Quantum job
    %(prog)s 12345678-1234-5678-9abc-123456789def --qctrl            # Check Q-CTRL job

  Monitor job status:
    %(prog)s 12345678-1234-5678-9abc-123456789def --watch            # Monitor IBM Quantum job
    %(prog)s 12345678-1234-5678-9abc-123456789def --qctrl --watch    # Monitor Q-CTRL job
    %(prog)s 12345678-1234-5678-9abc-123456789def --watch --interval 5

  List jobs:
    %(prog)s --list                                                  # List IBM Quantum jobs
    %(prog)s --list --qctrl                                          # List Q-CTRL jobs
    %(prog)s --list --limit 20                                       # List up to 20 jobs

  Cancel jobs:
    %(prog)s 12345678-1234-5678-9abc-123456789def --cancel           # Cancel IBM job (with prompt)
    %(prog)s 12345678-1234-5678-9abc-123456789def --cancel --qctrl   # Cancel Q-CTRL job
    %(prog)s 12345678-1234-5678-9abc-123456789def --cancel --force   # Cancel without prompt

  Retrieve results:
    %(prog)s 12345678-1234-5678-9abc-123456789def --result           # Get IBM job result
    %(prog)s 12345678-1234-5678-9abc-123456789def --result --qctrl   # Get Q-CTRL job result
        """,
    )

    parser.add_argument(
        "job_id",
        nargs="?",
        help="Job ID to check, cancel, or retrieve (IBM Quantum or Q-CTRL)",
    )

    parser.add_argument(
        "--qctrl",
        action="store_true",
        help="Target Q-CTRL Qiskit Function jobs instead of IBM Quantum Runtime jobs",
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

    parser.add_argument(
        "--list",
        action="store_true",
        help="List recent jobs",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of jobs to show when listing (default: 10)",
    )

    parser.add_argument(
        "--cancel",
        action="store_true",
        help="Cancel the specified job",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip confirmation prompt when cancelling (use with --cancel)",
    )

    parser.add_argument(
        "--result",
        action="store_true",
        help="Retrieve and display the job result",
    )

    args = parser.parse_args()

    # Validate arguments
    action_count = sum([args.watch, args.list, args.cancel, args.result])

    if action_count > 1:
        print(
            "ERROR: Only one action can be specified at a time "
            "(--watch, --list, --cancel, or --result)",
        )
        sys.exit(1)

    if args.list:
        # List jobs doesn't require job_id
        pass
    elif not args.job_id:
        print("ERROR: Job ID is required unless using --list")
        parser.print_help()
        sys.exit(1)
    elif not validate_job_id(args.job_id):
        print(f"ERROR: Invalid job ID format. Expected a valid UUID, got: {args.job_id}")
        print(
            "Example valid formats: 12345678-1234-5678-9abc-123456789def "
            "or 123456781234567889abc123456789def",
        )
        sys.exit(1)

    if args.force and not args.cancel:
        print("ERROR: --force can only be used with --cancel")
        sys.exit(1)

    # Route to appropriate function based on action and job type
    try:
        if args.list:
            if args.qctrl:
                list_qctrl_jobs(args.limit)
            else:
                list_ibm_jobs(args.limit)
        elif args.cancel:
            if args.qctrl:
                cancel_qctrl_job(args.job_id, force=args.force)
            else:
                cancel_ibm_job(args.job_id, force=args.force)
        elif args.result:
            if args.qctrl:
                retrieve_qctrl_job_result(args.job_id)
            else:
                retrieve_ibm_job_result(args.job_id)
        elif args.watch:
            if args.qctrl:
                watch_qctrl_job_status(args.job_id, args.interval)
            else:
                watch_job_status(args.job_id, args.interval)
        # Default action: check status
        elif args.qctrl:
            check_qctrl_job_status(args.job_id)
        else:
            check_job_status(args.job_id)
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
        sys.exit(0)


if __name__ == "__main__":
    main()
